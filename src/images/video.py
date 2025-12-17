import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm, wandb, hydra, tqdm
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d) # Jepa Projector

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1):
        self.V = V
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)



@hydra.main(version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="LeJEPA", config=dict(cfg))
    torch.manual_seed(0)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=8)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 100)).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    # gradient accumulation: number of batches to accumulate before stepping
    accum_steps = getattr(cfg, "accum_steps", 1)
    train_batches = len(train)
    # how many optimizer steps per epoch when using accumulation
    train_steps_per_epoch = (train_batches + accum_steps - 1) // accum_steps
    warmup_steps = train_steps_per_epoch
    total_steps = train_steps_per_epoch * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    t_max = max(1, total_steps - warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled="cuda" == "cuda")
    # Training
    for epoch in range(cfg.epochs):
        net.train(), probe.train()
        # zero grads before accumulating
        opt.zero_grad()
        accum_counter = 0
        for vs, y in tqdm.tqdm(train, total=len(train)):
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss

                # average loss across accumulation steps
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            accum_counter += 1

            # step optimizer when we've accumulated enough gradients
            if accum_counter >= accum_steps:
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad()
                accum_counter = 0

            wandb.log(
                {
                    "train/probe": probe_loss.item(),
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                }
            )

        # final step for leftover gradients if train size not divisible by accum_steps
        if accum_counter > 0:
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt.zero_grad()

        # Evaluation
        net.eval(), probe.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
        wandb.log({"test/acc": correct / len(test_ds), "test/epoch": epoch})
    wandb.finish()


if __name__ == "__main__":
    main()