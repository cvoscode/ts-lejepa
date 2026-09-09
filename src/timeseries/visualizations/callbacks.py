import os
import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import logging

import pacmap
from torch.utils.data import DataLoader, Subset

from ..core.types import Batch

class VisualizationCallback(L.Callback):
    def __init__(
        self,
        num_samples_plot=3,
        pacmap_every_n_epochs=1,
        num_pacmap_batches=10,
        sensor_indices=None,
        sample_indices=None,
        fixed_samples=False,
        fixed_window_index=None,
        fixed_window_stage="both",
    ):
        """
        Args:
            num_samples_plot: Number of samples (time series) to plot.
            pacmap_every_n_epochs: PaCMAP is expensive, so only run it every N epochs.
            num_pacmap_batches: How many batches to collect for PaCMAP (more = more accurate, but slower).
            sensor_indices: Optional list of sensor indices to plot (fixed order).
            sample_indices: Optional list of batch sample indices to plot (fixed order).
            fixed_samples: If True, use the first N samples each time instead of random.
            fixed_window_index: If set, always visualize the same dataset index.
            fixed_window_stage: "train", "val", or "both" (default) for fixed window.
        """
        super().__init__()
        self.num_samples_plot = num_samples_plot
        self.pacmap_every_n_epochs = pacmap_every_n_epochs
        self.num_pacmap_batches = num_pacmap_batches
        self.sensor_indices = sensor_indices
        self.sample_indices = sample_indices
        self.fixed_samples = fixed_samples
        self.fixed_window_index = fixed_window_index
        self.fixed_window_stage = str(fixed_window_stage).lower()

    def _get_fixed_batch(self, dataloader: DataLoader, stage: str):
        if self.fixed_window_index is None:
            return None
        stage = str(stage).lower()
        if self.fixed_window_stage not in {"both", stage}:
            return None
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None:
            return None
        index = int(self.fixed_window_index)
        if index < 0 or index >= len(dataset):
            return None
        subset = Subset(dataset, [index])
        collate_fn = getattr(dataloader, "collate_fn", None)
        fixed_loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        try:
            return next(iter(fixed_loader))
        except Exception:
            return None

    def _plot_data_from_batch(self, batch, trainer, pl_module):
        vs, target, view_times, future_times = self._unpack_batch(batch)

        device = pl_module.device
        vs = vs.to(device)
        if target is not None:
            target = target.to(device)
        if view_times is not None:
            view_times = view_times.to(device)
        if future_times is not None:
            future_times = future_times.to(device)

        if vs.dim() == 5:
            B, V, C, L, F = vs.shape
        else:
            B, V, C, L = vs.shape
        if V < 2:
            raise ValueError(f"Expected at least 2 views (prev/curr). Got V={V}.")
        vs_flat = vs.reshape(B * V, *vs.shape[2:])

        if view_times is not None:
            view_times_flat = view_times.reshape(B * V, *view_times.shape[2:])
            emb_all = self._backbone_forward(pl_module, vs_flat, view_times_flat)
        else:
            emb_all = self._backbone_forward(pl_module, vs_flat)

        if emb_all.dim() == 3:
            BV, T, D = emb_all.shape
            emb_views = emb_all.view(B, V, T, D)
            emb_t0 = emb_views[:, 1:-1, :, :].mean(dim=1)  # [B, T, D]
            emb_flat = emb_t0.mean(dim=1)  # [B, D]
        else:
            BV, D = emb_all.shape
            emb_views = emb_all.view(B, V, D)
            emb_flat = emb_views[:, 1:-1, :].mean(dim=1)  # [B, D]

        if getattr(pl_module, "use_covariates", False) and future_times is not None and hasattr(pl_module, "future_cov_encoder"):
            future_cov_emb = pl_module.future_cov_encoder(future_times)
            forecast_input = torch.cat([emb_flat, future_cov_emb], dim=-1)
        else:
            forecast_input = emb_flat

        yhat = self._forecast_from_module(pl_module, forecast_input, future_times)
        if target is None or yhat is None:
            return None

        if vs.dim() == 5:
            input_seq = vs[:, 1:-1, :, :, :].mean(dim=1).mean(dim=-1)  # [B, C, L]
        else:
            input_seq = vs[:, 1:-1, :, :].mean(dim=1)  # [B, C, L]

        if target.shape[1] == pl_module.horizon and target.shape[2] == pl_module.input_dim:
            target = target.transpose(1, 2)

        scaler = getattr(pl_module, "scaler", None)
        if scaler is None and trainer is not None and getattr(trainer, "datamodule", None) is not None:
            scaler = getattr(trainer.datamodule, "scaler", None)

        if scaler is not None:
            try:
                scaler = scaler.to(input_seq.device)
            except Exception:
                pass

            input_inv = scaler.inverse_transform(input_seq)
            target_inv = scaler.inverse_transform(target)
            yhat_inv = scaler.inverse_transform(yhat)
        else:
            input_inv = input_seq
            target_inv = target
            yhat_inv = yhat

        return {
            "input": input_inv.cpu(),
            "target": target_inv.cpu(),
            "prediction": yhat_inv.cpu(),
        }

    @staticmethod
    def _unpack_batch(batch):
        """Handle Batch dataclass and legacy tuple batches."""
        if isinstance(batch, Batch):
            return batch.views, batch.targets, batch.view_times, batch.future_times
        if isinstance(batch, (tuple, list)):
            if len(batch) == 4:
                return batch
            if len(batch) == 2:
                views, targets = batch
                return views, targets, None, None
        raise ValueError(f"Unsupported batch type for visualization: {type(batch)!r}")

    @staticmethod
    def _backbone_forward(pl_module, x, time_features=None):
        """Call backbone forward across SSLPretrainModule or LeJEPA_Forecaster."""
        if hasattr(pl_module, "_backbone_forward"):
            return pl_module._backbone_forward(x, time_features)
        if hasattr(pl_module, "ssl_core") and hasattr(pl_module.ssl_core, "encoder"):
            if time_features is None:
                return pl_module.ssl_core.encoder(x)
            try:
                return pl_module.ssl_core.encoder(x, time_features)
            except TypeError:
                return pl_module.ssl_core.encoder(x)
        raise AttributeError("No backbone forward method found on pl_module")

    @staticmethod
    def _forecast_from_module(pl_module, emb, future_times=None):
        """Forecast using model head or probe if available."""
        if hasattr(pl_module, "forecast_head"):
            if getattr(pl_module, "use_covariates", False) and future_times is not None:
                future_cov_emb = pl_module.future_cov_encoder(future_times)
                forecast_input = torch.cat([emb, future_cov_emb], dim=-1)
            else:
                forecast_input = emb
            return pl_module.forecast_head(forecast_input).view(
                emb.shape[0], pl_module.input_dim, pl_module.horizon
            )
        if hasattr(pl_module, "probe") and pl_module.probe is not None:
            return pl_module.probe(emb, future_times)
        return None

    @staticmethod
    def _ensure_pl_module_attrs(pl_module):
        """Ensure optional attributes exist on pl_module to avoid AttributeError in callbacks.

        Also try to fill in values from `probe` if available (common with SSLPretrainModule).
        """
        if not hasattr(pl_module, "horizon"):
            setattr(pl_module, "horizon", None)
        if not hasattr(pl_module, "input_dim"):
            setattr(pl_module, "input_dim", None)

        # If a probe is attached, use its dims to populate missing fields
        if getattr(pl_module, "horizon", None) is None and hasattr(pl_module, "probe") and pl_module.probe is not None:
            try:
                setattr(pl_module, "horizon", int(pl_module.probe.horizon))
            except Exception:
                pass
        if getattr(pl_module, "input_dim", None) is None and hasattr(pl_module, "probe") and pl_module.probe is not None:
            try:
                setattr(pl_module, "input_dim", int(pl_module.probe.output_channels))
            except Exception:
                pass

    @staticmethod
    def _log_figure(trainer, tag: str, fig) -> None:
        """Log a matplotlib figure to all supported loggers and save to disk if possible."""
        if trainer is None:
            return

        loggers = []
        if hasattr(trainer, "loggers") and trainer.loggers is not None:
            loggers = list(trainer.loggers)
        if not loggers and getattr(trainer, "logger", None) is not None:
            loggers = [trainer.logger]

        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_figure"):
                experiment.add_figure(tag, fig, global_step=trainer.global_step)
                if hasattr(experiment, "flush"):
                    try:
                        experiment.flush()
                    except Exception:
                        pass
            if experiment is not None and hasattr(experiment, "add_image"):
                try:
                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img = buf.reshape(h, w, 3).transpose(2, 0, 1)  # CHW
                    experiment.add_image(tag, img, global_step=trainer.global_step)
                except Exception:
                    pass

        # Save to disk for easier inspection
        primary_logger = loggers[0] if loggers else None
        log_dir = getattr(primary_logger, "log_dir", None)
        if log_dir:
            out_dir = os.path.join(log_dir, "figures")
            os.makedirs(out_dir, exist_ok=True)
            safe_tag = tag.replace("/", "_")
            out_path = os.path.join(out_dir, f"{safe_tag}_epoch{trainer.current_epoch}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")

    def on_validation_epoch_end(self, trainer, pl_module):
        # 1. Get data (take a subset of the val set)
        if trainer.datamodule is None:
            return
        
        val_loader = trainer.datamodule.val_dataloader()
        fixed_batch = self._get_fixed_batch(val_loader, "val")

        # Containers for PaCMAP
        all_embeddings = []
        all_times = [] # Time index for coloring
        
        # Container for Prediction Plots (only from the first batch)
        plot_data = None
        
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval() # Ensure eval mode
        # Ensure optional attrs exist to avoid AttributeError when inspecting targets
        self._ensure_pl_module_attrs(pl_module)

        try:
            with torch.no_grad():
                if fixed_batch is not None:
                    plot_data = self._plot_data_from_batch(fixed_batch, trainer, pl_module)
                for i, batch in enumerate(val_loader):
                    if i >= self.num_pacmap_batches:
                        break
                
                    # Batch extraction: handle Batch dataclass and legacy tuple formats
                    vs, target, view_times, future_times = self._unpack_batch(batch)
                        
                    vs = vs.to(device)
                    if target is not None:
                        target = target.to(device)
                    if view_times is not None:
                        view_times = view_times.to(device)
                    if future_times is not None:
                        future_times = future_times.to(device)
                    
                    # Forward Pass
                    # Get backbone embeddings from t0 view(s).
                    # Convention: view[0]=t-1, view[1:V-1]=repeated augmented t0 views, view[V-1]=t+1.
                    if vs.dim() == 5:
                        B, V, C, L, F = vs.shape
                    else:
                        B, V, C, L = vs.shape
                    if V < 3:
                        raise ValueError(f"Expected at least 3 views (prev, curr, next). Got V={V}.")
                    vs_flat = vs.reshape(B * V, *vs.shape[2:])
                    
                    if view_times is not None:
                        view_times_flat = view_times.reshape(B * V, *view_times.shape[2:])
                        emb_all = self._backbone_forward(pl_module, vs_flat, view_times_flat)
                    else:
                        emb_all = self._backbone_forward(pl_module, vs_flat)
                    
                    # Get t0 embeddings for PaCMAP and forecasting (derive shapes from emb_all)
                    if emb_all.dim() == 3:
                        BV, T, D = emb_all.shape
                        emb_views = emb_all.view(B, V, T, D)
                        emb_t0 = emb_views[:, 1:-1, :, :].mean(dim=1)  # [B, T, D]
                        emb_flat = emb_t0.mean(dim=1)  # [B, D]
                    else:
                        BV, D = emb_all.shape
                        emb_views = emb_all.view(B, V, D)
                        emb_flat = emb_views[:, 1:-1, :].mean(dim=1)  # [B, D]

                    # Generate forecast -- only use covariates if the module actually supports them
                    if getattr(pl_module, "use_covariates", False) and future_times is not None and hasattr(pl_module, "future_cov_encoder"):
                        future_cov_emb = pl_module.future_cov_encoder(future_times)
                        forecast_input = torch.cat([emb_flat, future_cov_emb], dim=-1)
                    else:
                        forecast_input = emb_flat

                    yhat = self._forecast_from_module(pl_module, forecast_input, future_times)
                    
                    # Collect embeddings for PaCMAP
                    all_embeddings.append(emb_flat.cpu().numpy())
                    
                    # Real time indices (robust to shuffling)
                    if isinstance(batch, Batch) and batch.time_index is not None:
                        all_times.append(batch.time_index.detach().cpu().numpy())
                    else:
                        batch_size = vs.shape[0]
                        time_indices = np.arange(i * batch_size, (i + 1) * batch_size)
                        all_times.append(time_indices)
                    
                    # Store plot data (only from the first batch)
                    if i == 0 and plot_data is None:
                        plot_data = self._plot_data_from_batch(batch, trainer, pl_module)
        finally:
            if was_training:
                pl_module.train()

        # --- A. Plot Predictions ---
        if plot_data is not None:
            try:
                self._plot_predictions(trainer, plot_data, step='Validation')
            except Exception:
                import traceback
                traceback.print_exc()

        # --- B. Plot PaCMAP (every N epochs) ---
        if (trainer.current_epoch + 1) % self.pacmap_every_n_epochs == 0 and len(all_embeddings) > 0:
            try:
                embeddings_concat = np.concatenate(all_embeddings, axis=0)
                times_concat = np.concatenate(all_times, axis=0)
                self._plot_pacmap(trainer, embeddings_concat, times_concat, step='Validation')
            except Exception:
                import traceback
                traceback.print_exc()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.datamodule is None:
           
            return
        
        train_loader = trainer.datamodule.train_dataloader()
        fixed_batch = self._get_fixed_batch(train_loader, "train")
        # Ensure optional attrs exist to avoid AttributeError when inspecting targets
        self._ensure_pl_module_attrs(pl_module)

        # Containers for PaCMAP
        all_embeddings = []
        all_times = [] # Time index for coloring

        # Container for Prediction Plots (only from the first batch)
        plot_data = None
        
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval() # Ensure eval mode

        try:
            with torch.no_grad():
                if fixed_batch is not None:
                    plot_data = self._plot_data_from_batch(fixed_batch, trainer, pl_module)
                for i, batch in enumerate(train_loader):
                    if i >= self.num_pacmap_batches:
                        break
                
                    # Batch extraction: handle Batch dataclass and legacy tuple formats
                    vs, target, view_times, future_times = self._unpack_batch(batch)
                        
                    vs = vs.to(device)
                    if target is not None:
                        target = target.to(device)
                    if view_times is not None:
                        view_times = view_times.to(device)
                    if future_times is not None:
                        future_times = future_times.to(device)
                    
                    # Forward Pass
                    # Get backbone embeddings from t0 view(s).
                    # Convention: view[0]=t-1, view[1:V-1]=repeated augmented t0 views, view[V-1]=t+1.
                    if vs.dim() == 5:
                        B, V, C, L, F = vs.shape
                    else:
                        B, V, C, L = vs.shape
                    if V < 3:
                        raise ValueError(f"Expected at least 3 views (prev, curr, next). Got V={V}.")
                    vs_flat = vs.reshape(B * V, *vs.shape[2:])
                    
                    if view_times is not None:
                        view_times_flat = view_times.reshape(B * V, *view_times.shape[2:])
                        emb_all = self._backbone_forward(pl_module, vs_flat, view_times_flat)
                    else:
                        emb_all = self._backbone_forward(pl_module, vs_flat)
                    
                    # Get t0 embeddings for PaCMAP and forecasting (derive shapes from emb_all)
                    if emb_all.dim() == 3:
                        BV, T, D = emb_all.shape
                        emb_views = emb_all.view(B, V, T, D)
                        emb_t0 = emb_views[:, 1:-1, :, :].mean(dim=1)  # [B, T, D]
                        emb_flat = emb_t0.mean(dim=1)  # [B, D]
                    else:
                        BV, D = emb_all.shape
                        emb_views = emb_all.view(B, V, D)
                        emb_flat = emb_views[:, 1:-1, :].mean(dim=1)  # [B, D]

                    # Generate forecast -- only use covariates if the module actually supports them
                    if getattr(pl_module, "use_covariates", False) and future_times is not None and hasattr(pl_module, "future_cov_encoder"):
                        future_cov_emb = pl_module.future_cov_encoder(future_times)
                        forecast_input = torch.cat([emb_flat, future_cov_emb], dim=-1)
                    else:
                        forecast_input = emb_flat

                    yhat = self._forecast_from_module(pl_module, forecast_input, future_times)
                    
                    # Collect embeddings for PaCMAP
                    all_embeddings.append(emb_flat.cpu().numpy())
                    
                    # Real time indices (robust to shuffling)
                    if isinstance(batch, Batch) and batch.time_index is not None:
                        all_times.append(batch.time_index.detach().cpu().numpy())
                    else:
                        batch_size = vs.shape[0]
                        time_indices = np.arange(i * batch_size, (i + 1) * batch_size)
                        all_times.append(time_indices)
                    
                    # Store plot data (only from the first batch)
                    if i == 0 and plot_data is None:
                        plot_data = self._plot_data_from_batch(batch, trainer, pl_module)
        finally:
            if was_training:
                pl_module.train()

        # --- A. Plot Predictions ---
        if plot_data is not None:
            
            self._plot_predictions(trainer, plot_data, step='Train')
               
            

        # --- B. Plot PaCMAP (every N epochs) ---
        if (trainer.current_epoch + 1) % self.pacmap_every_n_epochs == 0 and len(all_embeddings) > 0:
            embeddings_concat = np.concatenate(all_embeddings, axis=0)
            times_concat = np.concatenate(all_times, axis=0)
            self._plot_pacmap(trainer, embeddings_concat, times_concat,step='Train')
                
           

    def _plot_predictions(self, trainer, data, step):
        """Creates Matplotlib plots for time series."""
        inputs = data.get('input', None)
        targets = data.get('target', None)
        preds = data.get('prediction', None)

        def _to_tensor(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return x

        inputs = _to_tensor(inputs)
        targets = _to_tensor(targets)
        preds = _to_tensor(preds)

        if inputs is None:
            return

        # Ensure everything is [B, C, L]
        C = inputs.shape[1]

        if targets is not None and targets.ndim == 3 and targets.shape[1] != C:
            targets = targets.transpose(1, 2).contiguous()
        if preds is not None and preds.ndim == 3 and preds.shape[1] != C:
            preds = preds.transpose(1, 2).contiguous()

        num_samples = min(self.num_samples_plot, inputs.shape[0])

        if self.sample_indices is not None:
            sample_indices = [int(i) for i in self.sample_indices if 0 <= int(i) < inputs.shape[0]]
            sample_indices = sample_indices[:num_samples] if sample_indices else list(range(num_samples))
        elif self.fixed_samples or self.sensor_indices is not None:
            sample_indices = list(range(num_samples))
        else:
            sample_indices = np.random.choice(inputs.shape[0], size=num_samples, replace=False).tolist()

        if self.sensor_indices is None:
            sensor_indices = [int(np.random.randint(0, max(1, C)))]
        else:
            sensor_indices = [int(s) for s in self.sensor_indices if 0 <= int(s) < C]
            if not sensor_indices:
                return

        total_plots = len(sample_indices) * len(sensor_indices)
        fig, axes = plt.subplots(total_plots, 1, figsize=(10, 3.5 * total_plots), sharex=False)
        if total_plots == 1:
            axes = [axes]

        plot_idx = 0
        for sample_idx in sample_indices:
            for sensor_idx in sensor_indices:
                ax = axes[plot_idx]
                plot_idx += 1

                # Create time axes
                L_in = inputs.shape[-1]
                L_out = 0
                if targets is not None:
                    L_out = targets.shape[-1]
                elif preds is not None:
                    L_out = preds.shape[-1]

                t_in = range(0, L_in)
                t_out = range(L_in, L_in + L_out)  # Target follows input

                # Data for sample and chosen sensor
                seq_in = inputs[sample_idx, sensor_idx, :].cpu().numpy()
                seq_target = targets[sample_idx, sensor_idx, :].cpu().numpy() if targets is not None else None
                seq_pred = preds[sample_idx, sensor_idx, :].cpu().numpy() if preds is not None else None

                ax.plot(t_in, seq_in, label='Input History', color='gray', linestyle='--')
                if seq_target is not None and len(seq_target) > 0:
                    ax.plot(t_out, seq_target, label='Ground Truth', color='green')
                if seq_pred is not None and len(seq_pred) > 0:
                    ax.plot(t_out, seq_pred, label='Prediction', color='red')

                ax.set_title(f"Sample {sample_idx} - Sensor {sensor_idx}")
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        # Log to Tensorboard (and save to disk)
        self._log_figure(trainer, f"{step}/Predictions", fig)
        plt.close(fig)

    def _plot_pacmap(self, trainer, embeddings, time_indices,step):
        """ Computes and plots PaCMAP """
        try:
            # Flatten embeddings if necessary
            if embeddings.ndim > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)

            reducer = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=10,
                MN_ratio=0.5,
                FP_ratio=2.0,
                apply_pca=True,
                random_state=42,
                verbose=False,
            )
            # Silence PaCMAP's "random state is set to ..." info-style warning
            pacmap_logger = logging.getLogger("pacmap.pacmap")
            prev_level = pacmap_logger.level
            pacmap_logger.setLevel(logging.CRITICAL)
            try:
                embedding_2d = reducer.fit_transform(embeddings, init="pca")
            finally:
                pacmap_logger.setLevel(prev_level)

            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=time_indices,
                cmap='viridis',
                s=10,
                alpha=0.6
            )
            ax.set_title("PaCMAP of Embeddings (Color = Time Index)")
            plt.colorbar(scatter, label=f'Time Index within {step} Subset')
            ax.grid(True, alpha=0.3)

            self._log_figure(trainer, f"{step}/PaCMAP_Embedding", fig)
            plt.close(fig)

        except Exception as e:
            print(f"PaCMAP visualization failed: {e}")
