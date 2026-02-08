# LeJEPA SSL — Implementation Plan for Improvements

## Analysis of Current Weaknesses

After a thorough review of the full pipeline, here are the identified issues and improvement opportunities, ordered by impact.

---

### 1. 🔴 Augmentation Strategy — OneOf Bottleneck

**Problem**: The current augmentation uses `RandomApply(OneOf([...]), p=0.8)` which selects **at most one** augmentation per view. This means most views differ from the anchor by only a single transform, producing weak invariance signal. SSL methods like SimCLR/BYOL show that **composing multiple augmentations** is critical for learning robust features.

**Impact**: The encoder may learn shallow invariances (e.g., invariant to scaling but not simultaneously to temporal masking + noise).

**Fix**: Switch from `OneOf` to `Compose` with independent `RandomApply` per augmentation, so multiple transforms can stack.

---

### 2. 🔴 No Learning Rate Schedule

**Problem**: The optimizer uses a flat learning rate (AdamW, lr=3e-4) with no warmup or decay. SSL methods universally benefit from cosine decay + linear warmup.

**Impact**: Training may diverge early (no warmup) and overfit later (no decay). SIGReg is particularly sensitive to learning rate since it regularizes distribution shape.

**Fix**: Add cosine annealing with linear warmup.

---

### 3. 🔴 View Builder 

**Problem**: The view builder builds a t-1 and a t-2 view when we choose include_prev 2. It should add 2 t-1 aug views

**Impact**: We reduce the implicit time encoding in the representations.

**Fix**: Add include_prev amount of t-1 augmented views and remove the t-2 and potentiolly t-x views.

---

### 4. 🟡 Validation DataLoader Shuffles

**Problem**: `val_dataloader` has `shuffle=True` with a comment "needs to be for sigreg". This is incorrect for validation — SIGReg should be computed on whatever mini-batch arrives. Shuffling validation makes metrics non-reproducible.

**Impact**: Validation metrics are noisy and non-reproducible across runs.

**Fix**: Set `shuffle=False` for validation. SIGReg works on the batch regardless of ordering.

---

### 5. 🟡 Augmentation Applied Per-Sample, Not Per-Batch

**Problem**: In `AugmentationViewBuilder.build_views()`, the augmented views are created by repeating the window and applying the transform to the entire `[R, C, T]` batch at once. However, `OneOf` and `RandomApply` sample a single random choice applied uniformly to all R views in the batch.

**Impact**: All augmented views within a single sample may receive the identical transform, reducing view diversity.

**Fix**: Apply transforms individually per view, or ensure transforms sample randomly per-element in the batch dimension.

---

### 7. 🟡 Probe Trains Jointly (Even If Weighted Low)

**Problem**: The probe loss is added to the total loss with weight 0.1, meaning its gradients flow into the encoder (via the `probe_loss_weight * probe_loss` term in `training_step`). This contradicts the pure SSL goal.

**Impact**: The probe can subtly bias encoder representations toward forecasting, which may hurt other downstream tasks (classification, virtual sensing).

**Fix**: Either set `probe_loss_weight=0.0` (pure monitoring) or fully detach probe gradients. Currently `_probe_loss` does `torch.no_grad()` for encoding but the probe parameters still receive gradients — this is correct. However, the `probe_loss` itself contributes to the total loss which can affect the computational graph if the probe output depends on any shared parameters.

**Status**: The current code correctly detaches encoder from probe gradients. But setting `probe_loss_weight=0.0` makes it purely monitoring. The 0.1 weight only trains the probe itself.

---

### 9. 🟢 Time Features Ignored by Most Encoders

**Problem**: `time_features` are passed through the data pipeline but most encoders ignore them (TransformerEncoder has a pass-through, CNN/RNN don't use them at all).

**Impact**: Cyclical time information (hour, weekday, day-of-year) is wasted. For traffic data, time-of-day is a critical feature.

**Fix**: Incorporate time features via concatenation, addition, or FiLM conditioning in all encoders.

---

### 10. 🟢 No Multi-Dataset Support

**Problem**: The data pipeline is tightly coupled to PeMS08. The `PeMS08SSLDataModule` hardcodes the dataset class.

**Impact**: Cannot easily benchmark on other time series datasets.

**Fix**: Create a generic `TimeSeriesSSLDataModule` that accepts any `[C, T]` tensor + optional metadata.

---

## Implementation Plan

### Phase 1: Critical SSL Improvements (High Impact, Low Risk)

These changes directly improve representation quality without major architectural changes.

#### 1.1 Compound Augmentations
**File**: [src/timeseries/data/pems.py](src/timeseries/data/pems.py) — `build_ssl_transform()`

Change from:
```python
# Current: OneOf selects one augmentation
pipeline.append(RandomApply(OneOf(ops), p=p_transform))
```

To:
```python
# Proposed: Each augmentation applied independently
for op in ops:
    pipeline.append(RandomApply(op, p=p_per_aug))
```

Each augmentation fires independently with probability ~0.3-0.5, allowing multiple transforms to compose. This is the biggest single improvement to SSL quality.

**Config additions**:
```python
'augmentation_mode': 'compound',  # 'compound' or 'oneof' (legacy)
'p_per_aug': 0.4,  # Per-augmentation probability in compound mode
```

#### 1.2 Cosine LR Schedule with Warmup
**File**: [src/timeseries/train/lightning_ssl.py](src/timeseries/train/lightning_ssl.py)

Add to `configure_optimizers()`:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, total_iters=warmup_steps
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup, scheduler], milestones=[warmup_steps]
)
```

**Config additions**:
```python
'warmup_epochs': 5,
'min_lr': 1e-6,
```

#### 1.3 Fix Validation Shuffle
**File**: [src/timeseries/data/pems.py](src/timeseries/data/pems.py) — `PeMS08SSLDataModule.val_dataloader()`

```python
# Change shuffle=True to shuffle=False
def val_dataloader(self):
    return DataLoader(self.val_ds, ..., shuffle=False, ...)
```

#### 1.4 Per-View Augmentation Diversity
**File**: [src/timeseries/data/view_builders.py](src/timeseries/data/view_builders.py)

Change from batch-applying transforms to individual application:
```python
# Current: applies same random choice to all views
t0_batch = window.unsqueeze(0).repeat(self.repeat_factor, 1, 1)
t0_augmented = self.transform(t0_batch)

# Proposed: apply transform individually per view
t0_augmented = torch.stack([
    self.transform(window.clone().unsqueeze(0)).squeeze(0)
    for _ in range(self.repeat_factor)
])
```


### Phase 5: Framework Generalization

#### 5.1 Generic DataModule
**New file**: `src/timeseries/data/generic_datamodule.py`

```python
class TimeSeriesSSLDataModule(L.LightningDataModule):
    """Generic SSL DataModule accepting any [C, T] tensor."""
    
    def __init__(self, data, timestamps=None, cfg=None, ...):
        ...
```

#### 5.2 Downstream Task Heads
Extend beyond forecast probe:

| Task | Head | Input | Output |
|------|------|-------|--------|
| Forecasting | `ForecastProbe` (exists) | `[B, D]` | `[B, C, H]` |
| Virtual Sensing | `SensorProbe` (new) | `[B, D]` w/ masked channels | `[B, C_masked, T]` |
| Classification | `ClassificationProbe` (new) | `[B, D]` | `[B, num_classes]` |
| Anomaly Detection | `AnomalyProbe` (new) | `[B, D]` | `[B, 1]` (score) |

#### 5.3 Downstream Evaluation Protocol
**New file**: `src/timeseries/tasks/eval_protocol.py`

Standardized evaluation:
1. Train SSL encoder (freeze)
2. Train linear probe → report linear probe accuracy
3. Train MLP probe → report MLP probe accuracy
4. Fine-tune encoder → report fine-tuned accuracy

---

## Implementation Priority Matrix

| Priority | Task | Phase | Estimated Effort | Impact |
|----------|------|-------|------------------|--------|
| **P0** | Compound augmentations | 1.1 | 1 hour | 🔴 High |
| **P0** | Cosine LR + warmup | 1.2 | 1 hour | 🔴 High |
| **P0** | Fix val shuffle | 1.3 | 5 min | 🟡 Medium |
| **P0** | Per-view augmentation | 1.4 | 30 min | 🟡 Medium |
| **P3** | Generic DataModule | 5.1 | 2 hours | 🟢 Low (now) |
| **P3** | Task heads | 5.2 | 3 hours | 🟢 Low (now) |
| **P3** | Time features in encoders | — | 2 hours | 🟢 Low |Time features in encoders (marked Low)

---

## Proposed Config After Phase 1-2

```python
cfg = DictConfig({
    # ... existing params ...
    
    # Phase 1: Augmentation
    'augmentation_mode': 'compound',  # 'compound' or 'oneof'
    'p_per_aug': 0.4,
    
    # Phase 1: LR Schedule
    'warmup_epochs': 5,
    'min_lr': 1e-6,
    'scheduler': 'cosine',  # 'cosine', 'none'
})
```

---

## Summary

The current framework has a solid modular design. The most impactful changes are:

1. **Compound augmentations** — stack multiple transforms instead of OneOf
2. **LR scheduling** — cosine decay with warmup

These changes, especially Phase 1 (quick wins), should meaningfully improve representation quality for all downstream tasks.
