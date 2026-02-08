# LeJEPA SSL Framework — Architecture Documentation

## Overview

LeJEPA is a **self-supervised learning (SSL) framework for multivariate time series** that learns general-purpose representations without task-specific labels. The learned representations can then be used for downstream tasks such as:

- **Forecasting** — predict future sensor values
- **Virtual Sensing** — estimate missing/unobserved channels from observed ones
- **State Classification** — classify operational regimes or anomalies

The framework follows the **JEPA (Joint Embedding Predictive Architecture)** philosophy: create multiple *views* of the same data, encode them independently, and train the encoder so all views map to similar embeddings — while a regularizer (SIGReg) prevents collapse to trivial solutions.

---

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW TIME SERIES                              │
│                     PeMS08: [T=17,856 × C=170]                      │
│                     5-min intervals, 170 sensors                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                                  │
│  1. Standard scaling (fit on train, transform val)                  │
│  2. Cyclical time encoding → [T, 6]                                 │
│     (day-of-year, day-of-week, hour sin/cos)                        │
│  3. Temporal split (80/20) or random-window / TS-CV                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WINDOW DATASET                                  │
│  Slides over [C, T] with window_size=96, stride=1                   │
│  Each sample returns:                                               │
│    • window     [C, 96]       ← current t0 window                   │
│    • prev_windows [list]      ← t-1 (and optional earlier windows) │
│    • targets    [H, C]        ← next 12 steps (for probe)           │
│    • time_features [96, 6]    ← cyclical time covariates            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   VIEW BUILDER (Augmentation)                       │
│                                                                     │
│  AugmentationViewBuilder constructs the multi-view tensor:          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ View Structure (include_prev=2, repeat_factor=10):        │      │
│  │                                                           │      │
│  │ idx: 0       1       2        3     4    ...    12         │      │
│  │     [t-1_a1] [t-1_a2] [t0_clean] [t0_aug₁] [t0_aug₂] ... [t0_aug₁₀]│ │
│  │      ↑        ↑         ↑           ↑                          │      │
│  │   t-1 aug   t-1 aug   anchor for   augmented t0 copies          │      │
│  │   copies    copies    probe eval  for SSL invariance            │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                     │
│  Output: Batch(views=[B,V,C,T], view_times=[B,V,T,6],              │
│               targets=[B,H,C], future_times=[B,H,6])               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SSL CORE (LeJEPA_SSL)                         │
│                                                                     │
│  views [B,V,C,T]                                                    │
│       │                                                             │
│       ├── Flatten to [B*V, C, T]                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────┐     ┌────────────┐     ┌──────────────────┐           │
│  │ ENCODER │ ──► │ PROJECTOR  │ ──► │ SSL LOSS COMPUTE │           │
│  │ (swap-  │     │ MLP        │     │                  │           │
│  │  able)  │     │ D→512→64   │     │ 1. Invariance    │           │
│  └─────────┘     └────────────┘     │ 2. SIGReg        │           │
│                                     └──────────────────┘           │
│       ▼                                                             │
│  emb   [B, V, D]        proj [B, V, 64]                            │
│                                                                     │
│  Loss = (1-λ)·inv_loss + λ·sigreg_loss                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LIGHTNING TRAINER                                │
│                                                                     │
│  SSLPretrainModule wraps:                                           │
│    • ssl_core  → gradients flow here                                │
│    • probe     → detached evaluation only (no encoder gradients)    │
│                                                                     │
│  Training step:                                                     │
│    ssl_loss = ssl_core(views, view_times)                           │
│    probe_loss = probe(detach(encoder(clean_t0)))  ← monitoring      │
│    total = ssl_loss + 0.1 * probe_loss                              │
│                                                                     │
│  Optimizer: AdamW, separate param groups for SSL & probe            │
│  Callbacks: Checkpointing, LR monitoring, UMAP visualization       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
src/timeseries/
│
├── core/
│   ├── types.py           ← Batch, InvarianceMix, ViewTensor
│   ├── base_encoder.py    ← BaseEncoder ABC (forward_backbone + pooling)
│   └── interfaces.py      ← Protocol definitions
│
├── data/
│   ├── pems.py            ← PeMS08 loader + SSL/Probe DataModules
│   ├── window_dataset.py  ← WindowDataset (sliding window + prev contexts)
│   ├── view_dataset.py    ← ViewDataset (wraps WindowDataset + ViewBuilder)
│   ├── view_builders.py   ← AugmentationViewBuilder (multi-view construction)
│   └── probe_dataset.py   ← ForecastProbeDataset (standalone probe eval)
│
├── preprocessing/
│   ├── scalers.py         ← TorchStandardScaler (fit/transform/inverse)
│   └── time_encoding.py   ← Cyclical sin/cos time features [T, 6]
│
├── transforms/
│   ├── base.py            ← Transform ABC
│   ├── compose.py         ← Compose, RandomApply
│   └── ops/
│       └── basic.py       ← Scaling, Drift, Jitter, Noise, FreqMask,
│                             MagnitudeWarp, TemporalCrop, TemporalBlockMask
│
├── encoders/
│   ├── layers.py          ← ChannelMixer, SqueezeExcite, ChannelSelfAttention
│   ├── upernet.py         ← UPerNetEncoder (multi-level FPN + PPM)
│   ├── transformer.py     ← TransformerEncoder (positional encoding + layers)
│   ├── cnn.py             ← CNNEncoder (dilated residual blocks)
│   ├── rnn.py             ← LSTMEncoder, GRUEncoder (bidirectional)
│   ├── mamba.py           ← (Mamba SSM encoder, experimental)
│   └── covariate.py       ← FutureCovariateEncoder (for probe head)
│
├── ssl/
│   ├── lejepa.py          ← LeJEPA_SSL (invariance + SIGReg composition)
│   └── regularizers.py    ← TemporalSIGReg, CovarianceReg, VarianceReg,
│                             CombinedRegularizer
│
├── tasks/
│   └── forecast_probe.py  ← ForecastProbe (MLP head, multi-level fusion)
│
├── train/
│   ├── lightning_ssl.py   ← SSLPretrainModule (Lightning wrapper)
│   └── ray_tune_ssl.py    ← Ray Tune hyperparameter search
│
└── visualizations/
    ├── callbacks.py       ← VisualizationCallback (UMAP, forecast plots)
    └── datamodule.py      ← visualize_pems_tuple (view inspection)
```

---

## Data Pipeline Detail

### 1. Raw Data (PeMS08)

| Property         | Value                                     |
|------------------|-------------------------------------------|
| Sensors          | 170 traffic flow sensors                  |
| Time steps       | 17,856 (≈62 days at 5-min intervals)      |
| Start date       | 2016-07-01 00:00                          |
| Features         | Flow count per sensor                     |
| Graph            | Distance-based adjacency (Gaussian kernel)|

### 2. Preprocessing

```
Raw [T, C]  ──►  StandardScaler.fit(train) ──►  Scaled [C, T]
                                                      │
Timestamps  ──►  encode_timestamps_torch()  ──►  time_features [T, 6]
                  (sin/cos: day_year, weekday, hour)
```

The scaler fits on **train data only** and transforms both train/val. This prevents data leakage. Time features are cyclical sin/cos encodings that capture periodic patterns.

### 3. Window Slicing

```
WindowDataset slides over [C, T]:

    ◄── prev_shift * num_prev ──►◄── window_size=96 ──►◄── horizon=12 ──►
    ┌───────┬───────┬────────────┬──────────────────────┬────────────────┐
     │  t-1  │     t0     │  t0 (current window) │    target      │
     │ [C,96]│  [C,96]    │      [C,96]          │   [C,12]       │
     └───────┴────────────┴──────────────────────┴────────────────┘
    
    prev_shift=12 means each previous window is 12 steps earlier
```

### 4. View Construction

The `AugmentationViewBuilder` takes a `WindowSample` and produces the multi-view tensor:

```
Input: window [C,96], prev_windows=[t-1]

Step 1: Add prev views (AUGMENTED copies of t-1 only)
     → [t-1_aug1, t-1_aug2, ...]  (no implicit t-2 ordering)

Step 2: Add clean t0 (UNAUGMENTED)
     → [t-1_aug..., t0_clean]  (anchor for probe)

Step 3: Add augmented t0 copies (repeat_factor=10)
     → Apply: Compose([RandomApply(op, p=p_per_aug) for op in ops]) + TemporalBlockMask
     → [t-1_aug..., t0_clean, t0_aug₁, ..., t0_aug₁₀]

Output: views [13, C, 96], view_times [13, 96, 6]
```

### 5. Augmentation Transforms

| Transform          | Description                            | Probability |
|--------------------|----------------------------------------|-------------|
| `Scaling`          | Random amplitude scaling [0.8, 1.2]   | p_per_aug  |
| `Drift`            | Linear trend injection                 | p_per_aug  |
| `FeatureJitter`    | Gaussian noise per feature (σ=0.08)    | p_per_aug  |
| `AddGaussianNoise` | Additive noise to all channels         | p=0.4       |
| `FrequencyMask`    | Zero out frequency bands in FFT        | p=0.3       |
| `MagnitudeWarp`    | Smooth cubic spline amplitude warp     | p=0.4       |
| `TemporalBlockMask`| Mask contiguous time blocks with zeros | p=0.9       |

Applied in compound mode: each op uses `RandomApply(op, p=p_per_aug)` so multiple transforms can stack per view. Then `TemporalBlockMask` is applied independently.

---

## Encoder Architecture

All encoders follow `BaseEncoder` with:
- Input: `[B, C, T]` (batch, channels=170, time=96)
- Output: `[B, D]` (pooled) or `[B, T, D]` (token-level, `pool_mode="none"`)
- Optional `forward_multilevel()` → list of `[B, D]` per layer/stage

### UPerNet Encoder (Default)

```
Input [B, 170, 96]
       │
       ▼
  ChannelMixer (self-attention over 170 channels)
       │
       ▼
  Stem Conv1d(170→32, k=3)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  4 Hierarchical Stages (channel_mult=2.0)   │
  │                                             │
  │  Stage 0: ConvBlock(32→32,  stride=1) ──► feat₀ [B, 32, 96]  │
  │  Stage 1: ConvBlock(32→64,  stride=2) ──► feat₁ [B, 64, 48]  │
  │  Stage 2: ConvBlock(64→128, stride=2) ──► feat₂ [B,128, 24]  │
  │  Stage 3: ConvBlock(128→256,stride=2) ──► feat₃ [B,256, 12]  │
  └─────────────────────────────────────────────┘
       │
       ▼
  Pyramid Pooling Module (PPM)
  on feat₃ → AdaptiveAvgPool at scales [1,2,4] + upsample + bottleneck
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  Feature Pyramid Network (FPN)              │
  │  Top-down pathway: PPM → lateral + upsample │
  │  Smooth: 3×1 conv + norm per level          │
  │                                             │
  │  fpn₀ [B, 256, 96]  ← finest (used for     │
  │  fpn₁ [B, 256, 48]     forward_backbone)    │
  │  fpn₂ [B, 256, 24]                          │
  │  fpn₃ [B, 256, 12]                          │
  └─────────────────────────────────────────────┘
       │
       ▼
  forward_backbone: fpn₀.transpose → [B, 96, 256] → mean pool → [B, 256]
  forward_multilevel: [mean(fpn₀), mean(fpn₁), mean(fpn₂), mean(fpn₃)]
                    = 4 × [B, 256]
```

### Other Encoders

| Encoder       | Architecture                                   | Multi-level |
|---------------|------------------------------------------------|-------------|
| **Transformer** | Input proj + positional encoding + N transformer layers | Per-layer heads |
| **CNN**       | Stem + dilated residual blocks (d=1,2,4,8)     | Per-block heads |
| **LSTM**      | Per-layer bidirectional LSTM + projection       | Per-layer heads |
| **GRU**       | Same structure as LSTM with GRU cells           | Per-layer heads |
| **Mamba**     | SSM-based encoder (experimental)               | —           |

All encoders expose `forward_multilevel()` for the probe's attention-based level fusion.

---

## SSL Loss Computation

### Invariance Loss (Anchor-Based)

The goal: augmented views of the same window should produce similar embeddings.

```
proj [B, V, 64]
  │
  ├── Select invariance views (subsample augmented + keep anchor)
  │   → inv_pool [B, V_sel, 64]
  │
  ├── anchor = clean t0 embedding [B, 1, 64]
  │
  └── inv_loss = mean( (inv_pool - anchor)² )
```

When `use_anchor_invariance=True`, all views are pulled toward the **clean t0 anchor**. This is more principled than mean-based invariance because the anchor is unaugmented.

### SIGReg Regularizer

SIGReg prevents collapse by encouraging projected embeddings to follow an **isotropic Gaussian** distribution:

```
proj [N_samples, 64]
  │
  ├── Sample random directions A [64, 1024]  (re-sampled each step)
  │
  ├── 1D projections: z·A → [N, 1024]
  │
  ├── Characteristic function at knots t=[0, 3]:
  │     φ_empirical(t) = mean(cos(z·t)) + i·mean(sin(z·t))
  │     φ_gaussian(t)  = exp(-t²/2)
  │
  └── loss = mean_slices( ∫ |φ_emp(t) - φ_gauss(t)|² · w(t) dt )
```

The loss is zero when the marginal distributions along all random directions match a standard Gaussian → the embedding distribution is isotropic Gaussian.

### Total Loss

$$\mathcal{L}_{\text{total}} = (1 - \lambda) \cdot \mathcal{L}_{\text{inv}} + \lambda \cdot \mathcal{L}_{\text{sigreg}}$$

Default: $\lambda = 0.5$ (equal weight).

### Health Diagnostics (No Gradient)

| Metric                   | Healthy Range | Meaning                          |
|--------------------------|---------------|----------------------------------|
| `embedding_std`          | ≈ 1.0         | Mean std across features         |
| `feature_collapse_ratio` | ≈ 0.0         | Fraction of dead features (<0.1) |
| `temporal_alignment`     | Decreasing    | Prev views → t0 alignment       |

---

## Probe (Evaluation Only)

The `ForecastProbe` evaluates representation quality **without affecting encoder training**:

```
clean_t0 embedding (detached from encoder)
  │
  ├── If multi-level: attention-weighted fusion
  │   [B, L, D] → softmax(attn_net(·))·levels → [B, D]
  │
  ├── LayerNorm + Dropout
  │
  ├── MLP: D → 512 → 512 → 512 (3 layers with GELU + LN + Dropout)
  │
  └── Linear: 512 → C×H = 170×12 → reshape [B, 170, 12]
```

Key: embeddings are **detached** (`torch.no_grad()`) so probe gradients never flow into the encoder. The probe loss is weighted at 0.1 and serves as a monitoring signal.

---

## Training Configuration Summary

| Parameter            | Value    | Purpose                              |
|----------------------|----------|--------------------------------------|
| `window_size`        | 96       | Input window length (8 hours)        |
| `target_window_size` | 12       | Forecast horizon (1 hour)            |
| `prev_shift`         | 12       | Gap between consecutive prev windows |
| `include_prev`       | 2        | Include 2 augmented t-1 views        |
| `repeat_factor`      | 10       | Number of augmented t0 views         |
| `batch_size`         | 128      | Batch size                           |
| `lr`                 | 3e-4     | Learning rate (AdamW)                |
| `lamb`               | 0.5      | inv/sigreg balance                   |
| `epochs`             | 100      | Training epochs                      |
| `encoder_backbone`   | upernet  | Default encoder type                 |
| `gradient_clip_val`  | 0.30     | Norm-based gradient clipping         |
| `p_transform`        | 0.8      | Overall augmentation probability     |
| `p_temporal_mask`    | 0.9      | Temporal block masking probability   |

---

## Data Flow Diagram (End-to-End)

```
PeMS08 Raw Data [17856, 170]
         │
         ▼
    ┌────────────┐
    │  Scaler    │  fit on train → transform both
    │  [C, T]    │
    └─────┬──────┘
          │
    ┌─────▼──────┐    ┌──────────────┐
    │ Window     │──► │ View Builder │
    │ Dataset    │    │ (augment)    │
    │ slide+prev │    └──────┬───────┘
    └────────────┘           │
                      ┌──────▼───────┐
                      │ View Dataset │
                      │ → Batch      │
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
                      │  DataLoader  │
                      │  collate_fn  │
                      └──────┬───────┘
                             │
                ┌────────────▼────────────────┐
                │    SSLPretrainModule         │
                │                             │
                │  ┌──────────────────────┐   │
                │  │ LeJEPA_SSL           │   │
                │  │  encoder → projector │   │
                │  │  inv_loss + sigreg   │   │
                │  └──────────┬───────────┘   │
                │             │               │
                │  ┌──────────▼───────────┐   │
                │  │ ForecastProbe        │   │
                │  │  (detached, eval)    │   │
                │  └──────────────────────┘   │
                └─────────────────────────────┘
                             │
                      ┌──────▼───────┐
                      │  Frozen      │
                      │  Encoder     │
                      └──────┬───────┘
                             │
               ┌─────────────┼──────────────┐
               ▼             ▼              ▼
          Forecasting   Virtual Sensing  Classification
          (downstream)  (downstream)     (downstream)
```

---

## File Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [core/types.py](src/timeseries/core/types.py) | Shared data types | `Batch`, `InvarianceMix` |
| [core/base_encoder.py](src/timeseries/core/base_encoder.py) | Encoder ABC | `BaseEncoder` |
| [data/pems.py](src/timeseries/data/pems.py) | Data loading + DataModules | `PeMS08`, `PeMS08SSLDataModule` |
| [data/window_dataset.py](src/timeseries/data/window_dataset.py) | Sliding windows | `WindowDataset`, `WindowSample` |
| [data/view_builders.py](src/timeseries/data/view_builders.py) | Multi-view construction | `AugmentationViewBuilder` |
| [data/view_dataset.py](src/timeseries/data/view_dataset.py) | View batch creation | `ViewDataset`, `collate_batches` |
| [ssl/lejepa.py](src/timeseries/ssl/lejepa.py) | SSL core logic | `LeJEPA_SSL`, `SSLBatchResult` |
| [ssl/regularizers.py](src/timeseries/ssl/regularizers.py) | SIGReg + VICReg variants | `TemporalSIGReg`, `CombinedRegularizer` |
| [encoders/upernet.py](src/timeseries/encoders/upernet.py) | Default encoder | `UPerNetEncoder` |
| [tasks/forecast_probe.py](src/timeseries/tasks/forecast_probe.py) | Probe head | `ForecastProbe` |
| [train/lightning_ssl.py](src/timeseries/train/lightning_ssl.py) | Lightning wrapper | `SSLPretrainModule` |
| [transforms/compose.py](src/timeseries/transforms/compose.py) | Transform composition | `Compose`, `RandomApply` |
