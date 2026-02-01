# Implementation Plan: Time‑Series SSL Framework (LeJEPA)

Date: 2026‑02‑01

## Goals (from requirements)
- Encoder‑agnostic SSL framework for time series.
- Strict data shape: `[batch, channels(sensors), timesteps, features(optional)]`.
- SSL uses **only past windows** (no t+1 in SSL views).
- LeJEPA‑style SSL: multi‑view invariance + SIGReg regularization.
- Forecasting probe for evaluation only (no gradient to encoder).
- Scalable and flexible augmentation recipes for later tuning (Ray Tune uses dicts).

## Non‑Goals
- No downstream finetuning tasks in this phase.
- No changes to model performance hyper‑optimization in this phase.

---

## Phase 0 — Baseline snapshot (no code changes)
Purpose: capture current status and risks.
1. Inventory current modules:
   - SSL core + training loop: `src/timeseries/lejepa.py`
   - Augmentations: `src/timeseries/augementation/v1.py`
   - Dataset + views: `src/timeseries/data/pems.py`
   - Encoders + covariates: `src/timeseries/models/*`
2. Identify hard couplings:
   - SSL logic mixed with forecasting probe.
   - Dataset returns t+1 view, which violates “past only” SSL goal.
   - Augmentation logic is monolithic and dataset‑embedded.

Deliverable: no files changed; plan validated against current modules.

---

## Phase 1 — Core contracts and data types (new files)
Purpose: formalize data shapes and interfaces used across datasets, views, encoders, and SSL.

### 1.1 New file: `src/timeseries/core/types.py`
- Define `Batch` dataclass:
  - `views`: `torch.Tensor` shaped `[B, V, C, T]` or `[B, V, C, T, F]`.
  - `view_times`: optional time covariates aligned to views.
  - `targets`: optional forecast targets `[B, H, C]`.
  - `future_times`: optional time features `[B, H, F_time]`.
- Enforce shape validation helpers:
  - `assert_view_shape(views)`
  - `assert_time_shape(view_times, views)`

### 1.2 New file: `src/timeseries/core/interfaces.py`
- Define abstract base classes:
  - `BaseEncoder` with `output_dim`, `output_mode` (“pooled” | “token”).
  - `BaseProjector` for pooled/token inputs.
  - `BaseViewBuilder` for constructing multi‑view SSL inputs from raw windows.
  - `BaseWindowDataset` for slicing raw data windows.

Deliverable: typed contracts for everything downstream.

---

## Phase 2 — Data pipeline separation
Purpose: decouple window slicing from view creation and remove t+1 from SSL views.

### 2.1 New file: `src/timeseries/data/window_dataset.py`
- `WindowDataset`:
  - Inputs: raw tensor `[C, T]`, time features `[T, F_time]`, window size, stride.
  - Output: a minimal sample containing `window` (past only) and optional `time_features`.
  - No augmentation; no views; no target.

### 2.2 New file: `src/timeseries/data/view_dataset.py`
- `ViewDataset` wraps a `WindowDataset` and a `ViewBuilder`.
- `__getitem__`:
  - Generates multiple augmented views from the *same* past window.
  - Outputs `Batch` with `views` and `view_times`.

### 2.3 Update `src/timeseries/data/pems.py`
- Keep dataset‑specific loading and scaling.
- Replace direct view creation with `WindowDataset + ViewDataset`.
- Ensure SSL views never include t+1 window.
- Move forecast target generation into a separate probe dataset (see Phase 5).

Deliverable: SSL data path uses only past windows.

---

## Phase 3 — Augmentation framework (critical)
Purpose: make augmentations modular, scalable, and Ray‑Tune‑friendly.

### 3.1 New package: `src/timeseries/transforms/`
Files:
- `base.py`: `Transform` interface with `__call__(x)`, `train()` and `eval()` modes.
- `compose.py`: `Compose`, `RandomApply`, `OneOf`.
- `registry.py`: register transforms by name; load from dicts.
- `ops/*.py`: individual transforms:
  - `Scaling`, `Drift`, `Jitter`, `GaussianNoise`, `FrequencyMask`, `MagnitudeWarp`, `TemporalCrop`, `TemporalBlockMask`.

### 3.2 Config‑first augmentation recipes
- Recipe format is a Python dict (Ray Tune‑friendly):
  - Example pattern:
    - `{"name": "Compose", "transforms": [ ... ]}`
- Add helper `build_transform_from_dict(config: dict)`.

### 3.3 Migrate current transforms
- Port logic from `augementation/v1.py` into modular ops.
- Preserve identical behavior for default config.

Deliverable: augmentations are fully config‑driven and easy to tune.

---

## Phase 4 — SSL Core module (encoder‑agnostic)
Purpose: separate SSL logic from Lightning and from probing.

### 4.1 New file: `src/timeseries/ssl/lejepa.py`
- `LeJEPA_SSL` (torch `nn.Module`):
  - Inputs: `encoder`, `projector`, `regularizer` (SIGReg), `output_mode`.
  - `forward(views, view_times=None)` returns:
    - `embeddings` (pooled or token)
    - `proj` (projected embeddings)
    - `loss_ssl` (invariance + SIGReg)
- Invariance loss:
  - Use repeated views of the same window.
  - `inv_loss = (v_proj.mean(dim=1) - v_proj).square().mean()`
- SIGReg:
  - Use existing `TemporalSIGReg` with randomized slices.

### 4.2 Move SIGReg into `src/timeseries/ssl/regularizers.py`
- Keep current implementation.
- Add unit tests for shape acceptance.

Deliverable: SSL core can be used in any training loop.

---

## Phase 5 — Forecasting probe (evaluation‑only)
Purpose: probe is decoupled and does not update encoder.

### 5.1 New file: `src/timeseries/tasks/forecast_probe.py`
- `ForecastProbe`:
  - Input: pooled encoder embeddings.
  - Optional covariate encoder for known future covariates.
  - Detach embeddings before probe forward.

### 5.2 Probe dataset
- `ForecastProbeDataset`:
  - Uses same `WindowDataset` but returns `target` and `future_times`.
  - Not used by SSL loss.

Deliverable: probe evaluation is isolated and safe.

---

## Phase 6 — Training runners
Purpose: support Lightning now, allow Ray Tune later.

### 6.1 Lightning module wrapper
- New file: `src/timeseries/train/lightning_ssl.py`:
  - Wrap `LeJEPA_SSL` and `ForecastProbe`.
  - Only SSL loss backproped.
  - Probe metrics logged but no encoder gradients.

### 6.2 Ray Tune compatibility
- Provide a pure‑PyTorch train loop in `src/timeseries/train/loop.py`:
  - Accepts dict config.
  - Can be wrapped by Ray Tune `Trainable`.
  - Uses same `LeJEPA_SSL` core.

Deliverable: scalable path without re‑architecting later.

---

## Phase 7 — Minimal tests (essential)
Purpose: guard against shape regressions and leakage.

### 7.1 Shape tests
- Check `views` shape compliance.
- Verify that views are derived only from past windows (no t+1 in SSL dataset).

### 7.2 SSL loss tests
- Invariance loss: identical views reduce loss to ~0.
- SIGReg: accepts both pooled and token outputs.

### 7.3 Probe tests
- Ensure probe uses detached embeddings.

Deliverable: smoke tests in `src/timeseries/tests`.

---

## Migration checklist (current code → new structure)
1. Keep existing `pems.py` loader but remove view creation from dataset.
2. Move augmentations from `augementation/v1.py` into new transform ops.
3. Extract `TemporalSIGReg` and SSL logic from `lejepa.py` into `ssl` module.
4. Keep `LeJEPA_Forecaster` temporarily as a thin wrapper until new Lightning module is ready.
5. Add a deprecation note in `lejepa.py` pointing to new modules.

---

## Risks and mitigations
- **Risk**: breaking current training notebooks.
  - Mitigation: keep old entry points until new ones are tested.
- **Risk**: inconsistent view shapes.
  - Mitigation: shape validation in `Batch` and dataset wrappers.
- **Risk**: augmentation mismatch with existing runs.
  - Mitigation: include a “legacy” recipe mirroring `v1.py` defaults.

---

## Immediate next steps (actionable)
1. Implement Phase 1 (core contracts).
2. Implement Phase 3 (augmentation framework).
3. Implement Phase 2 (dataset refactor).
4. Implement Phase 4 (SSL core).
5. Implement Phase 5 (forecast probe).

---

## Open items (confirm later)
- Decide whether to keep Hydra in the future.
- Decide how to serialize augmentation recipes (JSON/YAML). Ray Tune expects dicts; YAML can be loaded into dicts.
