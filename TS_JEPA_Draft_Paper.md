# LeJEPA-TS: Heuristics-Free Joint-Embedding Predictive Learning for Traffic Forecasting

**Authors:** (draft)
**Date:** January 2026
**Workspace:** `lejepa`

This draft is grounded in the current repository implementation under `src/timeseries/` and is positioned relative to:
- LeJEPA (arXiv:2511.08544)
- V-JEPA 2 (arXiv:2506.09985)
- Time-Series JEPA for remote control (arXiv:2406.04853)
- Joint Embeddings Go Temporal (arXiv:2509.25449)

---

## Abstract

We present **LeJEPA-TS**, a time-series adaptation of Joint-Embedding Predictive Architectures (JEPAs) that targets stable, scalable self-supervised representation learning for traffic forecasting. The method combines (i) a **temporal multi-view JEPA objective** using shifted windows $\{t-\Delta, t, t+\Delta\}$, (ii) a **Sketched Isotropic Gaussian Regularization** (SIGReg) term inspired by LeJEPA to shape latent embeddings toward an isotropic Gaussian, and (iii) a lightweight **forecasting probe** trained from frozen (detached) encoder features. We instantiate the approach on the **PeMS08** multivariate traffic dataset with 170 sensors, using time-aware covariates (cyclical calendar features) and domain-specific augmentations (temporal cropping, block masking, frequency masking, and noise). This document describes the implemented objective, data pipeline, and training recipe and lays out an evaluation protocol for forecasting and representation quality.

---

## 1. Introduction

Traffic forecasting requires models that (a) capture temporally evolving latent state, (b) remain robust to missing/noisy sensor measurements, and (c) scale across many correlated channels (sensors). While supervised predictors can perform well, they often learn brittle features and depend heavily on the availability and quality of labels.

JEPAs propose learning by **predicting in representation space** rather than reconstructing raw inputs. Recent work provides both theory and practical training objectives (LeJEPA), large-scale instantiations in video (V-JEPA 2), and first time-series adaptations (TS-JEPA variants). This repo implements a time-series JEPA training loop that explicitly aligns temporally shifted views and regularizes latent space with SIGReg-style constraints.

### 1.1 Why This Might Work (Evidence + Motivation)

This implementation is motivated by three recurring themes in the JEPA literature:

1) **Predict in latent space to focus on predictable structure.** Video JEPA work argues that predicting representations rather than pixels encourages models to capture predictable aspects of a scene and deemphasize highly variable, hard-to-predict details that reconstruction/generation objectives must model (V-JEPA 2; arXiv:2506.09985).

2) **Prevent collapse and improve “foundation-ness” via principled embedding geometry.** LeJEPA proposes that enforcing an isotropic Gaussian embedding distribution jointly with a JEPA prediction loss eliminates representational collapse “by construction,” replacing common stabilization heuristics (stop-gradient, teacher-student EMA schedules, whitening, etc.). It further motivates isotropic Gaussian embeddings as minimizing worst-case downstream risk over broad probe families, and introduces SIGReg (random projections + characteristic-function matching) as a scalable way to enforce this target distribution (LeJEPA; arXiv:2511.08544).

3) **Time-series JEPAs can be competitive and robust.** A systematic TS-JEPA study reports that JEPA-style pretraining for time series yields strong classification performance and competitive forecasting, and explicitly motivates JEPA-style latent prediction as more robust to noise/confounders than input-space reconstruction (Joint Embeddings Go Temporal; arXiv:2509.25449). Separately, TS-JEPA for predictive remote control uses low-dimensional semantic embeddings with latent prediction to reduce communication and maintain control performance under capacity constraints, reinforcing the appeal of “embedding-first” time-series modeling (Time-Series JEPA for predictive remote control; arXiv:2406.04853).

---

## 2. Related Work

### 2.1 JEPAs and LeJEPA

LeJEPA (arXiv:2511.08544) develops a theory suggesting embeddings should approach an isotropic Gaussian to reduce downstream risk, and introduces SIGReg to enforce this without common stabilization heuristics.

### 2.2 World Models via JEPA

V-JEPA 2 (arXiv:2506.09985) demonstrates the scalability of JEPA-style training for video understanding and latent dynamics modeling, and extends to action-conditioned latent world modeling for planning.

### 2.3 JEPA for Time Series

Time-Series JEPA for predictive remote control (arXiv:2406.04853) uses latent prediction to reduce communication and enable control under capacity constraints.

Joint Embeddings Go Temporal (arXiv:2509.25449) focuses on a general-purpose time-series JEPA setup and evaluates across classification and forecasting benchmarks.

---

## 3. Method

### 3.1 Problem Setup

Let $x \in \mathbb{R}^{C \times L}$ be a multivariate time-series window with $C$ channels (sensors) and window length $L$. Given a forecast horizon $H$, the supervised target is $y \in \mathbb{R}^{C \times H}$.

We construct three temporally shifted “views” of the same underlying process:
$$
x^{(-)} = x_{t-\Delta:t-\Delta+L},\quad
x^{(0)} = x_{t:t+L},\quad
x^{(+)} = x_{t+\Delta:t+\Delta+L}.
$$

### 3.2 Architecture

The implementation in `src/timeseries/lejepa.py` uses:
- Encoder backbone $f_\theta$: maps $x$ to a feature vector $h$ (e.g., Conv1D encoder in `src/timeseries/models/cnn.py`).
- Projector $g_\phi$: maps $h \mapsto z$ in a lower-dimensional JEPA space.
- Predictor $p_\psi$: predicts future/adjacent embeddings in JEPA space.
- Forecast head $q_\omega$: predicts $\hat y$ from encoder features.

Optionally, known future covariates can be encoded via a small Conv1D module (`FutureCovariateEncoder` in `src/timeseries/models/covariate.py`) and concatenated into the forecast head input.

### 3.3 Losses

Let $z^{(-)}, z^{(0)}, z^{(+)}$ be projected embeddings for the three views.

**(1) Temporal invariance (view consistency).** Encourage local temporal stability:
$$
\mathcal{L}_{\text{inv}} = \tfrac{1}{2}\big(\|z^{(-)} - z^{(0)}\|_2^2 + \|z^{(+)} - z^{(0)}\|_2^2\big).
$$

**(2) Predictive embedding loss.** The implemented predictor trains on adjacent prediction:
$$
\mathcal{L}_{\text{pred}} = \|p(z^{(-)}) - z^{(0)}\|_2^2 + \|p(z^{(0)}) - z^{(+)}\|_2^2.
$$

**(3) SIGReg-style regularization.** We apply a sketched univariate normality regularizer over many random 1D projections (“slices”) of the embedding set. In the repo this is either:
- `lejepa.multivariate.SlicingUnivariateTest` if the external `lejepa` package is available, or
- a built-in `TemporalSIGReg` module (characteristic-function matching against a Gaussian along random directions).

We denote this regularizer by $\mathcal{R}(z)$.

**(4) Forecasting loss.** Predict the future horizon from the current view’s encoder features:
$$
\mathcal{L}_{\text{fc}} = \|\hat y - y\|_2^2.
$$

**Important implementation detail (probe-style head).** In `src/timeseries/lejepa.py`, the forecast head is fed with *detached* encoder features (i.e., gradients from $\mathcal{L}_{\text{fc}}$ do not update the encoder). This makes $q_\omega$ a probe on top of self-supervised features.

**Overall objective (as implemented).** With trade-off parameter $\lambda \in [0,1]$:
$$
\mathcal{L}_{\text{ssl}} = (\mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{inv}})(1-\lambda) + \lambda\,\mathcal{R}(z),
$$
and total loss:
$$
\mathcal{L} = 0.1\,\mathcal{L}_{\text{fc}} + \mathcal{L}_{\text{ssl}}.
$$

---

## 4. Data Pipeline and Augmentations

### 4.1 Dataset (PeMS08)

The repo includes a PeMS08 loader and Lightning datamodule in `src/timeseries/data/pems.py`. The raw file is expected under `data/pems08/` (already present in this workspace). The datamodule:
- loads traffic flow for 170 sensors;
- performs an 80/20 temporal split;
- standardizes each channel using `TorchStandardScaler` (mean/std stored for unscaled metrics);
- constructs three shifted views and a forecast target per sample.

### 4.2 Time Covariates

For each timestamp, `src/timeseries/preprocessing/time_encoding.py` produces 6 cyclical features:
$$
[\sin(\text{day}),\cos(\text{day}),\sin(\text{weekday}),\cos(\text{weekday}),\sin(\text{hour}),\cos(\text{hour})].
$$
The dataset returns view-wise time features and future-horizon time features.

### 4.3 Augmentations

The training datamodule uses `TimeSeriesTransform` in `src/timeseries/augementation/v1.py`, which (stochastically) applies:
- scaling, drift, magnitude warping,
- frequency masking via FFT,
- additive Gaussian noise,
- temporal crop + resize to fixed length,
- temporal block masking.

---

## 5. Experimental Protocol (Implemented Recipe)

This repo currently provides a notebook-based training entrypoint in `src/timeseries/dev.ipynb`.

### 5.1 Default Notebook Configuration

The notebook constructs the following config:
- window size $L=96$
- horizon $H=12$
- temporal shift $\Delta=24$
- batch size $128$
- learning rate $10^{-3}$
- $\lambda=0.5$
- epochs $800$

### 5.2 Backbone

The notebook instantiates a simple Conv1D encoder (`TimeSeriesEncoder`) with `output_dim=256`.

### 5.3 Logging and Checkpointing

The notebook logs to TensorBoard and checkpoints on `val/mae_unscaled` (computed by inverting the standardization).

---

## 6. Results (To Be Filled)

This repository contains the training code and logging hooks, but this draft intentionally does **not** report concrete numerical results until they are reproduced from saved runs.

Recommended reporting:
- forecasting: MAE / RMSE / MAPE at standard horizons;
- representation: linear-probe forecasting and/or classification on frozen embeddings;
- ablations: remove SIGReg ($\lambda=0$), remove predictor loss, remove augmentations.

---

## 7. Discussion

**Why SIGReg for time series?** LeJEPA motivates isotropic-Gaussian embeddings as a principled target (downstream-risk minimization) and introduces SIGReg as a scalable, stable distribution-matching term via random projections and characteristic-function matching (arXiv:2511.08544). For time series—often noisy and non-stationary—this kind of explicit anti-collapse and geometry control is a plausible way to stabilize JEPA-style training without relying on teacher–student heuristics.

**Probe-style forecasting head.** Detaching encoder features for $\mathcal{L}_{\text{fc}}$ keeps the SSL objective as the primary driver of representation learning. This matches the common “frozen backbone + probe” evaluation protocol used to assess representation quality in JEPA-style systems (e.g., V-JEPA 2 uses frozen evaluation protocols for representation quality; arXiv:2506.09985) and in TS-JEPA-style studies (arXiv:2509.25449). If the end goal is best forecasting accuracy (not just representation quality), an end-to-end variant (removing `detach()`) should be evaluated separately.

---

## 8. Conclusion

LeJEPA-TS is a practical time-series JEPA instantiation that combines multi-view temporal embedding prediction with SIGReg-style regularization, plus a forecasting probe for downstream evaluation. The repo provides an end-to-end PeMS08 pipeline (data, augmentations, Lightning module, training notebook). The next step is to run controlled experiments and fill in the results section with reproducible metrics.

---

## References

1. Randall Balestriero, Yann LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* arXiv:2511.08544.
2. Mido Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985.
3. Abanoub M. Girgis, Alvaro Valcarce, Mehdi Bennis. *Time-Series JEPA for Predictive Remote Control under Capacity-Limited Networks.* arXiv:2406.04853.
4. Sofiane Ennadir, Siavash Golkar, Leopoldo Sarra. *Joint Embeddings Go Temporal.* arXiv:2509.25449.

---

## Appendix A. Implementation Map

- Objective + Lightning module: `src/timeseries/lejepa.py` (`LeJEPA_Forecaster`, `TemporalSIGReg`)
- PeMS08 datamodule + dataset: `src/timeseries/data/pems.py`
- Augmentations: `src/timeseries/augementation/v1.py`
- Time covariates: `src/timeseries/preprocessing/time_encoding.py`
- Scaling: `src/timeseries/preprocessing/scalers.py`
- Example training run: `src/timeseries/dev.ipynb`

```python
class LeJEPA_Forecaster(L.LightningModule):
    def __init__(self, encoder_backbone, input_dim, horizon,
                 proj_dim=128, num_slices=1024, lamb=0.5, lr=1e-3,
                 scaler_mean=None, scaler_std=None,
                 use_covariates=False, num_time_features=6):
        # Implementation details...
```

### A.2 Training Pipeline

```python
def shared_step(self, batch, batch_idx, mode="train"):
    # Unpack batch
    # Encode views
    # Compute losses
    # Log metrics
    # Return total loss
```

### A.3 Data Module

```python
class PeMS08DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        # Implementation details...
```

---

## Appendix B: Experimental Details

### B.1 Hyperparameter Search (Placeholder)

This section is intentionally left as a placeholder until runs are executed and logged reproducibly (see Section 6). Suggested ablations are:

| Factor | Suggested Values |
|--------|------------------|
| $\lambda$ (SIGReg weight) | {0, 0.05, 0.1, 0.3, 0.5} |
| # slices / projections | {256, 512, 1024, 2048} |
| Augmentations | {off, on} |
| Forecast head | {probe (detach), end-to-end} |

### B.2 Computational Resources (Placeholder)

Record the actual hardware, wall-clock, and peak memory from your run logs (e.g., TensorBoard + `nvidia-smi`) once you execute experiments.

---

## Appendix C: Future Work

1. **Extension to other modalities**: Incorporate weather data, events, etc.
2. **Real-time adaptation**: Online learning for changing traffic patterns
3. **Explainable AI**: Methods to interpret model decisions
4. **Edge deployment**: Optimization for resource-constrained devices

---

**End of Draft Paper**
