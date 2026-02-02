import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from typing import TYPE_CHECKING, Any, Optional

import torch

from ..core.types import Batch

if TYPE_CHECKING:
    from ..data.pems import PeMS08DataModule

pio.templates.default = "plotly_white"
def visualize_pems_tuple(
    batch_or_datamodule: Any,
    stage: str = 'train',
    batch_index: int = 0,
    sensor_index: int = 0,
    *,
    window_size: Optional[int] = None,
    target_window_size: Optional[int] = None,
    temporal_shift: Optional[int] = None,
    show_prev: bool = True,
    show_next: bool = False,
    show_target: bool = True,
    show_t0_mean: bool = True,
    opacity_aug: float = 0.7,
    max_t0_views: Optional[int] = None,
    align_prev_to_t0: bool = True,
    align_method: str = "xcorr",
    show: bool = True,
):
    """
    Visualize PeMS multi-view windows and the (optional) future target.

    This function is intentionally flexible:
    - Pass a datamodule (historical behavior): `visualize_pems_tuple(datamodule, stage=..., batch_index=..., ...)`
    - Pass a single batch directly (new): `visualize_pems_tuple(batch, batch_index=..., ...)`

    Notes:
    - The dataset can return a variable number of views V.
      Convention is: view[0]=previous window, view[1:V-1]=repeated augmented current windows, view[V-1]=next window (if present).
    - `sensor_index` is treated as 1-based if > 0 (legacy behavior), else 0-based.

    Args:
        batch_or_datamodule: Either a datamodule with train/val dataloaders, or a single batch.
        stage: Used only when a datamodule is provided.
        batch_index: Index of the sample within the batch (effectively sample index).
        sensor_index: 1-based if > 0 else 0-based.
        window_size/target_window_size/temporal_shift: Optional overrides when passing a batch directly.
        show_prev: If True and V>=3, plot the previous window (t-1) as a dashed trace.
        show_next: If True and V>=3, plot the next window (t+1). Defaults to False since many datasets are past-only.
        show_target: If True and targets exist, plot them after t0.
        show_t0_mean: Plot mean across t0 augmented views.
        opacity_aug: Opacity for individual t0 augmentation traces.
        max_t0_views: Optional cap on how many t0 augmented views to draw (for speed/readability).
    """
    def _unpack_batch(batch):
        if isinstance(batch, Batch):
            return batch.views, batch.targets
        if isinstance(batch, (tuple, list)):
            # Handle both old (2-tuple) and new (4-tuple) batch formats
            if len(batch) == 4:
                views_b, target_b, _, _ = batch
                return views_b, target_b
            if len(batch) == 2:
                views_b, target_b = batch
                return views_b, target_b
        raise ValueError(f"Unsupported batch type: {type(batch)!r}")

    # If a datamodule-like object is passed, fetch one batch.
    if hasattr(batch_or_datamodule, 'train_dataloader') or hasattr(batch_or_datamodule, 'val_dataloader'):
        datamodule = batch_or_datamodule
        dataloader = datamodule.train_dataloader() if stage == 'train' else datamodule.val_dataloader()
        try:
            batch = next(iter(dataloader))
        except (StopIteration, Exception) as e:
            print(f"Error loading batch: {e}")
            return

        try:
            views_batch, target_batch = _unpack_batch(batch)
        except Exception as e:
            print(f"Error unpacking the batch: {e}")
            return

        # Pull defaults from cfg if not overridden
        if window_size is None and hasattr(datamodule, 'cfg'):
            window_size = getattr(datamodule.cfg, 'window_size', None)
        if target_window_size is None and hasattr(datamodule, 'cfg'):
            target_window_size = getattr(datamodule.cfg, 'target_window_size', None)
        if temporal_shift is None and hasattr(datamodule, 'cfg'):
            temporal_shift = getattr(datamodule.cfg, 'temporal_shift', None)
            if temporal_shift is None:
                temporal_shift = getattr(datamodule.cfg, 'prev_shift', None)
    else:
        # A single batch was passed directly.
        batch = batch_or_datamodule
        try:
            views_batch, target_batch = _unpack_batch(batch)
        except Exception as e:
            print(f"Error unpacking the batch: {e}")
            return

    if not isinstance(views_batch, torch.Tensor):
        print(f"Error: views_batch expected torch.Tensor, got {type(views_batch)!r}")
        return

    if batch_index >= views_batch.size(0):
        print(f"Error: batch_index {batch_index} is out of range (batch size: {views_batch.size(0)})")
        return

    if views_batch.ndim == 5:
        # If extra feature dimension exists, reduce for visualization
        views_batch = views_batch.mean(dim=-1)
    if views_batch.ndim != 4:
        print(f"Error: Expected views_batch with 4 dimensions [B, V, C, L], got shape={tuple(views_batch.shape)}")
        return
    
    # Infer lengths if not provided
    if window_size is None:
        window_size = int(views_batch.shape[-1])
    if target_window_size is None and target_batch is not None and isinstance(target_batch, torch.Tensor):
        # target can be [B, H, C] or [B, C, H]
        if target_batch.ndim == 3:
            # Heuristic: in typical forecasting, H << C, so prefer the smaller of the last two dims
            # but keep it deterministic for ambiguous shapes.
            if int(target_batch.shape[1]) <= int(target_batch.shape[2]):
                target_window_size = int(target_batch.shape[1])
            else:
                target_window_size = int(target_batch.shape[2])
    if target_window_size is None:
        target_window_size = 0

    if temporal_shift is None:
        temporal_shift = 0

    L_in = int(window_size)
    L_target = int(target_window_size)
    temporal_shift = int(temporal_shift)

    V = int(views_batch.shape[1])
    if V < 2:
        print(f"Error: Expected at least 2 views, got V={V}")
        return

    # If V>=3, we assume [t-1, t0..., t+1]. Otherwise, treat all as t0-like.
    has_prev_next = V >= 3
    has_prev = has_prev_next
    has_next = has_prev_next
    repeat_factor = V - 2 if has_prev_next else V

    sensor_idx = sensor_index - 1 if sensor_index > 0 else 0
    if sensor_idx < 0 or sensor_idx >= int(views_batch.shape[2]):
        print(
            f"Error: sensor_index={sensor_index} (interpreted idx={sensor_idx}) is out of range "
            f"[0, {int(views_batch.shape[2]) - 1}]"
        )
        return

    print(f"--- Visualization ({stage} data) ---")
    print(f"Window lengths: L_in={L_in}, Target={L_target}. | Temporal shift: {temporal_shift}")
    print(f"Views: V={V} => repeat_factor={repeat_factor} (t0 repeated views)")
    
    # 3. Visualization
    fig = go.Figure()
    
    # base_start in the dataset is temporal_shift.
    # In the visualization, we can set t0 start to 0 for convenience.
    # Then t-1 starts at -temporal_shift, and t+1 starts at +temporal_shift.
    
    time_curr = np.arange(0, L_in)
    time_target = np.arange(L_in, L_in + L_target) if L_target > 0 else None
    if has_prev and show_prev:
        time_prev = np.arange(-temporal_shift, -temporal_shift + L_in)
    if has_prev and show_next:
        time_next = np.arange(temporal_shift, temporal_shift + L_in) 

    # target_batch shape: often [B, H, C] but sometimes [B, C, H]
    v_prev = None
    v_next = None
    if has_prev and show_prev:
        v_prev = views_batch[batch_index, 0, sensor_idx, :].cpu().numpy()
    if has_prev and show_next:
        v_next = views_batch[batch_index, -1, sensor_idx, :].cpu().numpy()
    target = None

    # Optionally align the prev view to the mean of t0 views using cross-correlation
    shift_info = None
    if align_prev_to_t0 and has_prev_next and v_prev is not None and len(views_batch.shape) >= 4:
        # Choose reference: mean of t0 views or first t0 view
        t0_start = 1 if has_prev_next else 0
        t0_end = V - 1 if has_prev_next else V
        ref_indices = list(range(t0_start, t0_end))
        if max_t0_views is not None:
            ref_indices = ref_indices[: int(max_t0_views)]
        if len(ref_indices) > 0:
            refs = [views_batch[batch_index, j, sensor_idx, :].cpu().numpy() for j in ref_indices]
            ref_signal = np.mean(np.stack(refs, axis=0), axis=0)

            if align_method == "xcorr":
                # Normalize and compute cross-correlation
                a = ref_signal - ref_signal.mean()
                b = v_prev - v_prev.mean()
                corr = np.correlate(a, b, mode='full')
                lag = corr.argmax() - (len(b) - 1)
                # Shift prev signal by lag (i.e. adjust its x positions)
                shift_info = int(lag)
            else:
                shift_info = 0
    if show_target and target_batch is not None and isinstance(target_batch, torch.Tensor) and L_target > 0:
        target_sample = target_batch[batch_index]
        if target_sample.ndim == 2:
            if target_sample.shape[0] == L_target:
                # [H, C]
                target = target_sample[:, sensor_idx].cpu().numpy()
            elif target_sample.shape[1] == L_target:
                # [C, H]
                target = target_sample[sensor_idx, :].cpu().numpy()
        if target is None:
            print(
                f"Warning: target shape {tuple(target_sample.shape)} does not match L_target={L_target}. "
                "Skipping target plot."
            )

    if v_prev is not None and has_prev and show_prev:
        # Apply computed shift (if any) to the plotting x axis
        if shift_info is not None and shift_info != 0:
            shifted_time_prev = time_prev + shift_info
            ann = f"aligned (lag={shift_info})"
        else:
            shifted_time_prev = time_prev
            ann = None

        fig.add_trace(
            go.Scatter(
                x=shifted_time_prev,
                y=v_prev,
                name='Previous window (t-1)',
                line=dict(color='blue', width=2, dash='dash'),
                opacity=0.5,
                hovertemplate=("t=%{x}<br>y=%{y}<extra>t-1</extra>"),
            )
        )
        if ann is not None:
            fig.add_annotation(x=shifted_time_prev.mean(), y=v_prev.max(), text=ann, showarrow=False, yanchor='bottom')

    # Plot each repeated augmented t0 view, plus their mean.
    # Plot augmented t0 views
    curr_views = []
    if has_prev_next:
        t0_start = 1
        t0_end = V - 1
    else:
        t0_start = 0
        t0_end = V

    t0_indices = list(range(t0_start, t0_end))
    if max_t0_views is not None:
        t0_indices = t0_indices[: int(max_t0_views)]

    for k, j in enumerate(t0_indices, start=1):
        v_curr_j = views_batch[batch_index, j, sensor_idx, :].cpu().numpy()
        curr_views.append(v_curr_j)
        fig.add_trace(
            go.Scatter(
                x=time_curr,
                y=v_curr_j,
                name=f'T0 aug {k}/{len(t0_indices)}',
                line=dict(color='rgba(0, 128, 0, 0.35)', width=1),
                opacity=float(opacity_aug),
            )
        )

    if show_t0_mean and len(curr_views) > 0:
        v_curr_mean = np.mean(np.stack(curr_views, axis=0), axis=0)
        fig.add_trace(
            go.Scatter(
                x=time_curr,
                y=v_curr_mean,
                name='T0 mean (over repeats)',
                line=dict(color='green', width=3),
                opacity=0.9,
            )
        )

    if v_next is not None and has_prev and show_next:
        fig.add_trace(
            go.Scatter(
                x=time_next,
                y=v_next,
                name='Next window (t+1)',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.5,
            )
        )
    
    if target is not None and time_target is not None:
        fig.add_trace(
            go.Scatter(
                x=time_target,
                y=target,
                name='Target (Future of T0)',
                line=dict(color='black', width=2, dash='dash'),
                opacity=0.65,
            )
        )

    # Vertical lines for window boundaries
    fig.add_vline(x=0, line_color='gray', line_width=2, line_dash='dash', opacity=0.3, annotation_text='Start T0')
    if show_target and L_target > 0:
        fig.add_vline(x=L_in, line_color='red', line_width=2, line_dash='dash', opacity=0.5, annotation_text='End T0 / Start Target')

    # 5. Titel und Achsenbeschriftungen
    fig.update_layout(
        title=(
            f'Time-series: Views + Augmentations + Target '
            f'(Sensor {sensor_idx}, V={V}, t0_views={len(t0_indices)}, Stage: {stage.upper()})'
        ),
        xaxis_title='Relative time steps (t0 start = 0)',
        yaxis_title='Scaled sensor values',
    )

    # Display control: if show=True the figure will be rendered and the function
    # returns None to avoid duplicated display in notebooks. If show=False the
    # function returns the figure object for programmatic use.
    if show:
        fig.show()
        return None
    return fig