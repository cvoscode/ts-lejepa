import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
def visualize_pems_tuple(
    datamodule: 'PeMS08DataModule',
    stage= 'train',
    batch_index: int = 0,
    sensor_index: int = 0,
):
    """
    Visualisiert die drei Eingabefenster (t-1, t0, t+1) und das Target-Fenster.
    """
    dataloader = datamodule.train_dataloader() if stage == 'train' else datamodule.val_dataloader()
    # 2. Den ersten Batch abrufen
    try:
        views_batch, target_batch = next(iter(dataloader))
    except (StopIteration, Exception) as e:
        print(f"Fehler beim Laden des Batches: {e}")
        return

    if batch_index >= views_batch.size(0):
        print(f"Fehler: Batch-Index {batch_index} ist außerhalb des Bereichs (Batch-Größe: {views_batch.size(0)})")
        return
    
    # Extrahiere Parameter
    L_in = datamodule.cfg.window_size
    L_target = datamodule.cfg.target_window_size
    temporal_shift = getattr(datamodule.cfg, "temporal_shift",0)

    print(f"--- Visualisierung ({stage}-Daten) ---")
    print(f"Fensterlängen: L_in={L_in}, Target={L_target}. | Temporal Shift: {temporal_shift}")
    
    # 3. Visualisierung
    fig = go.Figure()
    
    # base_start in the dataset is temporal_shift.
    # In the visualization, we can set t0 start to 0 for convenience.
    # Then t-1 starts at -temporal_shift, and t+1 starts at +temporal_shift.
    
    time_prev = np.arange(-temporal_shift, -temporal_shift + L_in)
    time_curr = np.arange(0, L_in)
    time_next = np.arange(temporal_shift, temporal_shift + L_in)
    time_target = np.arange(L_in, L_in + L_target)

    # target_batch shape: [B, L_target, C] (transposed in dataset)
    v_prev = views_batch[batch_index, 0, sensor_index-1, :].cpu().numpy()
    v_curr = views_batch[batch_index, 1, sensor_index-1, :].cpu().numpy()
    v_next = views_batch[batch_index, 2, sensor_index-1, :].cpu().numpy()
    target = target_batch[batch_index, :, sensor_index-1].cpu().numpy()

    fig.add_trace(go.Scatter(x=time_prev, y=v_prev, name='T-1 (Shifted Back)', line=dict(color='blue', width=2, dash='dash'), opacity=0.5))
    fig.add_trace(go.Scatter(x=time_curr, y=v_curr, name='T0 (Reference)', line=dict(color='green', width=2), opacity=0.8))
    fig.add_trace(go.Scatter(x=time_next, y=v_next, name='T+1 (Shifted Forward)', line=dict(color='red', width=2, dash='dash'), opacity=0.5))
    
    fig.add_trace(go.Scatter(x=time_target, y=target, name='Target (Future of T0)', line=dict(color='black', width=2, dash='dash'), opacity=0.5))

    # Vertikale Linien für Fenster-Grenzen
    fig.add_vline(x=0, line_color='gray', line_width=2, line_dash='dash', opacity=0.3, annotation_text='Start T0')
    fig.add_vline(x=L_in, line_color='red', line_width=2, line_dash='dash', opacity=0.5, annotation_text='Ende T0 / Start Target')

    # 5. Titel und Achsenbeschriftungen
    fig.update_layout(title=f'PeMS-Zeitreihe: Multi-Timestamp Views (Sensor {sensor_index}, Stage: {stage.upper()})',
                      xaxis_title='Relativer Zeitschritt (t0 Start = 0)',
                      yaxis_title='Skalierter Verkehrsfluss')
    
    fig.show()