import numpy as np
from matplotlib import pyplot as plt
def visualize_pems_tuple(
    datamodule: 'PeMS08DataModule',
    stage= 'train',
    batch_index: int = 0,
    sensor_index: int = 0,
):
    """
    Visualisiert das t0-Eingabefenster (mit allen V augmentierten Views), 
    das t1-Fenster (falls im Dataset enthalten) und das Target-Fenster.
    """
    
    # 1. Daten vorbereiten und Setup durchführen
    datamodule.prepare_data()
    datamodule.setup(stage=stage)
    
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
    V_config = datamodule.cfg.V

    # Ermittle, ob t1-Fenster vorhanden sind (relevant für Training)
    is_t1_present = datamodule.train_ds.t1_window if stage == 'train' else datamodule.val_ds.t1_window
    
    total_views_in_batch = views_batch.size(1) 
    num_views_per_window = V_config # Die Anzahl der Views, die pro Fenster (t0/t1) generiert wurden
    
    print(f"--- Visualisierung ({stage}-Daten) ---")
    print(f"Fensterlängen: T0/T1={L_in}, Target={L_target}. | Views pro Fenster (V): {num_views_per_window}. | T1 enthalten: {is_t1_present}.")
    print(f"Tatsächliche Anzahl Views im Batch: {total_views_in_batch}")
    
    # 3. Visualisierung
    plt.figure(figsize=(14, 6))
    
    colors_t0 = plt.cm.Blues(np.linspace(0.4, 0.8, num_views_per_window))
    colors_t1 = plt.cm.Reds(np.linspace(0.4, 0.8, num_views_per_window))
    
    # --- Plot t0 Views ---
    t0_views_end = num_views_per_window
    
    # Slice für t0-Views: von 0 bis V
    t0_views = views_batch[batch_index, 0:t0_views_end, sensor_index, :].cpu().numpy()
    time_t0 = np.arange(L_in)
    
    if len(t0_views) == 0:
        print("WARNUNG: Keine T0-Views gefunden. Kann nicht visualisiert werden.")
        return

    for i, view in enumerate(t0_views):
        label = f'T0 View {i+1} (Augmented)'
        plt.plot(time_t0, view, label=label, color=colors_t0[i % len(colors_t0)], alpha=0.7, linewidth=1.0)
    
    # --- Plot t1 Views (falls vorhanden) ---
    t1_views = np.array([]) # Standardmäßig leeres Array
    if is_t1_present:
        t1_views_start = num_views_per_window
        
        # Slice für t1-Views: von V bis total_views
        t1_views = views_batch[batch_index, t1_views_start:total_views_in_batch, sensor_index, :].cpu().numpy()
        
        if len(t1_views) > 0:
            time_t1 = np.arange(L_in, L_in + L_in)
            
            for i, view in enumerate(t1_views):
                label = f'T1 View {i+1} (Augmented)'
                plt.plot(time_t1, view, label=label, color=colors_t1[i % len(colors_t1)], alpha=0.7, linestyle='--', linewidth=1.0)
                
            # Vertikale Linie zwischen t0 und t1
            plt.axvline(x=L_in - 0.5, color='green', linestyle=':', label='Ende T0 / Start T1')
            target_start_time = L_in + L_in # Target beginnt nach t1
        else:
            # Falls t1_window=True, aber keine t1-Daten im Batch
            print("WARNUNG: T1-Fenster erwartet, aber keine T1-Views im Batch gefunden.")
            target_start_time = L_in # Target beginnt nach t0
    else:
        # Falls t1_window=False
        target_start_time = L_in # Target beginnt nach t0
        
    # Vertikale Linie nach dem letzten Input-Fenster (entweder t0 oder t1)
    plt.axvline(x=target_start_time - 0.5, color='red', linestyle=':', label='Ende Input / Start Target')


    # --- Plot Target Window ---
    target_sample = target_batch[batch_index, sensor_index, :].cpu().numpy()
    time_target = np.arange(target_start_time, target_start_time + L_target)
    
    plt.plot(time_target, target_sample, label=f'Target (Clean/Unaugmented)', color='black', marker='o', markersize=3, linewidth=2.0)
    
    # 4. Verbindung zur letzten Zeitreihe
    # Die letzte View zum Target verbinden
    if len(t1_views) > 0:
        last_view_data = t1_views[-1] 
    elif len(t0_views) > 0:
        last_view_data = t0_views[-1]
    else:
        last_view_data = None # Sollte durch die frühere Prüfung abgefangen werden

    if last_view_data is not None:
        plt.plot([target_start_time-1, target_start_time], 
                 [last_view_data[-1], target_sample[0]], 
                 color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    
    # 5. Titel und Achsenbeschriftungen
    plt.title(f'PeMS-Zeitreihe: Augmentierte T0/T1 Views und Target (Sensor {sensor_index}, Stage: {stage.upper()})')
    plt.xlabel('Zeitschritt')
    plt.ylabel('Skalierter Verkehrsfluss')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()