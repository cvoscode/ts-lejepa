import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import umap
from torch.utils.data import DataLoader

class VisualizationCallback(L.Callback):
    def __init__(self, num_samples_plot=3, umap_every_n_epochs=1, num_umap_batches=10):
        """
        Args:
            num_samples_plot: Number of samples (time series) to plot.
            umap_every_n_epochs: UMAP is expensive, so only run it every N epochs.
            num_umap_batches: How many batches to collect for UMAP (more = more accurate, but slower).
        """
        super().__init__()
        self.num_samples_plot = num_samples_plot
        self.umap_every_n_epochs = umap_every_n_epochs
        self.num_umap_batches = num_umap_batches

    def on_validation_epoch_end(self, trainer, pl_module):
        # 1. Get data (take a subset of the val set)
        if trainer.datamodule is None:
            return
        
        val_loader = trainer.datamodule.val_dataloader()
        
        # Containers for UMAP
        all_embeddings = []
        all_times = [] # Time index for coloring
        
        # Container for Prediction Plots (only from the first batch)
        plot_data = None
        
        device = pl_module.device
        pl_module.eval() # Ensure eval mode

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= self.num_umap_batches:
                    break
                
                # Batch extraction: (views, target)
                # target shape: [B, C, L_target]
                # views shape: [B, V, C, L_input]
                vs, target = batch
                vs = vs.to(device)
                target = target.to(device)
                
                # Forward Pass
                # Use only the first view for inference in val mode
                # The model might return (emb, proj). We only need emb.
                out = pl_module(vs)
                if isinstance(out, tuple):
                    emb = out[0]
                else:
                    emb = out
                
                # emb shape: [B, V, D] (from pl_module.forward) or [B, D]
                # We want the first view (usually 'current' t0) for UMAP
                if emb.ndim == 3: 
                    # Assuming shape [B, V, D], take the middle view (V=1 is t0)
                    # or just view 0 if V=1. 
                    v_idx = emb.shape[1] // 2 
                    emb_flat = emb[:, v_idx, :] # [B, D]
                else:
                    emb_flat = emb 

                yhat = pl_module.probe(emb_flat)
                
                # Collect embeddings for UMAP
                all_embeddings.append(emb_flat.cpu().numpy())
                
                # Time index simulation
                batch_size = vs.shape[0]
                time_indices = np.arange(i * batch_size, (i + 1) * batch_size)
                all_times.append(time_indices)
                
                # Store plot data (only from the first batch)
                if i == 0:
                    # Input sequence for View 0
                    input_seq = vs[:, 0, :, :] 
                    
                    # Inverse scaling
                    if hasattr(pl_module, 'scaler') and pl_module.scaler is not None:
                        input_inv = pl_module.scaler.inverse_transform(input_seq)
                        target_inv = pl_module.scaler.inverse_transform(target)
                        yhat_inv = pl_module.scaler.inverse_transform(yhat)
                    else:
                        input_inv = input_seq
                        target_inv = target
                        yhat_inv = yhat
                        
                    plot_data = {
                        'input': input_inv.cpu(),
                        'target': target_inv.cpu(),
                        'prediction': yhat_inv.cpu()
                    }

        # --- A. Plot Predictions ---
        if plot_data is not None:
            self._plot_predictions(trainer, plot_data)

        # --- B. Plot UMAP (every N epochs) ---
        if (trainer.current_epoch + 1) % self.umap_every_n_epochs == 0 and len(all_embeddings) > 0:
            embeddings_concat = np.concatenate(all_embeddings, axis=0)
            times_concat = np.concatenate(all_times, axis=0)
            self._plot_umap(trainer, embeddings_concat, times_concat)

    def _plot_predictions(self, trainer, data):
        """ Creates Matplotlib plots for time series """
        inputs = data['input']      # Expected [B, C, L_in]
        targets = data['target']    # Might be [B, L_out, C] or [B, C, L_out]
        preds = data['prediction']  # Might be [B, L_out, C] or [B, C, L_out]
        
        # Ensure everything is [B, C, L] based on inputs shape
        C = inputs.shape[1]
        
        if targets.shape[1] != C and targets.ndim == 3:
            targets = targets.transpose(1, 2)
        if preds.shape[1] != C and preds.ndim == 3:
            preds = preds.transpose(1, 2)
        
        num_samples = min(self.num_samples_plot, inputs.shape[0])
        sensor_idx = np.random.randint(0, C)
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples), sharex=False)
        if num_samples == 1: axes = [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            
            # Create time axes
            L_in = inputs.shape[-1]
            L_out = targets.shape[-1]
            
            t_in = range(0, L_in)
            t_out = range(L_in, L_in + L_out) # Target follows input
            
            # Data for sample i and chosen sensor
            seq_in = inputs[i, sensor_idx, :]
            seq_target = targets[i, sensor_idx, :]
            seq_pred = preds[i, sensor_idx, :]
            
            ax.plot(t_in, seq_in, label='Input History', color='gray', linestyle='--')
            ax.plot(t_out, seq_target, label='Ground Truth', color='green')
            ax.plot(t_out, seq_pred, label='Prediction', color='red')
            
            ax.set_title(f"Sample {i} - Sensor {sensor_idx}")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to Tensorboard
        if trainer.logger and hasattr(trainer.logger.experiment, 'add_figure'):
            trainer.logger.experiment.add_figure("Validation/Predictions", fig, global_step=trainer.global_step)
        plt.close(fig)

    def _plot_umap(self, trainer, embeddings, time_indices):
        """ Computes and plots UMAP """
        try:
            # Flatten embeddings if necessary
            if embeddings.ndim > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
                
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
            embedding_2d = reducer.fit_transform(embeddings)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                embedding_2d[:, 0], 
                embedding_2d[:, 1], 
                c=time_indices, 
                cmap='viridis', 
                s=10, 
                alpha=0.6
            )
            ax.set_title("UMAP of Embeddings (Color = Time Index)")
            plt.colorbar(scatter, label='Time Index within Validation Subset')
            ax.grid(True, alpha=0.3)
            
            if trainer.logger and hasattr(trainer.logger.experiment, 'add_figure'):
                trainer.logger.experiment.add_figure("Validation/UMAP_Embedding", fig, global_step=trainer.global_step)
            plt.close(fig)
            
        except Exception as e:
            print(f"UMAP visualization failed: {e}")
