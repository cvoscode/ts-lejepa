import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric_temporal.nn.recurrent import GConvLSTM

# ============================================================
# FIXED VERSION – dimensionally consistent GConvLSTM encoder
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(norm_layer(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.last_layer = nn.Linear(in_dim, output_dim)
        self.last_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.last_layer(x)
        return self.last_norm(x)


class GNN_TimeSeriesEncoder(nn.Module):
    """
    TRUE spatiotemporal encoder.

    Expected input:
        x: [B, T, N, F]

    Output:
        graph embedding: [B, encoder_output_dim]
    """
    def __init__(self,
                 input_channels: int,   # F
                 hidden_channels: int,
                 encoder_output_dim: int,
                 K: int = 2,
                 dropout: float = 0.0,
                 readout: str = "mean"):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.readout = readout

        self.gconv_lstm = GConvLSTM(
            in_channels=input_channels,
            out_channels=hidden_channels,
            K=K
        )

        self.dropout = nn.Dropout(dropout)

        self.encoder_head = nn.Sequential(
            nn.Linear(hidden_channels, encoder_output_dim),
            nn.LayerNorm(encoder_output_dim)
        )

    def _readout(self, h):
        if self.readout == "mean":
            return h.mean(dim=1)
        raise ValueError("Unsupported readout")

    def forward(self, x, edge_index, edge_weight=None):
        # x: [B, T, N, F]
        B, T, N, F = x.shape

        h, c = None, None
        for t in range(T):
            x_t = x[:, t]                    # [B, N, F]
            x_t = x_t.reshape(B * N, F)      # [B*N, F]

            if h is not None:
                h = h.view(B * N, self.hidden_channels)
                c = c.view(B * N, self.hidden_channels)

            

            h, c = self.gconv_lstm(x_t, edge_index, edge_weight, h, c)


            h = self.dropout(h)

        h = h.view(B, N, self.hidden_channels)     # [B, N, H]
        graph_latent = self._readout(h)            # [B, H]
        return self.encoder_head(graph_latent)     # [B, D]
    


class TimeSeriesGraphLSTMEncoder(nn.Module):
    """
    Correct JEPA-style wrapper.

    Input:
        x: [B, V, C, T]

    Interpretation:
        V = independent views (ensembles)
        C = nodes
        T = time
        F = 1 (scalar per node)
    """
    def __init__(self,
                 gnn_hidden_channels: int,
                 encoder_output_dim: int = 12,
                 proj_dim: int = 128,
                 K: int = 2,
                 edge_index: torch.Tensor | None = None,
                 edge_weight: torch.Tensor | None = None):
        super().__init__()

        self.encoder_output_dim = encoder_output_dim
        # LeJEPA_Forecaster expects the backbone to expose .output_dim
        self.output_dim = encoder_output_dim

        # Optional: store graph structure on the module so forward(x) works.
        # If you don't pass edge_index here, you must pass it to forward(...).
        if edge_index is not None:
            self.register_buffer("edge_index", edge_index)
        else:
            self.edge_index = None
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        self.encoder = GNN_TimeSeriesEncoder(
            input_channels=1,
            hidden_channels=gnn_hidden_channels,
            encoder_output_dim=encoder_output_dim,
            K=K
        )

        self.proj = MLP(
            input_dim=encoder_output_dim,
            hidden_dims=[proj_dim, proj_dim],
            output_dim=proj_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        time_features: torch.Tensor | None = None,
        *,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        return_proj: bool = False,
    ):
        """Encode spatiotemporal windows.

        LeJEPA_Forecaster will call `forward(x, time_features=None)` with x shaped
        like [B*V, C, T] (or [B, V, C, T] if you call it directly). This model
        also needs a graph `edge_index`; you can provide it at init or per-call.
        """
        _ = time_features  # accepted for API compatibility; not used

        if edge_index is None:
            edge_index = getattr(self, "edge_index", None)
        if edge_weight is None:
            edge_weight = getattr(self, "edge_weight", None)
        if edge_index is None:
            raise ValueError("TimeSeriesGraphLSTMEncoder requires edge_index (pass at init or forward(..., edge_index=...)).")

        # Normalize input shapes.
        if x.dim() == 3:
            # Treat as [B*V, C, T] with scalar feature per node.
            BV, C, T = x.shape
            B, V = BV, 1
            x = x.view(B, V, C, T)
        elif x.dim() != 4:
            raise ValueError(f"Expected x to have 3 or 4 dims, got shape={tuple(x.shape)}")

        # x: [B, V, C, T]
        B, V, C, T = x.shape

        # Reorder to true spatiotemporal format
        # [B, V, C, T] -> [B*V, T, C, 1]
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B * V, T, C, 1)
        # Bv = B * V
        # edge_index, edge_weight = self.expand_edge_index(
        #     edge_index, edge_weight, C, Bv, x.device
        # )
        emb_flat = self.encoder(x, edge_index, edge_weight)  # [B*V, D]
        # Keep JEPA-consistent semantics:
        # - if caller provided [B, V, ...], return per-view embeddings as [B, V, D]
        # - if caller provided [B*V, ...] (V inferred as 1), return [B*V, D]
        if V == 1:
            emb = emb_flat
        else:
            emb = emb_flat.view(B, V, self.encoder_output_dim)

        if return_proj:
            proj = self.proj(emb_flat)
            return emb, proj
        return emb
    # def expand_edge_index(self,edge_index, edge_weight, num_nodes, batch_size, device):
    #     edge_indices = []
    #     edge_weights = []

    #     for i in range(batch_size):
    #         offset = i * num_nodes
    #         ei = edge_index + offset
    #         edge_indices.append(ei)
    #         edge_weights.append(edge_weight)

    #     edge_index = torch.cat(edge_indices, dim=1).to(device)
    #     edge_weight = torch.cat(edge_weights).to(device)
    #     return edge_index, edge_weight


# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------
if __name__ == '__main__':
    B, V, C, T = 4, 3, 16, 10
    x = torch.randn(B, V, C, T)

    edge_index = torch.empty((2, 0), dtype=torch.long)

    model = TimeSeriesGraphLSTMEncoder(
        gnn_hidden_channels=64,
        encoder_output_dim=8,
        proj_dim=16
    )

    emb, proj = model(x, edge_index)
    print(emb.shape)   # [B, 8]
    print(proj.shape)  # [B, 16]
