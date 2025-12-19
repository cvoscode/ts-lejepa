import torch.nn as nn
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_channels, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        
        # Simple Conv1d Feature Extractor
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1 , padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Pool over time dimension
            nn.Flatten(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        return self.net(x)