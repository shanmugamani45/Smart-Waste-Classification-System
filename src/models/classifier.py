import torch
import torch.nn as nn
from typing import Optional


class CustomWasteClassifierHead(nn.Module):
    """
    Standardized classification head with regularization (Dropout & BatchNorm)
    to prevent overfitting during transfer learning on waste classification datasets.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
