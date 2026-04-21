from __future__ import annotations

from math import prod

import torch
from torch import nn


class ConvBlock(nn.Module):
    """A lightweight convolutional block for 2D observation maps."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class CompactCnnEstimator(nn.Module):
    """Compact CNN regressor for cascaded channel estimation."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
        *,
        conv_channels: tuple[int, ...] = (32, 64, 64),
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape

        feature_blocks: list[nn.Module] = []
        in_channels = input_shape[0]
        for out_channels in conv_channels:
            feature_blocks.append(ConvBlock(in_channels, out_channels, dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*feature_blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feature_dim = int(self.features(dummy).numel())

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prod(output_shape)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        outputs = self.regressor(features)
        return outputs.view(inputs.shape[0], *self.output_shape)
