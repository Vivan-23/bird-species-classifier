"""Custom Convolutional Neural Network (CNN) architecture in PyTorch for Bird Species Classification.

Designed specifically for academic demonstration, interpretability, and robust performance
on 224x224 RGB image inputs without relying on external pretrained backbones.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Reusable Convolutional Block comprising Conv2d, Batch Normalization, ReLU, and Max Pooling.

    Architecture per block:
        Conv2d (3x3 kernel, stride=1, padding=1) -> Preserves spatial dimensions
        BatchNorm2d -> Stabilizes activations and accelerates convergence
        ReLU -> Introduces element-wise non-linearity
        MaxPool2d (2x2 kernel, stride=2) -> Halves spatial dimensions
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BirdClassifierCNN(nn.Module):
    """Custom Convolutional Neural Network for Bird Species Classification.

    Tensor Flow Progression for input [Batch, 3, 224, 224]:
      1. Stage 1 (Block 1): [B, 3, 224, 224]   -> ConvBlock(3, 32)   -> [B, 32, 112, 112]
      2. Stage 2 (Block 2): [B, 32, 112, 112]  -> ConvBlock(32, 64)  -> [B, 64, 56, 56]
      3. Stage 3 (Block 3): [B, 64, 56, 56]   -> ConvBlock(64, 128) -> [B, 128, 28, 28]
      4. Stage 4 (Block 4): [B, 128, 28, 28]  -> ConvBlock(128, 256)-> [B, 256, 14, 14]
      5. Adaptive Pooling:  [B, 256, 14, 14]  -> AdaptiveAvgPool2d((4, 4)) -> [B, 256, 4, 4]
      6. Flatten:           [B, 256, 4, 4]    -> [B, 4096]
      7. Dense FC Stage:    [B, 4096]         -> Linear(4096, 256) + ReLU -> [B, 256]
      8. Regularization:    [B, 256]          -> Dropout(p=dropout_prob)
      9. Output Logits:     [B, 256]          -> Linear(256, num_classes) -> [B, num_classes]

    Important Design Decisions:
      - Uses raw logits output (no internal Softmax) for numerical stability with torch.nn.CrossEntropyLoss.
      - Supports dynamic num_classes rather than hardcoding.
      - Includes Dropout in the classification head for structural regularization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout_prob: float = 0.4,
        init_weights: bool = True
    ):
        super(BirdClassifierCNN, self).__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        # Feature Extraction Stage
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),    # -> [B, 32, 112, 112]
            ConvBlock(32, 64),             # -> [B, 64, 56, 56]
            ConvBlock(64, 128),            # -> [B, 128, 28, 28]
            ConvBlock(128, 256),           # -> [B, 256, 14, 14]
        )

        # Spatial Aggregation Stage
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # -> [B, 256, 4, 4]
        self.flatten = nn.Flatten()

        # Classification Head
        flattened_dim = 256 * 4 * 4  # 4096 features
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming (He) normal initialization to conv/linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


def build_model(config: dict) -> BirdClassifierCNN:
    """Factory helper to build BirdClassifierCNN from project configuration."""
    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = config["dataset"].get("num_classes", 10)
    dropout_prob = model_cfg.get("dropout_prob", 0.4)
    init_weights = model_cfg.get("init_weights", True)

    model = BirdClassifierCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        init_weights=init_weights
    )
    return model
