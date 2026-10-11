import torch
import torch.nn as nn

from rop.models.backbone import EfficientNetB3Backbone
from rop.models.transformer import TransformerEncoder
from rop.models.fuzzy_layer import FuzzyClassificationLayer

class HybridFuzzyROPModel(nn.Module):
    """
    Full Hybrid Architecture: EfficientNet-B3 + Transformer + Fuzzy Logic.
    
    Args:
        n_classes (int): Number of output classes (5 for ROP staging).
        pretrained_backbone (bool): Whether to use ImageNet-pretrained weights.
        d_model (int): Transformer embedding dimension.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of Transformer encoder layers.
        ff_dim (int): Feed-forward dimension in Transformer.
        dropout (float): Dropout rate.
        classifier_hidden (int): Hidden dimension of the FC layer before fuzzy logic.
        fuzzy_features (int): Input dimension to the Fuzzy Logic layer.
        n_fuzzy_terms (int): Number of fuzzy membership terms per feature.
    """
    def __init__(
        self, 
        n_classes: int = 5, 
        pretrained_backbone: bool = True, 
        d_model: int = 512,
        n_heads: int = 8, 
        n_layers: int = 2, 
        ff_dim: int = 2048, 
        dropout: float = 0.1,
        classifier_hidden: int = 256, 
        fuzzy_features: int = 64, 
        n_fuzzy_terms: int = 3
    ):
        super().__init__()
        
        # 1. Feature Extraction Backbone
        self.backbone = EfficientNetB3Backbone(pretrained=pretrained_backbone)
        
        # 2. Transformer Encoder
        self.transformer = TransformerEncoder(
            in_channels=self.backbone.out_channels, 
            d_model=d_model,
            n_heads=n_heads, 
            n_layers=n_layers, 
            ff_dim=ff_dim, 
            dropout=dropout,
            # max_spatial depends on input size. For 224x224 and stride 32, it's 7x7=49.
            # We set 256 as a safe upper bound.
            max_spatial=256
        )
        
        # 3. Intermediate Classifier / Projection
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, fuzzy_features), 
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 4. Fuzzy Logic Output Layer
        self.fuzzy = FuzzyClassificationLayer(
            in_features=fuzzy_features, 
            n_classes=n_classes, 
            n_terms=n_fuzzy_terms
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W).
            
        Returns:
            Raw class logits of shape (B, n_classes).
        """
        # (B, 3, H, W) -> (B, C, H', W')
        features = self.backbone(x)
        
        # (B, C, H', W') -> (B, d_model)
        encoded = self.transformer(features)
        
        # (B, d_model) -> (B, fuzzy_features)
        projected = self.classifier(encoded)
        
        # (B, fuzzy_features) -> (B, n_classes)
        return self.fuzzy(projected)
