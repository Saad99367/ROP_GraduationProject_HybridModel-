import torch
import torch.nn as nn

class FuzzyClassificationLayer(nn.Module):
    """
    Neural Classifier with Fuzzy Membership Features.
    
    This layer computes learnable Gaussian membership functions for input features,
    transforming the features into a 'fuzzy' representation. The flattened memberships
    are then passed through a standard linear classification layer (acting as rule weights).
    """
    def __init__(self, in_features: int, n_classes: int = 5, n_terms: int = 3):
        super().__init__()
        self.in_features = in_features
        self.n_terms = n_terms
        
        # Initialize centers linearly spaced between -1 and 1
        centers = torch.linspace(-1.0, 1.0, n_terms).unsqueeze(0).expand(in_features, -1)
        self.centers = nn.Parameter(centers.clone())
        
        # Initialize widths to a constant value
        self.widths = nn.Parameter(torch.ones(in_features, n_terms) * 0.5)
        
        # Linear layer mapping memberships to class logits
        self.rule_weights = nn.Linear(in_features * n_terms, n_classes)
        self.input_bn = nn.BatchNorm1d(in_features)

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian membership for each feature.
        Args:
            x: Input features (B, in_features)
        Returns:
            memberships: (B, in_features, n_terms)
        """
        x = x.unsqueeze(-1)  # (B, in_features, 1)
        c = self.centers.unsqueeze(0)  # (1, in_features, n_terms)
        
        # Add epsilon outside abs for strict positivity to avoid div by zero
        w = torch.abs(self.widths.unsqueeze(0)) + 1e-6  # (1, in_features, n_terms)
        
        return torch.exp(-0.5 * ((x - c) / w) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize inputs for stable membership computation
        x = self.input_bn(x)
        
        # Fuzzification
        memberships = self.fuzzify(x)
        
        # Flatten and compute final logits
        flat = memberships.reshape(x.size(0), -1)
        return self.rule_weights(flat)
