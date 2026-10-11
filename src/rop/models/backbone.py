import torch.nn as nn
import timm

class EfficientNetB3Backbone(nn.Module):
    """Pretrained EfficientNet-B3 feature extractor."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # We use features_only=True to extract intermediate spatial feature maps.
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)
        # feature_info[-1] gives the properties of the last stage.
        self.out_channels = self.backbone.feature_info[-1]['num_chs']

    def forward(self, x):
        # Return the features from the final stage
        return self.backbone(x)[-1]
