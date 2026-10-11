import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Dict

from rop.models.hybrid_model import HybridFuzzyROPModel
from rop.data.transforms import get_eval_transforms
from rop.training.checkpoint import load_checkpoint

class ROPPredictor:
    """Inference wrapper for the ROP classification model."""
    def __init__(self, checkpoint_path: str, device: torch.device = torch.device('cpu')):
        self.device = device
        
        # Load raw dict to instantiate the model first
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        config_dict = checkpoint['config']
        m_cfg = config_dict.get('model', config_dict)
        self.class_names = checkpoint['class_names']
        
        self.model = HybridFuzzyROPModel(
            n_classes=m_cfg['n_classes'],
            pretrained_backbone=False,
            d_model=m_cfg['d_model'],
            n_heads=m_cfg['n_heads'],
            n_layers=m_cfg['n_layers'],
            ff_dim=m_cfg['ff_dim'],
            classifier_hidden=m_cfg['classifier_hidden'],
            fuzzy_features=m_cfg['fuzzy_features'],
            n_fuzzy_terms=m_cfg['n_fuzzy_terms']
        ).to(device)
        
        load_checkpoint(checkpoint_path, self.model, device=device)
        self.model.eval()
        
        # Preprocessing
        if 'dataset' in config_dict:
            img_size = config_dict['dataset']['img_size']
        else:
            img_size = config_dict.get('img_size', 224)
        self.transform = get_eval_transforms(img_size)
        
    @torch.inference_mode()
    def predict(self, image_path: str) -> Tuple[int, str, Dict[str, float]]:
        """
        Predicts ROP stage for a given image.
        Returns:
            class_idx: Integer class index
            class_name: String class name
            probabilities: Dictionary mapping class name to probability
        """
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to read image {image_path}: {e}")
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            logits = self.model(tensor)
            
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        class_idx = int(np.argmax(probs))
        
        prob_dict = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        
        return class_idx, self.class_names[class_idx], prob_dict
