import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class DatasetConfig:
    data_dir: str = "Dataset"
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42

@dataclass
class ModelConfig:
    n_classes: int = 5
    pretrained_backbone: bool = True
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    ff_dim: int = 2048
    dropout: float = 0.1
    classifier_hidden: int = 256
    fuzzy_features: int = 64
    n_fuzzy_terms: int = 3

@dataclass
class TrainingConfig:
    epochs: int = 50
    freeze_backbone_epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 10
    accum_steps: int = 2
    save_dir: str = "model_checkpoints"

@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        dataset = DatasetConfig(**data.get("dataset", {}))
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))
        
        return cls(dataset=dataset, model=model, training=training)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
        }
