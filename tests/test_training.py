import torch
import pytest
import os
import shutil

from rop.config import Config
from rop.models.hybrid_model import HybridFuzzyROPModel
from rop.training.trainer import Trainer
from rop.data.dataset import ROPDataset
from torch.utils.data import DataLoader

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=16, img_size=224):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return torch.randn(3, self.img_size, self.img_size), torch.tensor(idx % 5, dtype=torch.long)

def test_synthetic_training_smoke():
    config = Config()
    config.training.epochs = 2
    config.training.freeze_backbone_epochs = 1
    config.training.save_dir = "artifacts/test_checkpoints"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridFuzzyROPModel(pretrained_backbone=False, n_classes=5)
    
    train_dataset = DummyDataset(size=8)
    val_dataset = DummyDataset(size=8)
    
    train_loader = DataLoader(train_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    results = trainer.fit()
    
    # Ensure checkpoints are saved
    assert os.path.exists(f"{config.training.save_dir}/best_model.pth")
    assert os.path.exists(f"{config.training.save_dir}/final_model.pth")
    
    # Cleanup
    shutil.rmtree(config.training.save_dir, ignore_errors=True)
