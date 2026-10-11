import torch
from torch.utils.data import DataLoader
from typing import Tuple

from rop.config import DatasetConfig
from rop.data.split import load_all_files, create_splits
from rop.data.dataset import ROPDataset
from rop.data.transforms import get_train_transforms, get_eval_transforms

class ROPDataModule:
    """Manages dataset instantiation and DataLoaders."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def setup(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        all_files, all_labels = load_all_files(self.config.data_dir)
        
        if not all_files:
            raise ValueError(f"No valid images found in {self.config.data_dir}")
            
        train_f, train_l, val_f, val_l, test_f, test_l = create_splits(
            all_files, all_labels, seed=self.config.seed
        )
        
        train_dataset = ROPDataset(train_f, train_l, transform=get_train_transforms(self.config.img_size))
        val_dataset = ROPDataset(val_f, val_l, transform=get_eval_transforms(self.config.img_size))
        test_dataset = ROPDataset(test_f, test_l, transform=get_eval_transforms(self.config.img_size))
        
        # Use pinned memory if CUDA is available, check properly
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
