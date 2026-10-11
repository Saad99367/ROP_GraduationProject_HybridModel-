import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import List, Tuple, Callable, Optional

class ROPDataset(Dataset):
    """
    Retinopathy of Prematurity Dataset.
    Loads images and labels from a list of paths.
    """
    def __init__(self, files: List[str], labels: List[int], transform: Optional[Callable] = None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.files[idx]
        
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image, label
