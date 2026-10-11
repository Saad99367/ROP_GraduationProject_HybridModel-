import os
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

FOLDER_TO_LABEL = {
    'Normal': 0, 
    'Stage_1': 1,
    'Stage_2': 2, 
    'Stage_3': 3, 
    'Stage_4': 4,
}

CLASS_NAMES = {
    0: 'No ROP (Normal)',
    1: 'Mild ROP (Stage 1)',
    2: 'Moderate ROP (Stage 2)',
    3: 'Severe ROP (Stage 3)',
    4: 'Critical ROP (Stage 4)',
}

def load_all_files(data_dir: str) -> Tuple[List[str], List[int]]:
    """Loads file paths and labels from the directory structure."""
    all_files = []
    all_labels = []
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist.")
        return [], []

    for folder_name, label in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            logger.warning(f"Missing class folder: {folder_path}")
            continue
            
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                all_files.append(os.path.join(folder_path, fname))
                all_labels.append(label)
                
    return all_files, all_labels

def create_splits(
    all_files: List[str], 
    all_labels: List[int], 
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Creates stratified train, validation, and test splits.
    
    WARNING: This performs a random stratified split over all images. 
    If the dataset contains multiple images per patient (e.g., both eyes, 
    multiple visits), this will leak patient-specific features across splits.
    Patient-level grouping should be used if metadata is available.
    """
    
    # First split off the test set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels, 
        test_size=test_size, 
        stratify=all_labels, 
        random_state=seed
    )
    
    # Calculate relative validation size
    relative_val_size = val_size / (1.0 - test_size)
    
    # Split train and val
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, 
        test_size=relative_val_size, 
        stratify=train_val_labels, 
        random_state=seed
    )
    
    return train_files, train_labels, val_files, val_labels, test_files, test_labels
