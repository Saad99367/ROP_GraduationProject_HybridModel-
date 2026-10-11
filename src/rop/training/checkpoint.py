import os
import torch
from typing import Dict, Any, Optional

def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    val_metric: float,
    config: Any,
    history: list,
    class_names: Dict[int, str]
):
    """Saves a comprehensive training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'scaler_state': scaler.state_dict() if scaler else None,
        'val_metric': val_metric,
        'config': config.to_dict() if hasattr(config, "to_dict") else config,
        'history': history,
        'class_names': class_names
    }
    torch.save(checkpoint, path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """Loads a checkpoint and restores states if objects are provided."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    if optimizer and 'optimizer_state' in checkpoint and checkpoint['optimizer_state']:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    if scheduler and 'scheduler_state' in checkpoint and checkpoint['scheduler_state']:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        
    if scaler and 'scaler_state' in checkpoint and checkpoint['scaler_state']:
        scaler.load_state_dict(checkpoint['scaler_state'])
        
    return checkpoint
