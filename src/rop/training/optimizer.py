import torch
from typing import Iterator

def create_optimizer(
    model: torch.nn.Module, 
    lr: float, 
    weight_decay: float,
    train_backbone: bool = False,
    backbone_lr_multiplier: float = 0.1
) -> torch.optim.Optimizer:
    """
    Creates an AdamW optimizer. 
    If train_backbone is True, the backbone is included with a scaled learning rate.
    Otherwise, only non-backbone parameters with requires_grad=True are included.
    """
    if train_backbone:
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': lr * backbone_lr_multiplier},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': lr}
        ]
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        # Filter explicitly requires_grad
        params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def update_optimizer_for_unfreezing(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    lr: float,
    backbone_lr_multiplier: float = 0.1
):
    """
    Adds backbone parameters to an existing optimizer to preserve Adam states.
    """
    # Verify backbone is unfrozen
    for param in model.backbone.parameters():
        param.requires_grad = True
        
    optimizer.add_param_group({
        'params': list(model.backbone.parameters()),
        'lr': lr * backbone_lr_multiplier
    })
