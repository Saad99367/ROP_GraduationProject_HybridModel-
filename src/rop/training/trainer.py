import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import logging
from sklearn.metrics import accuracy_score, f1_score

from rop.config import Config
from rop.training.checkpoint import save_checkpoint
from rop.training.optimizer import create_optimizer, update_optimizer_for_unfreezing

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.t_cfg = config.training
        
        # Loss function
        # Calculate class weights if needed, for simplicity we use equal weights or no weights here, 
        # but the prompt mentions it. Let's stick to standard CrossEntropy with label smoothing.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.t_cfg.label_smoothing)
        
        # Initially freeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        self.optimizer = create_optimizer(
            self.model, 
            lr=self.t_cfg.lr, 
            weight_decay=self.t_cfg.weight_decay,
            train_backbone=False
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.t_cfg.epochs, eta_min=self.t_cfg.lr * 0.01
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
        self.history = []
        self.best_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.t_cfg.epochs} [Train]", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels) / self.t_cfg.accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels) / self.t_cfg.accum_steps
                loss.backward()
                
            if (batch_idx + 1) % self.t_cfg.accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
            running_loss += loss.item() * self.t_cfg.accum_steps * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix(loss=f"{running_loss / len(all_labels):.4f}")
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return epoch_loss, acc, f1

    def validate(self) -> tuple[float, float, float]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.inference_mode():
            for images, labels in tqdm(self.val_loader, desc="[Val]", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                running_loss += loss.item() * images.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return epoch_loss, acc, f1

    def fit(self) -> dict:
        os.makedirs(self.t_cfg.save_dir, exist_ok=True)
        from rop.data.split import CLASS_NAMES
        
        logger.info("Starting training...")
        
        for epoch in range(1, self.t_cfg.epochs + 1):
            if epoch == self.t_cfg.freeze_backbone_epochs + 1:
                logger.info(f"Unfreezing backbone at epoch {epoch}")
                update_optimizer_for_unfreezing(
                    self.optimizer, self.model, self.t_cfg.lr, backbone_lr_multiplier=0.1
                )
                
            t0 = time.time()
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.validate()
            self.scheduler.step()
            
            elapsed = time.time() - t0
            lr_now = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {lr_now:.6f} | {elapsed:.1f}s"
            )
            
            self.history.append({
                'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
                'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'lr': lr_now
            })
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                
                save_checkpoint(
                    path=os.path.join(self.t_cfg.save_dir, "best_model.pth"),
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    val_metric=val_f1,
                    config=self.config,
                    history=self.history,
                    class_names=CLASS_NAMES
                )
                logger.info(f"   Best model saved! F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.t_cfg.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Save final model
        save_checkpoint(
            path=os.path.join(self.t_cfg.save_dir, "final_model.pth"),
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            val_metric=val_f1,
            config=self.config,
            history=self.history,
            class_names=CLASS_NAMES
        )
        logger.info("Training complete.")
        return {'best_f1': self.best_f1, 'history': self.history}
