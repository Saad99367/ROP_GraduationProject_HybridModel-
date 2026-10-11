import argparse
import logging
import torch
import sys
import os

# Add the src directory to the python path so 'rop' can be imported without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rop.config import Config
from rop.data.datamodule import ROPDataModule
from rop.models.hybrid_model import HybridFuzzyROPModel
from rop.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train ROP Classification Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    # Set seed
    torch.manual_seed(config.dataset.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data
    datamodule = ROPDataModule(config.dataset)
    train_loader, val_loader, test_loader = datamodule.setup()
    
    logger.info(f"Train/Val/Test sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")
    
    # Model
    model = HybridFuzzyROPModel(
        n_classes=config.model.n_classes,
        pretrained_backbone=config.model.pretrained_backbone,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        ff_dim=config.model.ff_dim,
        dropout=config.model.dropout,
        classifier_hidden=config.model.classifier_hidden,
        fuzzy_features=config.model.fuzzy_features,
        n_fuzzy_terms=config.model.n_fuzzy_terms
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    results = trainer.fit()
    
    # Quick Test Eval (usually done separately, but keeping it simple)
    # We should load best model for test eval
    from rop.training.checkpoint import load_checkpoint
    best_path = f"{config.training.save_dir}/best_model.pth"
    try:
        load_checkpoint(best_path, model, device=device)
        logger.info("Loaded best model for final test evaluation.")
    except Exception as e:
        logger.warning(f"Could not load best model: {e}")
        
    trainer.model = model
    trainer.val_loader = test_loader
    test_loss, test_acc, test_f1 = trainer.validate()
    logger.info(f"Final Test - Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
