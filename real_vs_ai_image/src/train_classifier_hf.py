"""
Real vs AI Image Classifier - HuggingFace Training Script

Uses the HuggingFace transformers-based model (RealVsAIClassifierHF)

Usage:
    python train_classifier_hf.py

Benefits over train_classifier.py:
    - Self-contained model saves (no torch.hub dependency)
"""

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

# Import shared utilities
from training_utils import (
    set_seed,
    get_device,
    RealVsAIClassifierHF,
    get_transforms,
    FocalLoss,
    EarlyStopping,
    train_epoch_single_gpu,
    validate_single_gpu,
    plot_confusion_matrix,
    plot_training_curves
)

# Import data loaders
from image_data_loaders import CIFAKEDataset, PickledCIFAKEDataset


CONFIG = {
    # Model
    'model_name': 'dinov2_vitl14_hf',
    'num_classes': 2,
    'unfreeze_last_n_blocks': 4,
    
    # Data
    'image_size': 518,  # Optimal for DINOv2 ViT-L/14
    'batch_size': 16,   # Adjust based on GPU memory
    'num_workers': 4,
    
    # Dataset loading method
    'use_pickled_splits': True,  # Set to True to use pre-prepared splits
    'max_split': 6,  # Number of splits to load (1-10), only used if use_pickled_splits=True
    'splits_data_dir': 'data/cifake_splits',  # Directory containing pickle files
    
    # Legacy parameters (used when use_pickled_splits=False)
    'max_train_samples': 10000,
    'max_test_samples': 10000,
    
    # Training
    'epochs': 15,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'focal_loss_gamma': 2.0,
    'grad_clip_norm': 1.0,
    
    # Scheduler
    'scheduler_T_0': 5,      # Restart every 5 epochs
    'scheduler_T_mult': 2,   # Double period after restart
    'scheduler_eta_min': 1e-6,
    
    # Early Stopping
    'early_stop_patience': 5,
    'early_stop_min_delta': 0.001,
    
    # Paths
    'checkpoint_dir': 'checkpoints_hf',
    'results_dir': 'results_hf',
    
    # Device - will be set dynamically in main()
    'device': None,
    
    # Reproducibility
    'seed': 42,
}


def save_checkpoint_hf(model, optimizer, scheduler, epoch, metrics, checkpoint_dir, rank=0):
    """Save HuggingFace model checkpoint."""
    if rank != 0:
        return
    
    # Unwrap DDP if needed
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Save model using HuggingFace method (self-contained)
    model_dir = os.path.join(checkpoint_dir, f"model_epoch_{epoch}")
    model_to_save.save_pretrained(model_dir)
    
    # Save training state separately
    training_state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    state_path = os.path.join(checkpoint_dir, f"training_state_epoch_{epoch}.pth")
    torch.save(training_state, state_path)
    
    print(f"✓ Checkpoint saved: {model_dir}")


def main():
    """Main training function."""
    
    # Setup
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Set device (CUDA > MPS > CPU)
    CONFIG['device'] = get_device()
    
    print("=" * 70)
    print("Real vs AI Image Classifier Training (HuggingFace Version)")
    print("=" * 70)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Load data
    print("\n" + "=" * 70)
    print("Loading CIFAKE Dataset")
    print("=" * 70)
    
    if CONFIG['use_pickled_splits']:
        # Use pre-prepared pickle splits
        print(f"Using pre-prepared splits (splits 1-{CONFIG['max_split']})")
        
        train_dataset = PickledCIFAKEDataset(
            split="train",
            split_numbers=list(range(1, CONFIG['max_split'] + 1)),
            transform=get_transforms(is_train=True, target_size=CONFIG['image_size']),
            target_size=(CONFIG['image_size'], CONFIG['image_size']),
            data_dir=CONFIG['splits_data_dir']
        )
        
        val_dataset = PickledCIFAKEDataset(
            split="test",
            split_numbers=list(range(1, CONFIG['max_split'] + 1)),
            transform=get_transforms(is_train=False, target_size=CONFIG['image_size']),
            target_size=(CONFIG['image_size'], CONFIG['image_size']),
            data_dir=CONFIG['splits_data_dir']
        )
    else:
        # Use original HuggingFace dataset loading
        print(f"Using HuggingFace dataset (max samples: {CONFIG['max_train_samples']})")
        
        train_dataset = CIFAKEDataset(
            split="train",
            transform=get_transforms(is_train=True, target_size=CONFIG['image_size']),
            target_size=(CONFIG['image_size'], CONFIG['image_size']),
            max_samples=CONFIG['max_train_samples']
        )
        
        val_dataset = CIFAKEDataset(
            split="test",
            transform=get_transforms(is_train=False, target_size=CONFIG['image_size']),
            target_size=(CONFIG['image_size'], CONFIG['image_size']),
            max_samples=CONFIG['max_test_samples']
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Initialize model (HuggingFace version)
    print("\n" + "=" * 70)
    print("Initializing HuggingFace Model")
    print("=" * 70)
    
    model = RealVsAIClassifierHF(
        num_classes=CONFIG['num_classes'],
        freeze_backbone=True,
        unfreeze_last_n_blocks=CONFIG['unfreeze_last_n_blocks']
    ).to(CONFIG['device'])
    
    # Loss and optimizer
    criterion = FocalLoss(gamma=CONFIG['focal_loss_gamma']).to(CONFIG['device'])
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG['scheduler_T_0'],
        T_mult=CONFIG['scheduler_T_mult'],
        eta_min=CONFIG['scheduler_eta_min']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['early_stop_patience'],
        min_delta=CONFIG['early_stop_min_delta'],
        mode='max'
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }
    
    best_acc = 0
    best_epoch = 0
    start_time = datetime.now()
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch_single_gpu(
            model, train_loader, criterion, optimizer, 
            CONFIG['device'], epoch, CONFIG['grad_clip_norm']
        )
        
        # Validate
        val_metrics = validate_single_gpu(
            model, val_loader, criterion, CONFIG['device'], epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_epoch = epoch
            
            # Save best model (self-contained)
            best_model_dir = os.path.join(CONFIG['checkpoint_dir'], "best_model")
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(best_model_dir)
            
            # Save best metrics
            with open(os.path.join(best_model_dir, "metrics.json"), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'accuracy': val_metrics['accuracy'],
                    'f1': val_metrics['f1'],
                    'auc': val_metrics['auc'],
                }, f, indent=2)
            
            print(f"  ✓ New best model saved to {best_model_dir}")
            
            # Save confusion matrix
            cm = np.array(val_metrics['confusion_matrix'])
            plot_confusion_matrix(
                cm, epoch,
                f"{CONFIG['results_dir']}/confusion_matrix_best.png"
            )
        
        # Save latest checkpoint
        save_checkpoint_hf(
            model, optimizer, scheduler, epoch, val_metrics,
            CONFIG['checkpoint_dir']
        )
        
        # Early stopping check
        early_stopping(val_metrics['accuracy'])
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break
    
    # Training complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Duration: {duration}")
    print(f"Best Validation Accuracy: {best_acc:.4f} (epoch {best_epoch})")
    
    # Save training curves
    plot_training_curves(history, f"{CONFIG['results_dir']}/training_curves.png")
    
    # Save history
    with open(f"{CONFIG['results_dir']}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Results saved to {CONFIG['results_dir']}/")
    print(f"✓ Best model saved to {CONFIG['checkpoint_dir']}/best_model/")
    print(f"\nTo load the model for inference:")
    print(f"  from training_utils import RealVsAIClassifierHF")
    print(f"  model = RealVsAIClassifierHF.from_pretrained('{CONFIG['checkpoint_dir']}/best_model')")
    
    return model, history


if __name__ == "__main__":
    model, history = main()