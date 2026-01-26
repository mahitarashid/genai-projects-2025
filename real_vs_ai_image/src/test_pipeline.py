"""
Quick test script to verify the training pipeline works correctly.
Tests with a small subset of data (100 samples, 1 epoch).
"""

import sys
import torch
from train_classifier import (
    CONFIG, set_seed, RealVsAIClassifier, FocalLoss,
    get_transforms, train_epoch, validate
)
from image_data_loaders import CIFAKEDataset
from torch.utils.data import DataLoader

def test_pipeline():
    """Test the complete training pipeline with minimal data."""
    
    print("=" * 70)
    print("Testing Training Pipeline")
    print("=" * 70)
    
    # Set seed
    set_seed(CONFIG['seed'])
    
    # Test configuration (small subset)
    test_config = CONFIG.copy()
    test_config['max_train_samples'] = 100
    test_config['max_test_samples'] = 100
    test_config['batch_size'] = 8
    test_config['num_workers'] = 0  # Avoid multiprocessing issues in testing
    
    print("\nTest Configuration:")
    print(f"  Train samples: {test_config['max_train_samples']}")
    print(f"  Test samples: {test_config['max_test_samples']}")
    print(f"  Batch size: {test_config['batch_size']}")
    print(f"  Device: {test_config['device']}")
    
    # Load small dataset
    print("\n" + "-" * 70)
    print("Loading test dataset...")
    print("-" * 70)
    
    try:
        train_dataset = CIFAKEDataset(
            split="train",
            transform=get_transforms(is_train=True, target_size=test_config['image_size']),
            target_size=(test_config['image_size'], test_config['image_size']),
            max_samples=test_config['max_train_samples']
        )
        
        val_dataset = CIFAKEDataset(
            split="test",
            transform=get_transforms(is_train=False, target_size=test_config['image_size']),
            target_size=(test_config['image_size'], test_config['image_size']),
            max_samples=test_config['max_test_samples']
        )
        
        print("✓ Datasets loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        return False
    
    # Create dataloaders
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=test_config['batch_size'],
            shuffle=True,
            num_workers=test_config['num_workers'],
            pin_memory=True if test_config['device'] == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=test_config['batch_size'],
            shuffle=False,
            num_workers=test_config['num_workers'],
            pin_memory=True if test_config['device'] == 'cuda' else False
        )
        
        print(f"✓ DataLoaders created ({len(train_loader)} train batches, {len(val_loader)} val batches)")
        
    except Exception as e:
        print(f"✗ Error creating dataloaders: {e}")
        return False
    
    # Test data loading
    print("\n" + "-" * 70)
    print("Testing data loading...")
    print("-" * 70)
    
    try:
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        print(f"  Sample labels: {batch['label'][:5].tolist()}")
        
    except Exception as e:
        print(f"✗ Error loading batch: {e}")
        return False
    
    # Initialize model
    print("\n" + "-" * 70)
    print("Initializing model...")
    print("-" * 70)
    
    try:
        model = RealVsAIClassifier(
            num_classes=test_config['num_classes'],
            freeze_backbone=True,
            unfreeze_last_n_blocks=test_config['unfreeze_last_n_blocks']
        ).to(test_config['device'])
        
        print("✓ Model initialized successfully")
        
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return False
    
    # Setup training components
    print("\n" + "-" * 70)
    print("Setting up training components...")
    print("-" * 70)
    
    try:
        criterion = FocalLoss(gamma=test_config['focal_loss_gamma']).to(test_config['device'])
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=test_config['learning_rate'],
            weight_decay=test_config['weight_decay']
        )
        
        print(f"✓ Training components ready")
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
    except Exception as e:
        print(f"✗ Error setting up training: {e}")
        return False
    
    # Test forward pass
    print("\n" + "-" * 70)
    print("Testing forward pass...")
    print("-" * 70)
    
    try:
        model.eval()
        with torch.no_grad():
            images = batch['image'].to(test_config['device'])
            outputs = model(images)
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False
    
    # Test training step
    print("\n" + "-" * 70)
    print("Testing training epoch...")
    print("-" * 70)
    
    try:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, test_config['device'], epoch=1
        )
        
        print(f"✓ Training epoch completed")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        
    except Exception as e:
        print(f"✗ Error in training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test validation
    print("\n" + "-" * 70)
    print("Testing validation...")
    print("-" * 70)
    
    try:
        val_metrics = validate(
            model, val_loader, criterion, test_config['device'], epoch=1
        )
        
        print(f"✓ Validation completed")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Val AUC: {val_metrics['auc']:.4f}")
        
    except Exception as e:
        print(f"✗ Error in validation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All tests passed
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe pipeline is ready for full training.")
    print("Run: python train_classifier.py")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
