"""
Shared training utilities for Real vs AI Image Classifier.
Used by both single-GPU and multi-GPU training scripts.

This module provides two model implementations:
1. RealVsAIClassifier - Uses torch.hub DINOv2 (original)
2. RealVsAIClassifierHF - Uses HuggingFace transformers DINOv2 (recommended for sharing)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)

# HuggingFace transformers imports
try:
    from transformers import Dinov2Model, Dinov2Config, PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers library not available. RealVsAIClassifierHF will not work.")


# ============================================================================
# Device Selection
# ============================================================================

def get_device():
    """
    Get the best available device for training/inference.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


# ============================================================================
# Random Seed
# ============================================================================

def set_seed(seed, rank=0):
    """Set random seeds for reproducibility."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Model Definition
# ============================================================================

class RealVsAIClassifier(nn.Module):
    """
    DINOv2-Large based classifier for real vs AI-generated image detection.
    
    Features:
    - Selective unfreezing of last N transformer blocks
    - Robust classification head with BatchNorm and Dropout
    - Designed for fine-tuning on detection tasks
    
    Note: The backbone is downloaded from torch hub on first use and cached
    in ~/.cache/torch/hub/ for subsequent runs.
    """
    
    def __init__(self, num_classes=2, freeze_backbone=True, unfreeze_last_n_blocks=4):
        super().__init__()
        
        print("Loading DINOv2-Large backbone...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        embed_dim = 1024
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            for param in self.head.parameters():
                param.requires_grad = True
                
            for block in self.backbone.blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
                    
        print(f"Model initialized. Last {unfreeze_last_n_blocks} blocks unfrozen.")
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


# ============================================================================
# HuggingFace Transformers Model (Recommended for sharing)
# ============================================================================

class RealVsAIClassifierHF(nn.Module):
    """
    DINOv2-Large based classifier using HuggingFace transformers.
    
    This version uses the transformers library instead of torch.hub, which allows:
    - Self-contained model saves with save_pretrained()
    - Easy loading with from_pretrained() without external dependencies
    - Works offline after first download
    - Easy sharing via HuggingFace Hub
    
    Features:
    - Selective unfreezing of last N transformer blocks
    - Robust classification head with BatchNorm and Dropout
    - Compatible with HuggingFace save/load methods
    
    Usage:
        # Create and train
        model = RealVsAIClassifierHF(num_classes=2)
        # ... train ...
        
        # Save (self-contained)
        model.save_pretrained('my_model/')
        
        # Load (no external downloads needed)
        model = RealVsAIClassifierHF.from_pretrained('my_model/')
    """
    
    # HuggingFace model name for DINOv2-Large
    DINOV2_MODEL_NAME = "facebook/dinov2-large"
    
    def __init__(
        self,
        num_classes=2,
        freeze_backbone=True,
        unfreeze_last_n_blocks=4,
        pretrained_backbone=True
    ):
        """
        Args:
            num_classes: Number of output classes
            freeze_backbone: Whether to freeze backbone weights
            unfreeze_last_n_blocks: Number of transformer blocks to unfreeze
            pretrained_backbone: Whether to load pretrained weights (set False when loading from checkpoint)
        """
        super().__init__()
        
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for RealVsAIClassifierHF. "
                "Install with: pip install transformers"
            )
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        
        # Load DINOv2 backbone from HuggingFace
        if pretrained_backbone:
            print(f"Loading DINOv2-Large backbone from HuggingFace ({self.DINOV2_MODEL_NAME})...")
            self.backbone = Dinov2Model.from_pretrained(self.DINOV2_MODEL_NAME)
        else:
            print("Creating DINOv2-Large backbone without pretrained weights...")
            config = Dinov2Config.from_pretrained(self.DINOV2_MODEL_NAME)
            self.backbone = Dinov2Model(config)
        
        # Get embedding dimension from config
        embed_dim = self.backbone.config.hidden_size  # 1024 for dinov2-large
        
        # Classification head (same as original)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        # Apply freezing
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n_blocks)
        
        self._print_trainable_params()
    
    def _freeze_backbone(self, unfreeze_last_n_blocks):
        """Freeze backbone except last N blocks."""
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze head
        for param in self.head.parameters():
            param.requires_grad = True
        
        # Unfreeze last N encoder layers
        if unfreeze_last_n_blocks > 0:
            num_layers = len(self.backbone.encoder.layer)
            for layer in self.backbone.encoder.layer[-unfreeze_last_n_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {unfreeze_last_n_blocks} of {num_layers} encoder layers.")
    
    def _print_trainable_params(self):
        """Print trainable parameter count."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def forward(self, x, labels=None):
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            labels: Optional labels for loss computation
        
        Returns:
            logits if labels is None, else (loss, logits)
        """
        # Get backbone features
        outputs = self.backbone(x)
        
        # Use CLS token (first token) as image representation
        features = outputs.last_hidden_state[:, 0]  # (B, embed_dim)
        
        # Classification
        logits = self.head(features)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        
        return logits
    
    def save_pretrained(self, save_directory):
        """
        Save model to directory (self-contained, no external dependencies).
        
        Args:
            save_directory: Directory to save model files
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save backbone
        backbone_dir = os.path.join(save_directory, "backbone")
        self.backbone.save_pretrained(backbone_dir)
        
        # Save head and config
        head_path = os.path.join(save_directory, "head.pth")
        config_path = os.path.join(save_directory, "classifier_config.pth")
        
        torch.save(self.head.state_dict(), head_path)
        torch.save({
            'num_classes': self.num_classes,
            'freeze_backbone': self.freeze_backbone,
            'unfreeze_last_n_blocks': self.unfreeze_last_n_blocks,
        }, config_path)
        
        print(f"✓ Model saved to {save_directory}")
        print(f"  - Backbone: {backbone_dir}")
        print(f"  - Head: {head_path}")
        print(f"  - Config: {config_path}")
    
    @classmethod
    def from_pretrained(cls, load_directory, device='cpu'):
        """
        Load model from directory (no external downloads needed).
        
        Args:
            load_directory: Directory containing saved model files
            device: Device to load model on
        
        Returns:
            Loaded model in eval mode
        """
        print(f"Loading model from {load_directory}...")
        
        # Load config
        config_path = os.path.join(load_directory, "classifier_config.pth")
        config = torch.load(config_path, map_location=device)
        
        # Create model without loading pretrained backbone
        model = cls(
            num_classes=config['num_classes'],
            freeze_backbone=config['freeze_backbone'],
            unfreeze_last_n_blocks=config['unfreeze_last_n_blocks'],
            pretrained_backbone=False  # Don't download, we'll load from disk
        )
        
        # Load backbone from saved directory
        backbone_dir = os.path.join(load_directory, "backbone")
        model.backbone = Dinov2Model.from_pretrained(backbone_dir)
        
        # Load head
        head_path = os.path.join(load_directory, "head.pth")
        model.head.load_state_dict(torch.load(head_path, map_location=device))
        
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully!")
        return model


# ============================================================================
# Data Augmentation
# ============================================================================

class SmartphoneSimulationAugment:
    """
    Custom transform to simulate smartphone computational photography artifacts.
    Helps model learn invariance to smartphone AI processing.
    """
    
    def __call__(self, img):
        if random.random() < 0.4:
            img = transforms.functional.adjust_sharpness(
                img, sharpness_factor=random.uniform(1.5, 3.0)
            )
        
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
        if random.random() < 0.4:
            img = transforms.functional.adjust_saturation(
                img, saturation_factor=random.uniform(1.2, 1.5)
            )
        
        return img


def get_transforms(is_train=True, target_size=518):
    """Get unified transform pipeline."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            SmartphoneSimulationAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


# ============================================================================
# Loss Function
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath, rank=0):
    """Save model checkpoint (only on main process in distributed training)."""
    if rank != 0:
        return
    
    # Unwrap DDP if needed
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_model_for_inference(checkpoint_path, device='cpu', num_classes=2, unfreeze_last_n_blocks=4):
    """
    Load a trained model for inference.
    
    This creates the model architecture (downloading backbone from torch hub if needed,
    but it's cached after first download), then loads the trained weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
        num_classes: Number of output classes (default: 2)
        unfreeze_last_n_blocks: Number of blocks that were unfrozen during training
    
    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dict with metadata
    
    Example:
        model, checkpoint = load_model_for_inference('checkpoints/best_model.pth', device='cuda')
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    Note:
        The DINOv2 backbone is downloaded from torch hub on first use and cached
        in ~/.cache/torch/hub/ for subsequent runs.
    """
    print(f"Loading model for inference from: {checkpoint_path}")
    
    # Create model (downloads backbone from torch hub if not cached)
    print("Creating model architecture...")
    model = RealVsAIClassifier(
        num_classes=num_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks
    )
    
    # Load trained weights
    print("Loading trained weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if isinstance(metrics, dict):
            print(f"  Validation Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    
    return model, checkpoint


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint and optionally restore training state."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint['metrics']}")
    
    return checkpoint['epoch'], checkpoint['metrics']




# ============================================================================
# Visualization
# ============================================================================

def plot_confusion_matrix(cm, epoch, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'AI'],
                yticklabels=['Real', 'AI'])
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['val_auc'], label='Val AUC')
    axes[1, 1].set_title('ROC-AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Training Functions (Single GPU)
# ============================================================================

def train_epoch_single_gpu(model, train_loader, criterion, optimizer, device, epoch, grad_clip_norm=1.0):
    """Train for one epoch (single GPU version)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate_single_gpu(model, val_loader, criterion, device, epoch):
    """Validate and compute comprehensive metrics (single GPU version)."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics