# Real vs AI Image Classifier

Fine-tuned DINOv2-Large classifier for detecting AI-generated images.

## Overview

This project trains a Vision Transformer (DINOv2-Large) to differentiate between real photos and AI-generated images, with special consideration for smartphone computational photography.

**Key Features:**
- DINOv2-Large backbone with selective unfreezing
- Smartphone simulation augmentation
- Focal Loss for hard example focus
- Comprehensive metrics and visualization
- Expected 92-95% accuracy on CIFAKE dataset

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (Recommended)

**Option A: Use Pre-prepared Splits (Recommended for reproducibility)**

```bash
cd real_vs_ai_image
python prepare_cifake_splits.py
```

This will:
- Download full CIFAKE dataset from HuggingFace
- Create 10 random shuffled splits for train (10k samples each)
- Create 10 random shuffled splits for test (10k samples each)
- Save splits as pickle files in `data/cifake_splits/`
- Takes ~30-60 minutes on first run

**Option B: Download on-the-fly (Legacy method)**

Skip this step and the dataset will be downloaded during training.

### 3. Run Training

```bash
python train_classifier_hf.py
```

The script will:
- Load pre-prepared splits (if available) or download from HuggingFace
- Train for up to 15 epochs with early stopping
- Save checkpoints to `checkpoints/`
- Save results and visualizations to `results_hf/` (or `results/` if using PyTorch version)

### 3. Monitor Training

Training progress is displayed with tqdm progress bars showing:
- Loss and accuracy per batch
- Validation metrics after each epoch
- Best model checkpoints

## Dataset

**CIFAKE** (120K samples total):
- Full dataset: 60K train + 60K test samples
- Each split: 10K samples (5K real CIFAR-10, 5K Stable Diffusion)
- 10 pre-prepared random splits for reproducible experiments

### Dataset Loading Methods

**Method 1: Pre-prepared Splits (Recommended)**
```python
CONFIG = {
    'use_pickled_splits': True,
    'max_split': 3,  # Use splits 1, 2, 3 (30k samples)
    'splits_data_dir': 'data/cifake_splits',
}
```

Benefits:
- ✓ Reproducible experiments (same splits across runs)
- ✓ Faster loading (no HuggingFace download)
- ✓ Easy data scaling (test with 1, 2, 3... splits)
- ✓ Consistent train/test distribution

**Method 2: HuggingFace Direct (Legacy)**
```python
CONFIG = {
    'use_pickled_splits': False,
    'max_train_samples': 10000,
    'max_test_samples': 10000,
}
```

## Model Architecture

```
DINOv2-Large (ViT-L/14)
├── Frozen: First 20 transformer blocks
├── Unfrozen: Last 4 transformer blocks (high-level features)
└── Classification Head:
    ├── Linear(1024 → 512)
    ├── BatchNorm + GELU + Dropout(0.3)
    ├── Linear(512 → 256)
    ├── GELU
    └── Linear(256 → 2)
```

**Trainable Parameters:** ~50M / 304M (16%)

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 518×518 | Optimal for DINOv2 ViT-L/14 |
| Batch Size | 16 | Adjust based on GPU memory |
| Learning Rate | 1e-4 | Conservative for fine-tuning |
| Optimizer | AdamW | With weight decay 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts | T_0=5, T_mult=2 |
| Loss | Focal Loss | γ=2.0 for hard examples |
| Epochs | 15 | With early stopping (patience=5) |
| Training Time | 2-3 hours | On single GPU |

## Augmentation Pipeline

**Smartphone Simulation** (40% probability):
- Aggressive sharpening (1.5-3.0x)
- Gaussian blur smoothing
- Saturation boost (1.2-1.5x)

**Standard Augmentations**:
- Random horizontal flip
- Random rotation (±10°)
- ImageNet normalization

## Results

After training, you'll find:

```
results_hf/
├── training_curves.png       # Loss, accuracy, F1, AUC curves
├── confusion_matrix_best.png # Best epoch confusion matrix
└── training_history.json     # Complete metrics history

checkpoints_hf/       
├── best_model/               # Model directory with the best validation accuracy
└── model_epoch_x             # Saved epochs

checkpoints/          # for PyTorch
├── best_model.pth            # Best validation accuracy
└── latest_checkpoint.pth     # Latest epoch (for resuming)
```

## Expected Performance

- **Accuracy**: 92-95%
- **Precision**: >93%
- **Recall**: >93%
- **F1 Score**: >93%
- **AUC**: >0.95

## File Structure

```
real_vs_ai_image/
├── train_classifier.py        # Main training script
├── prepare_cifake_splits.py   # Dataset preparation script (NEW)
├── image_data_loaders.py      # Dataset utilities (includes PickledCIFAKEDataset)
├── training_utils.py          # Shared training utilities
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── data/                      # Dataset storage (created by prepare_cifake_splits.py)
│   └── cifake_splits/
│       ├── train_split_1.pkl through train_split_10.pkl
│       ├── test_split_1.pkl through test_split_10.pkl
│       └── metadata.json
├── notebooks/
│   ├── dino-tune.ipynb       # Original notebook
│   └── image_eda.ipynb       # Exploratory analysis
├── checkpoints/              # Model checkpoints (created during training)
├── results/                  # Training results (created during training)
```

## Usage Examples

### Prepare Dataset Splits (One-time)

```bash
python prepare_cifake_splits.py
```

This creates 10 random splits (10k samples each) for both train and test.

### Basic Training with Pre-prepared Splits

```bash
python train_classifier.py
```

Default configuration uses `max_split=3` (30k samples).

### Experiment with Different Data Sizes

Edit the `CONFIG` dictionary in [`train_classifier_hf.py`](train_classifier_hf.py):

```python
# Use 1 split (10k samples) - quick testing
CONFIG = {
    'use_pickled_splits': True,
    'max_split': 1,
}

# Use 5 splits (50k samples) - medium scale
CONFIG = {
    'use_pickled_splits': True,
    'max_split': 5,
}

# Use all 10 splits (100k samples) - full scale
CONFIG = {
    'use_pickled_splits': True,
    'max_split': 10,
}
```

### Legacy Method (HuggingFace Direct)

```python
CONFIG = {
    'use_pickled_splits': False,
    'max_train_samples': 5000,  # Use subset for quick testing
    'max_test_samples': 5000,
}
```

### Other Configuration Options

```python
CONFIG = {
    'batch_size': 8,           # Reduce if GPU memory limited
    'epochs': 10,              # Fewer epochs
    'learning_rate': 5e-5,     # Lower learning rate
    # ... other parameters
}
```

### Load Trained Model

```python
import torch
from train_classifier import RealVsAIClassifier

# Load model
model = RealVsAIClassifier(num_classes=2, unfreeze_last_n_blocks=4)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1)
```



## Next Steps

1. **Expand Dataset**: Add ForenSynths for diverse AI generators
2. **Test Generalization**: Evaluate on DALL-E 3, Midjourney v6
3. **Smartphone Photos**: Collect real smartphone photos for validation


## References

- **DINOv2**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **CIFAKE**: [HuggingFace Dataset](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
