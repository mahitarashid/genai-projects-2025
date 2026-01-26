"""
Data loader for Real vs AI-Generated Image Detection
Structured to easily swap between CIFAKE, ForenSynths, and custom datasets.
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any, List
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageSource(Enum):
    """Track the origin of each image for analysis."""
    REAL = "real"
    STABLE_DIFFUSION = "stable_diffusion"
    PROGAN = "progan"
    STYLEGAN = "stylegan"
    BIGGAN = "biggan"
    CYCLEGAN = "cyclegan"
    DALLE = "dalle"
    MIDJOURNEY = "midjourney"
    UNKNOWN_AI = "unknown_ai"


# =============================================================================
# Base Dataset Class
# =============================================================================

class BaseDetectionDataset(Dataset):
    """
    Base class for real vs AI detection datasets.
    Subclass this for specific dataset formats.
    """
    
    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.transform = transform or self._default_transform(target_size)
        self.target_size = target_size
        self.samples: list[Dict[str, Any]] = []  # List of {path, label, source}
    
    def _default_transform(self, size: Tuple[int, int]) -> Callable:
        """Standard preprocessing for ViT."""
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": sample["label"],  # 0 = real, 1 = AI
            "source": sample["source"].value,
            "path": str(sample["path"]),
        }


# =============================================================================
# CIFAKE Dataset
# =============================================================================

class CIFAKEDataset(BaseDetectionDataset):
    """
    CIFAKE dataset loader.
    Uses HuggingFace datasets library.
    
    Labels: 0 = real (CIFAR-10), 1 = AI (Stable Diffusion)
    """
    
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        max_samples: Optional[int] = None,
    ):
        super().__init__(transform, target_size)
        self.split = split
        self._load_data(max_samples)
    
    def _load_data(self, max_samples: Optional[int] = None):
        """Load CIFAKE from HuggingFace."""
        from datasets import load_dataset
        
        print(f"Loading CIFAKE {self.split} split...")
        dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split=self.split)
        print(f"[{self.split}] got {len(dataset):,d} samples")

        if max_samples:
            # Shuffle before selection to get random samples instead of first N
            dataset = dataset.shuffle(seed=42)
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        if max_samples:
            # Shuffle before selection to get random samples instead of first N
            dataset = dataset.shuffle(seed=42)
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Convert to our format
        # CIFAKE: label 0 = real, label 1 = fake
        for i, item in enumerate(dataset):
            self.samples.append({
                "image_data": item["image"],  # PIL Image directly
                "label": item["label"],
                "source": ImageSource.REAL if item["label"] == 0 else ImageSource.STABLE_DIFFUSION,
            })
        
        print(f"Loaded {len(self.samples)} samples")
        self._print_distribution()
    
    def _print_distribution(self):
        """Print class distribution."""
        real = sum(1 for s in self.samples if s["label"] == 0)
        fake = len(self.samples) - real
        print(f"  Real: {real}, AI: {fake}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = sample["image_data"].convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": sample["label"],
            "source": sample["source"].value,
        }


# =============================================================================
# Pickled CIFAKE Dataset (Pre-prepared Splits)
# =============================================================================

class PickledCIFAKEDataset(BaseDetectionDataset):
    """
    Load pre-prepared CIFAKE dataset splits from pickle files.
    
    This dataset loads from pickle files created by prepare_cifake_splits.py.
    Supports loading multiple splits at once (e.g., splits 1, 2, 3).
    
    Usage:
        # Load single split
        dataset = PickledCIFAKEDataset(split="train", split_numbers=[1])
        
        # Load multiple splits (e.g., 30k samples from splits 1, 2, 3)
        dataset = PickledCIFAKEDataset(split="train", split_numbers=[1, 2, 3])
    """
    
    def __init__(
        self,
        split: str = "train",
        split_numbers: List[int] = [1],
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        data_dir: str = "data/cifake_splits",
    ):
        """
        Args:
            split: "train" or "test"
            split_numbers: List of split numbers to load (e.g., [1, 2, 3])
            transform: Optional transform to apply to images
            target_size: Target image size for default transform
            data_dir: Directory containing pickle files
        """
        super().__init__(transform, target_size)
        self.split = split
        self.split_numbers = split_numbers
        self.data_dir = Path(data_dir)
        self._load_splits()
    
    def _load_splits(self):
        """Load specified pickle files and combine samples."""
        import pickle
        
        print(f"Loading {self.split} splits: {self.split_numbers}")
        
        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please run prepare_cifake_splits.py first to create the splits."
            )
        
        # Load each split
        for split_num in self.split_numbers:
            filepath = self.data_dir / f"{self.split}_split_{split_num}.pkl"
            
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Split file not found: {filepath}\n"
                    f"Available splits: {sorted([f.stem for f in self.data_dir.glob(f'{self.split}_split_*.pkl')])}"
                )
            
            print(f"  Loading {filepath.name}...", end=" ")
            
            with open(filepath, 'rb') as f:
                split_data = pickle.load(f)
            
            # Add samples to our list
            self.samples.extend(split_data)
            
            print(f"✓ ({len(split_data)} samples)")
        
        print(f"Total loaded: {len(self.samples)} samples")
        self._print_distribution()
    
    def _print_distribution(self):
        """Print class distribution."""
        real = sum(1 for s in self.samples if s["label"] == 0)
        fake = len(self.samples) - real
        print(f"  Real: {real}, AI: {fake}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        sample = self.samples[idx]
        
        # Convert numpy array to PIL Image
        # The pickle files store images as numpy arrays for better compatibility
        import numpy as np
        if isinstance(sample["image"], np.ndarray):
            image = Image.fromarray(sample["image"])
        else:
            # Fallback for old pickle format with PIL Images
            image = sample["image"]
        
        image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": sample["label"],
            "source": sample["source"],
        }


# =============================================================================
# ForenSynths Dataset
# =============================================================================

class ForenSynthsDataset(BaseDetectionDataset):
    """
    ForenSynths dataset loader.
    Expects the CNNDetection dataset structure.
    
    Directory structure:
    root/
      train/
        real/
        progan/
        stylegan/
        ...
      val/
        ...
    """
    
    SOURCE_MAPPING = {
        "real": ImageSource.REAL,
        "progan": ImageSource.PROGAN,
        "stylegan": ImageSource.STYLEGAN,
        "stylegan2": ImageSource.STYLEGAN,
        "biggan": ImageSource.BIGGAN,
        "cyclegan": ImageSource.CYCLEGAN,
        "stargan": ImageSource.UNKNOWN_AI,
        "gaugan": ImageSource.UNKNOWN_AI,
        "deepfake": ImageSource.UNKNOWN_AI,
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        max_samples_per_source: Optional[int] = None,
        include_sources: Optional[list[str]] = None,
        exclude_sources: Optional[list[str]] = None,
    ):
        """
        Args:
            root_dir: Path to ForenSynths root directory
            split: "train" or "val"
            max_samples_per_source: Limit samples per generator (for balancing)
            include_sources: Only include these sources (e.g., ["real", "progan"])
            exclude_sources: Exclude these sources (for holdout testing)
        """
        super().__init__(transform, target_size)
        self.root_dir = Path(root_dir)
        self.split = split
        self._load_data(max_samples_per_source, include_sources, exclude_sources)
    
    def _load_data(
        self,
        max_per_source: Optional[int],
        include: Optional[list[str]],
        exclude: Optional[list[str]],
    ):
        """Scan directory and load image paths."""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        print(f"Loading ForenSynths {self.split} split from {split_dir}...")
        
        for source_dir in split_dir.iterdir():
            if not source_dir.is_dir():
                continue
            
            source_name = source_dir.name.lower()
            
            # Filter sources
            if include and source_name not in include:
                continue
            if exclude and source_name in exclude:
                print(f"  Excluding {source_name} (holdout)")
                continue
            
            # Determine source type and label
            source_type = self.SOURCE_MAPPING.get(source_name, ImageSource.UNKNOWN_AI)
            label = 0 if source_type == ImageSource.REAL else 1
            
            # Collect image paths
            image_paths = list(source_dir.glob("*.png")) + \
                         list(source_dir.glob("*.jpg")) + \
                         list(source_dir.glob("*.jpeg"))
            
            if max_per_source:
                random.shuffle(image_paths)
                image_paths = image_paths[:max_per_source]
            
            for path in image_paths:
                self.samples.append({
                    "path": path,
                    "label": label,
                    "source": source_type,
                })
            
            print(f"  {source_name}: {len(image_paths)} images")
        
        print(f"Total: {len(self.samples)} samples")
        self._print_distribution()
    
    def _print_distribution(self):
        """Print detailed class distribution."""
        from collections import Counter
        sources = Counter(s["source"].value for s in self.samples)
        real = sum(1 for s in self.samples if s["label"] == 0)
        fake = len(self.samples) - real
        print(f"  Real: {real}, AI: {fake}")
        print(f"  By source: {dict(sources)}")


# =============================================================================
# Custom Mixed Dataset
# =============================================================================

class CustomMixedDataset(BaseDetectionDataset):
    """
    Load images from a custom directory structure.
    
    Expected structure:
    root/
      real/
        source_name/  (e.g., "laion", "raise", "coco")
          *.jpg
      ai/
        generator_name/  (e.g., "stable_diffusion", "midjourney")
          *.jpg
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        max_samples_per_source: Optional[int] = None,
    ):
        super().__init__(transform, target_size)
        self.root_dir = Path(root_dir)
        self._load_data(max_samples_per_source)
    
    def _load_data(self, max_per_source: Optional[int]):
        """Scan custom directory structure."""
        print(f"Loading custom dataset from {self.root_dir}...")
        
        for category in ["real", "ai"]:
            category_dir = self.root_dir / category
            if not category_dir.exists():
                continue
            
            label = 0 if category == "real" else 1
            
            for source_dir in category_dir.iterdir():
                if not source_dir.is_dir():
                    continue
                
                source_name = source_dir.name.lower()
                
                # Map to ImageSource enum
                if category == "real":
                    source_type = ImageSource.REAL
                else:
                    source_type = {
                        "stable_diffusion": ImageSource.STABLE_DIFFUSION,
                        "sd": ImageSource.STABLE_DIFFUSION,
                        "midjourney": ImageSource.MIDJOURNEY,
                        "mj": ImageSource.MIDJOURNEY,
                        "dalle": ImageSource.DALLE,
                    }.get(source_name, ImageSource.UNKNOWN_AI)
                
                image_paths = list(source_dir.glob("**/*.png")) + \
                             list(source_dir.glob("**/*.jpg")) + \
                             list(source_dir.glob("**/*.jpeg"))
                
                if max_per_source:
                    random.shuffle(image_paths)
                    image_paths = image_paths[:max_per_source]
                
                for path in image_paths:
                    self.samples.append({
                        "path": path,
                        "label": label,
                        "source": source_type,
                    })
                
                print(f"  {category}/{source_name}: {len(image_paths)} images")
        
        print(f"Total: {len(self.samples)} samples")


# =============================================================================
# Data Augmentation
# =============================================================================

def get_training_transforms(
    target_size: Tuple[int, int] = (224, 224),
    jpeg_compression: bool = True,
) -> Callable:
    """
    Training augmentations.
    Includes JPEG compression simulation (important for forensics).
    """
    transform_list = [
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]
    
    if jpeg_compression:
        # Simulate JPEG compression artifacts
        transform_list.append(JPEGCompressionTransform(quality_range=(70, 100)))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transforms.Compose(transform_list)


def get_eval_transforms(target_size: Tuple[int, int] = (224, 224)) -> Callable:
    """Standard evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class JPEGCompressionTransform:
    """Simulate JPEG compression artifacts."""
    
    def __init__(self, quality_range: Tuple[int, int] = (70, 100)):
        self.quality_range = quality_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        import io
        import random
        
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloaders(
    dataset_name: str = "cifake",
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224),
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create train/val dataloaders.
    
    Args:
        dataset_name: "cifake", "forensynths", or "custom"
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        target_size: Image size (H, W)
        **dataset_kwargs: Additional args passed to dataset class
    
    Returns:
        (train_loader, val_loader)
    """
    train_transform = get_training_transforms(target_size)
    eval_transform = get_eval_transforms(target_size)
    
    if dataset_name.lower() == "cifake":
        train_dataset = CIFAKEDataset(
            split="train",
            transform=train_transform,
            target_size=target_size,
            **dataset_kwargs,
        )
        val_dataset = CIFAKEDataset(
            split="test",
            transform=eval_transform,
            target_size=target_size,
            **dataset_kwargs,
        )
    
    elif dataset_name.lower() == "forensynths":
        root_dir = dataset_kwargs.pop("root_dir")
        train_dataset = ForenSynthsDataset(
            root_dir=root_dir,
            split="train",
            transform=train_transform,
            target_size=target_size,
            **dataset_kwargs,
        )
        val_dataset = ForenSynthsDataset(
            root_dir=root_dir,
            split="val",
            transform=eval_transform,
            target_size=target_size,
            **dataset_kwargs,
        )
    
    elif dataset_name.lower() == "custom":
        root_dir = dataset_kwargs.pop("root_dir")
        # For custom, you'd typically split manually
        full_dataset = CustomMixedDataset(
            root_dir=root_dir,
            transform=train_transform,
            target_size=target_size,
            **dataset_kwargs,
        )
        # 80/20 split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Test CIFAKE loading
    print("=" * 60)
    print("Testing CIFAKE Dataset")
    print("=" * 60)
    
    train_loader, val_loader = create_dataloaders(
        dataset_name="cifake",
        batch_size=32,
        num_workers=0,  # Set to 0 for testing
        max_samples=10000,  # Small subset for testing
    )
    
    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch info:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Labels: {batch['label'][:10]}")
    print(f"  Sources: {batch['source'][:5]}")
    
    print("\n✓ Data loader working correctly!")