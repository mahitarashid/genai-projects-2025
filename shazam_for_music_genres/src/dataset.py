"""
PyTorch Dataset for GTZAN music genre classification.

Uses ASTFeatureExtractor to convert audio waveforms to mel spectrograms
suitable for the Audio Spectrogram Transformer model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

from .preprocessing import (
    load_audio,
    random_crop,
    center_crop,
    SEGMENT_SAMPLES,
    TARGET_SAMPLE_RATE
)


GENRE_LABELS = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}

LABEL_TO_GENRE = {v: k for k, v in GENRE_LABELS.items()}
NUM_LABELS = len(GENRE_LABELS)


class GTZANDataset(Dataset):
    """
    PyTorch Dataset for GTZAN music genre classification.
    
    Loads audio files, preprocesses them, and converts to spectrograms
    using the ASTFeatureExtractor.
    
    Args:
        split_file: Path to JSONL file containing file paths and labels (one sample per line)
        feature_extractor: ASTFeatureExtractor instance
        mode: "train" for random cropping, "eval" for center cropping
        sample_rate: Target sample rate (default: 16000)
        segment_samples: Number of samples per segment (default: 160000)
    """
    
    def __init__(
        self,
        split_file: Union[str, Path],
        feature_extractor: Optional[ASTFeatureExtractor] = None,
        mode: str = "train",
        sample_rate: int = TARGET_SAMPLE_RATE,
        segment_samples: int = SEGMENT_SAMPLES
    ):
        self.split_file = Path(split_file)
        self.mode = mode
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        
        # Load split data from JSONL (one sample per line)
        self.file_paths = []
        self.labels = []
        with open(self.split_file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.file_paths.append(sample["file_path"])
                self.labels.append(sample["label"])
        
        # Initialize feature extractor
        if feature_extractor is None:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
        else:
            self.feature_extractor = feature_extractor
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - input_values: Mel spectrogram tensor
                - labels: Genre label
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        waveform, _ = load_audio(file_path, self.sample_rate)
        
        # Apply cropping based on mode
        if self.mode == "train":
            waveform = random_crop(waveform, self.segment_samples)
        else:
            waveform = center_crop(waveform, self.segment_samples)
        
        # Convert to numpy for feature extractor (expects 1D array)
        waveform_np = waveform.squeeze(0).numpy()
        
        # Extract features (mel spectrogram)
        features = self.feature_extractor(
            waveform_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by feature extractor
        input_values = features.input_values.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
    def get_label_name(self, label: int) -> str:
        """Convert numeric label to genre name."""
        return LABEL_TO_GENRE.get(label, "unknown")


class GTZANDataModule:
    """
    Data module for managing train/val/test datasets.
    
    Convenience class to create all dataloaders with consistent settings.
    
    Args:
        data_dir: Directory containing processed split files
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        feature_extractor: Shared ASTFeatureExtractor instance
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 8,
        num_workers: int = 4,
        feature_extractor: Optional[ASTFeatureExtractor] = None
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Shared feature extractor
        if feature_extractor is None:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
        else:
            self.feature_extractor = feature_extractor
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    @property
    def train_dataset(self) -> GTZANDataset:
        if self._train_dataset is None:
            self._train_dataset = GTZANDataset(
                split_file=self.data_dir / "train_split.jsonl",
                feature_extractor=self.feature_extractor,
                mode="train"
            )
        return self._train_dataset
    
    @property
    def val_dataset(self) -> GTZANDataset:
        if self._val_dataset is None:
            self._val_dataset = GTZANDataset(
                split_file=self.data_dir / "val_split.jsonl",
                feature_extractor=self.feature_extractor,
                mode="eval"
            )
        return self._val_dataset
    
    @property
    def test_dataset(self) -> GTZANDataset:
        if self._test_dataset is None:
            self._test_dataset = GTZANDataset(
                split_file=self.data_dir / "test_split.jsonl",
                feature_extractor=self.feature_extractor,
                mode="eval"
            )
        return self._test_dataset
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Stacks individual samples into batched tensors.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with stacked tensors
    """
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_values": input_values,
        "labels": labels
    }