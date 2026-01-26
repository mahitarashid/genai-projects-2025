"""
Audio preprocessing utilities for the genre classifier.

Handles:
- Loading audio files (WAV format)
- Resampling to 16kHz (AST requirement)
- Segmenting into 10-second chunks
- Random cropping for data augmentation
- Device selection (cuda/mps/cpu)
"""

import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


def get_device(preferred: Optional[str] = None) -> str:
    """
    Get the best available device for PyTorch operations.
    
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
    
    Args:
        preferred: Preferred device (e.g., "cuda", "mps", "cpu").
                   If None, auto-detects the best available device.
                   
    Returns:
        Device string suitable for torch operations.
    """
    if preferred is not None:
        return preferred
    
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    # Fallback to CPU
    return "cpu"


# Constants for AST model
TARGET_SAMPLE_RATE = 16000  # AST expects 16kHz
SEGMENT_DURATION = 10  # seconds
SEGMENT_SAMPLES = TARGET_SAMPLE_RATE * SEGMENT_DURATION  # 160000 samples


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 16000 for AST)
        
    Returns:
        Tuple of (waveform, sample_rate)
        waveform shape: (1, num_samples) - mono audio
    """
    # Load audio
    waveform, sr = torchaudio.load(str(file_path))
    
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, 
            new_freq=target_sr
        )
        waveform = resampler(waveform)
    
    return waveform, target_sr


def random_crop(
    waveform: torch.Tensor,
    target_samples: int = SEGMENT_SAMPLES,
    pad_if_short: bool = True
) -> torch.Tensor:
    """
    Randomly crop a segment from the waveform.
    
    For training data augmentation - each epoch sees a different
    10-second segment from the 30-second audio file.
    
    Args:
        waveform: Audio waveform of shape (1, num_samples)
        target_samples: Number of samples to crop (default: 160000 for 10s)
        pad_if_short: Whether to pad if audio is shorter than target
        
    Returns:
        Cropped waveform of shape (1, target_samples)
    """
    num_samples = waveform.shape[1]
    
    if num_samples >= target_samples:
        # Random start position
        max_start = num_samples - target_samples
        start = random.randint(0, max_start)
        return waveform[:, start:start + target_samples]
    else:
        # Audio is shorter than target
        if pad_if_short:
            # Pad with zeros
            padding = target_samples - num_samples
            return torch.nn.functional.pad(waveform, (0, padding))
        else:
            return waveform


def center_crop(
    waveform: torch.Tensor,
    target_samples: int = SEGMENT_SAMPLES,
    pad_if_short: bool = True
) -> torch.Tensor:
    """
    Take a center crop from the waveform.
    
    For evaluation - deterministic cropping for reproducibility.
    
    Args:
        waveform: Audio waveform of shape (1, num_samples)
        target_samples: Number of samples to crop (default: 160000 for 10s)
        pad_if_short: Whether to pad if audio is shorter than target
        
    Returns:
        Cropped waveform of shape (1, target_samples)
    """
    num_samples = waveform.shape[1]
    
    if num_samples >= target_samples:
        # Center crop
        start = (num_samples - target_samples) // 2
        return waveform[:, start:start + target_samples]
    else:
        if pad_if_short:
            # Pad with zeros (center the audio)
            padding = target_samples - num_samples
            pad_left = padding // 2
            pad_right = padding - pad_left
            return torch.nn.functional.pad(waveform, (pad_left, pad_right))
        else:
            return waveform


def segment_audio(
    waveform: torch.Tensor,
    segment_samples: int = SEGMENT_SAMPLES,
    overlap: float = 0.0
) -> list:
    """
    Split audio into fixed-length segments.
    
    Useful for inference when we want to classify multiple segments
    and aggregate predictions.
    
    Args:
        waveform: Audio waveform of shape (1, num_samples)
        segment_samples: Number of samples per segment
        overlap: Overlap ratio between segments (0.0 to 1.0)
        
    Returns:
        List of segment tensors
    """
    num_samples = waveform.shape[1]
    step = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_samples <= num_samples:
        segment = waveform[:, start:start + segment_samples]
        segments.append(segment)
        start += step
    
    # Handle remaining samples
    if start < num_samples and num_samples - start > segment_samples // 2:
        # Pad the last segment if it's at least half the target length
        last_segment = waveform[:, start:]
        padding = segment_samples - last_segment.shape[1]
        last_segment = torch.nn.functional.pad(last_segment, (0, padding))
        segments.append(last_segment)
    
    return segments


def preprocess_audio(
    file_path: Union[str, Path],
    mode: str = "random_crop",
    target_sr: int = TARGET_SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for a single audio file.
    
    Args:
        file_path: Path to the audio file
        mode: Cropping mode - "random_crop", "center_crop", or "full"
        target_sr: Target sample rate
        segment_samples: Target segment length in samples
        
    Returns:
        Preprocessed waveform tensor
    """
    # Load and resample
    waveform, _ = load_audio(file_path, target_sr)
    
    # Crop based on mode
    if mode == "random_crop":
        waveform = random_crop(waveform, segment_samples)
    elif mode == "center_crop":
        waveform = center_crop(waveform, segment_samples)
    elif mode == "full":
        # Return full audio (for multi-segment inference)
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return waveform


def normalize_waveform(
    waveform: torch.Tensor,
    target_db: float = -20.0
) -> torch.Tensor:
    """
    Normalize audio to a target dB level.
    
    Args:
        waveform: Audio waveform
        target_db: Target dB level
        
    Returns:
        Normalized waveform
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(waveform ** 2))
    
    if rms > 0:
        # Calculate target RMS
        target_rms = 10 ** (target_db / 20)
        # Scale waveform
        waveform = waveform * (target_rms / rms)
    
    return waveform