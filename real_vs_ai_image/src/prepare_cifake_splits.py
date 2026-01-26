"""
CIFAKE Dataset Splits Preparation Script

- downloads the full CIFAKE dataset from HuggingFace
- creates 10 random shuffled splits for both train and test sets
    - each split is saved as a pickle file for fast loading during training

Usage:
    python prepare_cifake_splits.py

Output:
    - data/cifake_splits/train_split_1.pkl through train_split_10.pkl
    - data/cifake_splits/test_split_1.pkl through test_split_10.pkl
    - data/cifake_splits/metadata.json
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from datasets import load_dataset


CONFIG = {
    'num_splits': 10,
    'train_base_seed': 42,
    'test_base_seed': 142,
    'output_dir': 'data/cifake_splits',
    'dataset_name': 'dragonintelligence/CIFAKE-image-dataset',
}

def download_cifake_dataset(split: str):
    """
    Download CIFAKE dataset from HuggingFace.
    
    Args:
        split: "train" or "test"
    
    Returns:
        HuggingFace Dataset object
    """
    print(f"\n{'='*70}")
    print(f"Downloading CIFAKE {split} split from HuggingFace")
    print(f"{'='*70}")
    
    dataset = load_dataset(CONFIG['dataset_name'], split=split)
    print(f"Downloaded {len(dataset):,} samples")
    
    return dataset


def create_random_splits(
    dataset,
    num_splits: int,
    base_seed: int,
    split_name: str
) -> List[List[Dict[str, Any]]]:
    """
    Create multiple disjoint random splits from a dataset.
    
    The dataset is shuffled once, then partitioned into num_splits disjoint subsets.
    This ensures no overlap between splits.
    
    Args:
        dataset: HuggingFace Dataset object
        num_splits: Number of splits to create
        base_seed: Random seed for shuffling
        split_name: "train" or "test" for logging
    
    Returns:
        List of splits, where each split is a list of sample dictionaries
    """
    # Calculate samples per split from total dataset size
    total_samples = len(dataset)
    samples_per_split = total_samples // num_splits
    
    print(f"\n{'='*70}")
    print(f"Creating {num_splits} disjoint random splits for {split_name}")
    print(f"Total samples: {total_samples:,}, Samples per split: {samples_per_split:,}")
    print(f"{'='*70}")
    
    # Shuffle dataset once with base seed
    print(f"\nShuffling dataset with seed={base_seed}...")
    shuffled = dataset.shuffle(seed=base_seed)
    
    # Create disjoint splits by partitioning the shuffled dataset
    splits = []
    
    for i in range(num_splits):
        split_num = i + 1
        
        # Calculate start and end indices for this split
        start_idx = i * samples_per_split
        end_idx = start_idx + samples_per_split
        
        print(f"\nCreating split {split_num}/{num_splits} (indices {start_idx:,} to {end_idx:,})...")
        
        # Select disjoint subset
        selected = shuffled.select(range(start_idx, end_idx))
        
        # Convert to our format (numpy array + label for better pickling)
        split_data = []
        for item in tqdm(selected, desc=f"Processing {split_name} split {split_num}"):
            # Convert PIL Image to numpy array for reliable pickling
            image_array = np.array(item["image"])
            split_data.append({
                "image": image_array,  # numpy array (H, W, C)
                "label": item["label"],  # 0 = real, 1 = AI
                "source": "real" if item["label"] == 0 else "stable_diffusion",
            })
        
        splits.append(split_data)
        
        # Print distribution
        real_count = sum(1 for s in split_data if s["label"] == 0)
        ai_count = len(split_data) - real_count
        print(f"  Split {split_num}: {len(split_data)} samples (Real: {real_count}, AI: {ai_count})")
    
    return splits


def save_split_to_pickle(split_data: List[Dict], filepath: Path):
    """
    Save split data to pickle file.
    
    Args:
        split_data: List of sample dictionaries
        filepath: Path to save pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Get file size
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Saved to {filepath.name} ({size_mb:.1f} MB)")


def create_metadata(
    train_splits_info: List[Dict],
    test_splits_info: List[Dict]
) -> Dict:
    """
    Create metadata JSON with split information.
    
    Args:
        train_splits_info: List of train split metadata
        test_splits_info: List of test split metadata
    
    Returns:
        Metadata dictionary
    """
    metadata = {
        "creation_date": datetime.utcnow().isoformat() + "Z",
        "dataset": CONFIG['dataset_name'],
        "num_splits": CONFIG['num_splits'],
        "train_splits": {},
        "test_splits": {},
    }
    
    for i, info in enumerate(train_splits_info):
        split_num = i + 1
        metadata["train_splits"][str(split_num)] = info
    
    for i, info in enumerate(test_splits_info):
        split_num = i + 1
        metadata["test_splits"][str(split_num)] = info
    
    return metadata


def main():
    print("=" * 70)
    print("CIFAKE Dataset Splits Preparation")
    print("=" * 70)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Download datasets
    train_dataset = download_cifake_dataset("train")
    test_dataset = download_cifake_dataset("test")
    
    # Create train splits
    train_splits = create_random_splits(
        train_dataset,
        CONFIG['num_splits'],
        CONFIG['train_base_seed'],
        "train"
    )
    
    # Create test splits
    test_splits = create_random_splits(
        test_dataset,
        CONFIG['num_splits'],
        CONFIG['test_base_seed'],
        "test"
    )
    
    # Save train splits
    print(f"\n{'='*70}")
    print("Saving train splits to pickle files")
    print(f"{'='*70}")
    
    train_splits_info = []
    for i, split_data in enumerate(train_splits):
        split_num = i + 1
        filepath = output_dir / f"train_split_{split_num}.pkl"
        save_split_to_pickle(split_data, filepath)
        
        train_splits_info.append({
            "path": filepath.name,
            "num_samples": len(split_data),
            "seed": CONFIG['train_base_seed'] + i,
            "real_count": sum(1 for s in split_data if s["label"] == 0),
            "ai_count": sum(1 for s in split_data if s["label"] == 1),
        })
    
    # Save test splits
    print(f"\n{'='*70}")
    print("Saving test splits to pickle files")
    print(f"{'='*70}")
    
    test_splits_info = []
    for i, split_data in enumerate(test_splits):
        split_num = i + 1
        filepath = output_dir / f"test_split_{split_num}.pkl"
        save_split_to_pickle(split_data, filepath)
        
        test_splits_info.append({
            "path": filepath.name,
            "num_samples": len(split_data),
            "seed": CONFIG['test_base_seed'] + i,
            "real_count": sum(1 for s in split_data if s["label"] == 0),
            "ai_count": sum(1 for s in split_data if s["label"] == 1),
        })
    
    # Create and save metadata
    print(f"\n{'='*70}")
    print("Creating metadata")
    print(f"{'='*70}")
    
    metadata = create_metadata(train_splits_info, test_splits_info)
    metadata_path = output_dir / "metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Preparation Complete!")
    print(f"{'='*70}")
    print(f"\nCreated {CONFIG['num_splits']} train splits:")
    for i in range(CONFIG['num_splits']):
        print(f"  train_split_{i+1}.pkl")
    
    print(f"\nCreated {CONFIG['num_splits']} test splits:")
    for i in range(CONFIG['num_splits']):
        print(f"  test_split_{i+1}.pkl")
    
    print(f"\nAll files saved to: {output_dir.absolute()}")
    print(f"\n✓ Ready to use with PickledCIFAKEDataset!")


if __name__ == "__main__":
    main()