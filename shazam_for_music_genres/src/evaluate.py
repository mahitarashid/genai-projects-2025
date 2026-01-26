"""
Evaluation script for music genre classification.

Computes:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix visualization
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import GTZANDataset, GENRE_LABELS, LABEL_TO_GENRE, NUM_LABELS
from src.preprocessing import get_device


def evaluate_model(
    model: ASTForAudioClassification,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a dataset and collect predictions.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        device: Device to run inference on
        
    Returns:
        Tuple of (all_predictions, all_labels)
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_values)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary containing all metrics
    """
    genre_names = [LABEL_TO_GENRE[i] for i in range(NUM_LABELS)]
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }
    
    # Per-class metrics
    per_class_report = classification_report(
        labels, predictions,
        target_names=genre_names,
        output_dict=True
    )
    metrics["per_class"] = per_class_report
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    output_path: Optional[Path] = None,
    normalize: bool = True
) -> None:
    """
    Plot and optionally save confusion matrix.
    
    Args:
        confusion_mat: Confusion matrix array
        output_path: Path to save the plot (optional)
        normalize: Whether to normalize the matrix
    """
    genre_names = [LABEL_TO_GENRE[i] for i in range(NUM_LABELS)]
    
    if normalize:
        # Normalize by row (true labels)
        cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1, keepdims=True)
        data = cm_normalized
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
    else:
        data = confusion_mat
        fmt = "d"
        title = "Confusion Matrix"
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=genre_names,
        yticklabels=genre_names,
        square=True
    )
    plt.xlabel("Predicted Genre", fontsize=12)
    plt.ylabel("True Genre", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(
    metrics: Dict,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot per-class precision, recall, and F1 scores.
    
    Args:
        metrics: Metrics dictionary from compute_all_metrics
        output_path: Path to save the plot (optional)
    """
    genre_names = [LABEL_TO_GENRE[i] for i in range(NUM_LABELS)]
    per_class = metrics["per_class"]
    
    precisions = [per_class[genre]["precision"] for genre in genre_names]
    recalls = [per_class[genre]["recall"] for genre in genre_names]
    f1_scores = [per_class[genre]["f1-score"] for genre in genre_names]
    
    x = np.arange(len(genre_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#2ecc71")
    ax.bar(x, recalls, width, label="Recall", color="#3498db")
    ax.bar(x + width, f1_scores, width, label="F1-Score", color="#e74c3c")
    
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(genre_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Per-class metrics plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_evaluation_report(metrics: Dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):       {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):    {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    
    print(f"\nPer-Class Performance:")
    print("-" * 60)
    print(f"{'Genre':12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    
    genre_names = [LABEL_TO_GENRE[i] for i in range(NUM_LABELS)]
    for genre in genre_names:
        stats = metrics["per_class"][genre]
        print(f"{genre:12} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
              f"{stats['f1-score']:>10.4f} {stats['support']:>10.0f}")
    
    print("=" * 60)


def evaluate(
    model_path: str,
    data_dir: str,
    split: str = "test",
    output_dir: Optional[str] = None,
    batch_size: int = 8
) -> Dict:
    """
    Full evaluation pipeline.
    
    Args:
        model_path: Path to saved model
        data_dir: Path to data directory with split files
        split: Which split to evaluate ("val" or "test")
        output_dir: Directory to save results and plots
        batch_size: Batch size for inference
        
    Returns:
        Dictionary of metrics
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and feature extractor
    print(f"\nLoading model from {model_path}...")
    model = ASTForAudioClassification.from_pretrained(model_path)
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
    
    # Load dataset
    data_dir = Path(data_dir)
    split_file = data_dir / f"{split}_split.jsonl"
    
    print(f"Loading {split} dataset from {split_file}...")
    dataset = GTZANDataset(
        split_file=split_file,
        feature_extractor=feature_extractor,
        mode="eval"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Run evaluation
    predictions, labels = evaluate_model(model, dataloader, device)
    
    # Compute metrics
    metrics = compute_all_metrics(predictions, labels)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics JSON
        metrics_path = output_dir / f"{split}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
        
        # Plot confusion matrix
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            output_path=output_dir / f"{split}_confusion_matrix.png"
        )
        
        # Plot per-class metrics
        plot_per_class_metrics(
            metrics,
            output_path=output_dir / f"{split}_per_class_metrics.png"
        )
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate genre classifier")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved model")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()