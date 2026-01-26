"""
Training script for music genre classification using AST.

Features:
- HuggingFace Trainer integration
- Early stopping based on validation accuracy
- Model checkpointing (best model saved)
- Console logging
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    ASTFeatureExtractor
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import GTZANDataset, GENRE_LABELS, NUM_LABELS
from src.model import load_ast_model, get_model_summary


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Data paths
    data_dir: str = "data/processed"
    output_dir: str = "checkpoints"
    
    # Model config
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    freeze_backbone: bool = True
    
    # Training hyperparameters
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0001
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    
    # Hardware
    fp16: bool = True  # Use mixed precision if available
    dataloader_num_workers: int = 4


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    # Per-class accuracy
    per_class_acc = {}
    for i, genre in enumerate(GENRE_LABELS.keys()):
        mask = labels == i
        if mask.sum() > 0:
            per_class_acc[f"acc_{genre}"] = accuracy_score(
                labels[mask], preds[mask]
            )
    
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        **per_class_acc
    }
    
    return metrics


def create_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Create HuggingFace TrainingArguments from config.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingArguments instance
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        
        # Logging
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        report_to=["none"],
        
        # Hardware optimization
        fp16=config.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        
        # Misc
        remove_unused_columns=False,
        push_to_hub=False,
    )


def train(config: Optional[TrainingConfig] = None) -> Trainer:
    """
    Run training pipeline.
    
    Args:
        config: Training configuration. If None, uses defaults.
        
    Returns:
        Trained Trainer instance
    """
    if config is None:
        config = TrainingConfig()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / config.data_dir
    output_dir = project_root / config.output_dir
    
    print("=" * 60)
    print("Music Genre Classification Training")
    print("=" * 60)
    
    # Load feature extractor
    print("\nLoading feature extractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(config.model_name)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = GTZANDataset(
        split_file=data_dir / "train_split.jsonl",
        feature_extractor=feature_extractor,
        mode="train"
    )
    val_dataset = GTZANDataset(
        split_file=data_dir / "val_split.jsonl",
        feature_extractor=feature_extractor,
        mode="eval"
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Load model
    print("\nLoading model...")
    model = load_ast_model(
        model_name=config.model_name,
        num_labels=NUM_LABELS,
        freeze_backbone=config.freeze_backbone
    )
    print(get_model_summary(model))
    
    # Create training arguments
    training_args = create_training_args(config)
    training_args.output_dir = str(output_dir)
    training_args.logging_dir = str(output_dir / "logs")
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 60)
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    feature_extractor.save_pretrained(str(final_model_path))
    
    # Print training results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Best model saved to: {final_model_path}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    return trainer


def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train genre classifier")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--unfreeze", action="store_true",
                        help="Unfreeze backbone (full fine-tuning)")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone=not args.unfreeze,
    )
    
    train(config)


if __name__ == "__main__":
    main()