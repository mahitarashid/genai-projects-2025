"""
Model configuration for music genre classification using AST.

Provides utilities to:
- Load pretrained AST model from HuggingFace
- Configure for 10-class GTZAN classification
- Freeze backbone for transfer learning
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    ASTForAudioClassification,
    ASTConfig,
    ASTFeatureExtractor
)

from src.preprocessing import get_device


# Model configuration
DEFAULT_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_LABELS = 10

# Genre labels for GTZAN
GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


def load_ast_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = NUM_LABELS,
    freeze_backbone: bool = True,
    dropout: float = 0.1
) -> ASTForAudioClassification:
    """
    Load and configure AST model for genre classification.
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of output classes (10 for GTZAN)
        freeze_backbone: Whether to freeze the transformer backbone
        dropout: Dropout rate for classifier
        
    Returns:
        Configured ASTForAudioClassification model
    """
    # Load model with new classification head
    model = ASTForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # Reinitialize classifier head
    )
    
    # Set up label mappings
    model.config.id2label = {i: label for i, label in enumerate(GENRE_LABELS)}
    model.config.label2id = {label: i for i, label in enumerate(GENRE_LABELS)}
    
    # Freeze backbone if requested
    if freeze_backbone:
        freeze_model_backbone(model)
    
    return model


def freeze_model_backbone(model: ASTForAudioClassification) -> None:
    """
    Freeze the AST backbone, only training the classifier head.
    
    This is recommended for small datasets like GTZAN to prevent
    overfitting and speed up training.
    
    Args:
        model: AST model to freeze
    """
    # Freeze all backbone parameters
    for param in model.audio_spectrogram_transformer.parameters():
        param.requires_grad = False
    
    # Keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_model_backbone(
    model: ASTForAudioClassification,
    unfreeze_layers: Optional[int] = None
) -> None:
    """
    Unfreeze the AST backbone for fine-tuning.
    
    Can optionally unfreeze only the last N transformer layers
    for gradual unfreezing.
    
    Args:
        model: AST model to unfreeze
        unfreeze_layers: Number of last layers to unfreeze.
                        If None, unfreeze all layers.
    """
    backbone = model.audio_spectrogram_transformer
    
    if unfreeze_layers is None:
        # Unfreeze everything
        for param in backbone.parameters():
            param.requires_grad = True
    else:
        # Gradual unfreezing - unfreeze last N encoder layers
        encoder_layers = backbone.encoder.layer
        num_layers = len(encoder_layers)
        
        for i, layer in enumerate(encoder_layers):
            if i >= num_layers - unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_summary(model: ASTForAudioClassification) -> str:
    """
    Get a summary of model configuration and parameters.
    
    Args:
        model: AST model
        
    Returns:
        Summary string
    """
    trainable, total = count_parameters(model)
    
    summary = [
        "=" * 50,
        "Model Summary",
        "=" * 50,
        f"Model: {model.config.model_type}",
        f"Number of labels: {model.config.num_labels}",
        f"Labels: {GENRE_LABELS}",
        "-" * 50,
        f"Total parameters: {total:,}",
        f"Trainable parameters: {trainable:,}",
        f"Frozen parameters: {total - trainable:,}",
        f"Trainable ratio: {trainable / total * 100:.2f}%",
        "=" * 50
    ]
    
    return "\n".join(summary)


def load_feature_extractor(
    model_name: str = DEFAULT_MODEL_NAME
) -> ASTFeatureExtractor:
    """
    Load the feature extractor for the AST model.
    
    The feature extractor converts raw audio waveforms to
    mel spectrograms suitable for the model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        ASTFeatureExtractor instance
    """
    return ASTFeatureExtractor.from_pretrained(model_name)


class GenreClassifier:
    """
    Wrapper class for genre classification inference.
    
    Combines the model and feature extractor for easy inference.
    
    Args:
        model_path: Path to saved model checkpoint or HuggingFace identifier
        device: Device to run inference on
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None
    ):
        self.device = get_device(device)
        
        # Load model
        if model_path == DEFAULT_MODEL_NAME:
            self.model = load_ast_model(freeze_backbone=False)
        else:
            self.model = ASTForAudioClassification.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load feature extractor
        self.feature_extractor = load_feature_extractor()
    
    @torch.no_grad()
    def predict(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> Tuple[str, float, dict]:
        """
        Predict genre from audio waveform.
        
        Args:
            waveform: Audio waveform tensor of shape (1, num_samples)
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (predicted_genre, confidence, all_probabilities)
        """
        # Convert to numpy for feature extractor
        waveform_np = waveform.squeeze().numpy()
        
        # Extract features
        features = self.feature_extractor(
            waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Move to device
        input_values = features.input_values.to(self.device)
        
        # Forward pass
        outputs = self.model(input_values)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1).squeeze()
        
        # Get prediction
        pred_idx = probs.argmax().item()
        pred_genre = GENRE_LABELS[pred_idx]
        confidence = probs[pred_idx].item()
        
        # All probabilities
        all_probs = {
            GENRE_LABELS[i]: probs[i].item() 
            for i in range(len(GENRE_LABELS))
        }
        
        return pred_genre, confidence, all_probs