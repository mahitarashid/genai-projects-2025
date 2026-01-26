"""
Inference script for music genre prediction.

Provides functionality to:
- Load a trained model
- Process audio files
- Predict genre with confidence scores
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_audio,
    center_crop,
    segment_audio,
    get_device,
    SEGMENT_SAMPLES,
    TARGET_SAMPLE_RATE
)
from src.dataset import GENRE_LABELS, LABEL_TO_GENRE, NUM_LABELS


class GenrePredictor:
    """
    Genre prediction from audio files.
    
    Loads a trained model and provides methods to predict
    genre from audio files.
    
    Args:
        model_path: Path to saved model directory
        device: Device to run inference on (auto-detected if None)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        
        self.device = get_device(device)
        
        print(f"Loading model from {self.model_path}...")
        print(f"Using device: {self.device}")
        
        # Load model and feature extractor
        self.model = ASTForAudioClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(str(self.model_path))
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def predict(
        self,
        audio_path: Union[str, Path],
        return_all_probs: bool = False
    ) -> Dict:
        """
        Predict genre from an audio file.
        
        Uses center crop for single prediction.
        
        Args:
            audio_path: Path to audio file
            return_all_probs: Whether to include all class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess audio
        waveform, sr = load_audio(audio_path, TARGET_SAMPLE_RATE)
        waveform = center_crop(waveform, SEGMENT_SAMPLES)
        
        # Convert to spectrogram
        waveform_np = waveform.squeeze(0).numpy()
        features = self.feature_extractor(
            waveform_np,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt"
        )
        
        # Move to device and run inference
        input_values = features.input_values.to(self.device)
        outputs = self.model(input_values)
        
        # Get probabilities
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        
        # Get top prediction
        pred_idx = probs.argmax().item()
        pred_genre = LABEL_TO_GENRE[pred_idx]
        confidence = probs[pred_idx].item()
        
        result = {
            "file": str(audio_path),
            "predicted_genre": pred_genre,
            "confidence": confidence
        }
        
        if return_all_probs:
            result["all_probabilities"] = {
                LABEL_TO_GENRE[i]: probs[i].item()
                for i in range(NUM_LABELS)
            }
        
        return result
    
    @torch.no_grad()
    def predict_with_segments(
        self,
        audio_path: Union[str, Path],
        overlap: float = 0.5,
        aggregation: str = "mean"
    ) -> Dict:
        """
        Predict genre using multiple overlapping segments.
        
        More robust prediction by averaging predictions across
        multiple segments of the audio.
        
        Args:
            audio_path: Path to audio file
            overlap: Overlap ratio between segments (0.0 to 1.0)
            aggregation: How to aggregate predictions ("mean" or "vote")
            
        Returns:
            Dictionary with prediction results
        """
        # Load audio (full file)
        waveform, sr = load_audio(audio_path, TARGET_SAMPLE_RATE)
        
        # Segment audio
        segments = segment_audio(waveform, SEGMENT_SAMPLES, overlap)
        
        if len(segments) == 0:
            # Audio too short, use padding
            waveform = center_crop(waveform, SEGMENT_SAMPLES)
            segments = [waveform]
        
        # Process each segment
        all_probs = []
        
        for segment in segments:
            waveform_np = segment.squeeze(0).numpy()
            features = self.feature_extractor(
                waveform_np,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt"
            )
            
            input_values = features.input_values.to(self.device)
            outputs = self.model(input_values)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()
            all_probs.append(probs)
        
        # Stack probabilities
        all_probs = torch.stack(all_probs)
        
        # Aggregate predictions
        if aggregation == "mean":
            avg_probs = all_probs.mean(dim=0)
            pred_idx = avg_probs.argmax().item()
            confidence = avg_probs[pred_idx].item()
            final_probs = avg_probs
        elif aggregation == "vote":
            # Majority voting
            votes = all_probs.argmax(dim=-1)
            pred_idx = votes.mode().values.item()
            # Confidence is proportion of votes
            confidence = (votes == pred_idx).float().mean().item()
            final_probs = all_probs.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        pred_genre = LABEL_TO_GENRE[pred_idx]
        
        result = {
            "file": str(audio_path),
            "predicted_genre": pred_genre,
            "confidence": confidence,
            "num_segments": len(segments),
            "aggregation": aggregation,
            "all_probabilities": {
                LABEL_TO_GENRE[i]: final_probs[i].item()
                for i in range(NUM_LABELS)
            }
        }
        
        return result
    
    def predict_batch(
        self,
        audio_paths: List[Union[str, Path]],
        use_segments: bool = False
    ) -> List[Dict]:
        """
        Predict genres for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            use_segments: Whether to use multi-segment prediction
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for path in audio_paths:
            try:
                if use_segments:
                    result = self.predict_with_segments(path)
                else:
                    result = self.predict(path, return_all_probs=True)
                results.append(result)
            except Exception as e:
                results.append({
                    "file": str(path),
                    "error": str(e)
                })
        
        return results


def predict_genre(
    audio_path: Union[str, Path],
    model_path: Union[str, Path],
    use_segments: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Convenience function for single-file prediction.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to saved model
        use_segments: Whether to use multi-segment prediction
        verbose: Whether to print results
        
    Returns:
        Prediction dictionary
    """
    predictor = GenrePredictor(model_path)
    
    if use_segments:
        result = predictor.predict_with_segments(audio_path)
    else:
        result = predictor.predict(audio_path, return_all_probs=True)
    
    if verbose:
        print("\n" + "=" * 50)
        print("Genre Prediction Results")
        print("=" * 50)
        print(f"File: {result['file']}")
        print(f"Predicted Genre: {result['predicted_genre'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if "num_segments" in result:
            print(f"Segments analyzed: {result['num_segments']}")
        
        if "all_probabilities" in result:
            print("\nAll Probabilities:")
            sorted_probs = sorted(
                result["all_probabilities"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for genre, prob in sorted_probs:
                bar = "█" * int(prob * 20)
                print(f"  {genre:12} {prob:6.2%} {bar}")
        
        print("=" * 50)
    
    return result


def main():
   
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict music genre")
    parser.add_argument("audio_path", type=str,
                        help="Path to audio file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved model")
    parser.add_argument("--segments", action="store_true",
                        help="Use multi-segment prediction")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    result = predict_genre(
        audio_path=args.audio_path,
        model_path=args.model_path,
        use_segments=args.segments,
        verbose=not args.quiet
    )
    
    # Return exit code based on success
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())