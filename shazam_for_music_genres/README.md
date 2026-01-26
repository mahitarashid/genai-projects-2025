# Shazam for Music Genres

A music genre classification system using the Audio Spectrogram Transformer (AST) model fine-tuned on the GTZAN dataset.

## Overview

This project implements a "Shazam for genres" - a system that takes a short audio clip and predicts its music genre. It uses transfer learning from the AST model pretrained on AudioSet.

**Input**: Short audio clip (WAV format)  
**Output**: Predicted genre (one of 10 GTZAN genres)

### Supported Genres
- Blues, Classical, Country, Disco, Hip-hop
- Jazz, Metal, Pop, Reggae, Rock

## Project Structure

```
shazam_for_music_genres/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── raw/                  # GTZAN dataset (download required)
│   └── processed/            # Train/val/test splits (JSONL, one sample per line)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Audio loading and preprocessing
│   ├── dataset.py            # PyTorch Dataset class
│   ├── model.py              # AST model loading and configuration
│   ├── train.py              # Training with HuggingFace Trainer
│   ├── evaluate.py           # Evaluation metrics and visualization
│   ├── inference.py          # Single-file genre prediction
│   └── analyze_embeddings.py # t-SNE visualization and genre overlap analysis
├── scripts/
│   ├── download_gtzan.py     # Download GTZAN from Kaggle
│   ├── prepare_splits.py     # Create train/val/test splits
│   └── test_pipeline.py      # End-to-end pipeline tests
├── checkpoints/              # Saved model checkpoints
└── notebooks/                # Jupyter notebooks for exploration
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download GTZAN from Kaggle:
- https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Or use the download script:
```bash
python scripts/download_gtzan.py
```

### 3. Prepare Data Splits

** note:** 
jazz.00054.wav file is malformed, so either ignore it or replace it with another one

```bash
python scripts/prepare_splits.py
```

This creates stratified train/val/test splits (80/10/10).

### 4. Run Pipeline Tests

Verify everything is working:
```bash
python scripts/test_pipeline.py
```

### 5. Train Model

```bash
# Basic training (frozen backbone, recommended)
python -m src.train --data-dir data/processed --output-dir checkpoints

# Full fine-tuning
python -m src.train --data-dir data/processed --unfreeze
```

### 6. Evaluate Model

```bash
python -m src.evaluate \
    --model-path checkpoints/final_model \
    --data-dir data/processed \
    --split test \
    --output-dir results
```

### 7. Predict Genre

```bash
# Single file prediction
python -m src.inference path/to/audio.wav --model-path checkpoints/final_model

# Multi-segment prediction (more robust)
python -m src.inference path/to/audio.wav --model-path checkpoints/final_model --segments
```

### 8. Analyze Embeddings (Genre Overlap Analysis)

```bash
# Analyze genre embeddings with t-SNE visualization (default: rock)
python -m src.analyze_embeddings \
    --model-path checkpoints/final_model \
    --data-dir data/processed \
    --split test \
    --output-dir analysis

# Analyze a specific genre (e.g., jazz, metal, classical)
python -m src.analyze_embeddings \
    --model-path checkpoints/final_model \
    --data-dir data/processed \
    --target-genre jazz \
    --output-dir analysis

# Get top 20 most confusing instances with 15 neighbors
python -m src.analyze_embeddings \
    --model-path checkpoints/final_model \
    --data-dir data/processed \
    --target-genre rock \
    --top-k 20 \
    --n-neighbors 15 \
    --output-dir analysis
```

Available genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

**Command line options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--target-genre` | rock | Genre to analyze |
| `--top-k` | 10 | Number of confusing instances to report |
| `--n-neighbors` | 10 | Neighbors to consider for confusion score |
| `--perplexity` | 30 | t-SNE perplexity parameter |

This generates (using `--target-genre rock` as example):
- `test_pca_variance.png` - Scree plot of PCA explained variance
- `test_tsne_rock.png` - t-SNE visualization with rock genre highlighted
- `test_rock_neighbors.png` - Bar chart of rock's nearest neighbors
- `test_centroid_distances.png` - Heatmap of genre similarity
- `test_rock_analysis.json` - Numerical results including PCA variance
- `test_rock_confusing_instances.jsonl` - Most confusing samples for manual review

## Model Details

### Architecture
- **Base Model**: Audio Spectrogram Transformer (AST)
- **Pretrained on**: AudioSet
- **Fine-tuned for**: 10-class GTZAN genre classification

### Training Configuration
| Parameter | Default Value |
|-----------|---------------|
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Epochs | 30 |
| Backbone | Frozen |
| Optimizer | AdamW |
| Early Stopping | Patience=5 |

### Audio Preprocessing
- Sample rate: 16 kHz
- Segment duration: 10 seconds
- Feature extraction: Mel spectrogram via ASTFeatureExtractor
- Augmentation: Random crop from 30s audio

## API Usage

### Python API

```python
from src.inference import GenrePredictor

# Load model
predictor = GenrePredictor("checkpoints/final_model")

# Predict genre
result = predictor.predict("path/to/audio.wav")
print(f"Genre: {result['predicted_genre']}")
print(f"Confidence: {result['confidence']:.2%}")

# Multi-segment prediction
result = predictor.predict_with_segments("path/to/audio.wav")
```

### Dataset Class

```python
from src.dataset import GTZANDataset, GTZANDataModule

# Single dataset
dataset = GTZANDataset(
    split_file="data/processed/train_split.json",
    mode="train"
)

# Full data module
data_module = GTZANDataModule(
    data_dir="data/processed",
    batch_size=8
)
train_loader = data_module.train_dataloader()
```

## Expected Results

With frozen backbone training on GTZAN:
- **Validation Accuracy**: ~70-80%
- **Test Accuracy**: ~65-75%

*Note: GTZAN is a small dataset with known annotation issues. Performance may vary.*

## Phase 0 Checklist

- [x] Project setup (requirements.txt, directory structure)
- [x] Data download and verification script
- [x] Train/val/test split preparation
- [x] Audio preprocessing module
- [x] PyTorch Dataset class
- [x] Data augmentation (random crop)
- [x] AST model loading and configuration
- [x] Training script with HuggingFace Trainer
- [x] Early stopping and checkpointing
- [x] Console logging
- [x] Evaluation script with metrics
- [x] Inference script
- [x] End-to-end tests

## Known Issues

### Rock Genre Performance
The 'rock' genre shows lower accuracy (30-70%) compared to other genres. Analysis shows significant overlap with:
- **Metal** - Similar instrumentation (electric guitars, drums)
- **Blues** - Rock's historical roots
- **Country** - Shared elements in country-rock

Run `python -m src.analyze_embeddings --target-genre rock` to visualize this overlap with t-SNE.

## Future Work (Phase 1)

- [x] Analyze model errors and confusion patterns (rock genre overlap)
- [ ] Test on audio beyond GTZAN dataset
- [ ] Compare with other pretrained models (e.g., Wav2Vec2)
- [ ] Add more data augmentation (time stretch, pitch shift)
- [ ] Web/API deployment

## License

MIT License

## References

- [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)
- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)