# GenAI Projects 2025

This repository contains various Generative AI and Deep Learning projects.

## Projects

### 1. Real vs AI Image Classifier
A computer vision project that fine-tunes a **DINOv2-Large** Vision Transformer to distinguish between real photographs and AI-generated images.
- **Dataset**: CIFAKE (120k images).
- **Key Techniques**: Transfer learning, smartphone simulation augmentation, focal loss.
- **Goal**: Detect synthetic media with high accuracy (92-95%).
- **Location**: [`real_vs_ai_image/`](real_vs_ai_image/)

### 2. Shazam for Music Genres
An audio classification project that identifies music genres from audio clips using **Audio Spectrogram Transformers (AST)**.
- **Dataset**: GTZAN (10 genres).
- **Key Techniques**: Mel spectrogram processing, AST fine-tuning, embedding space analysis (t-SNE/PCA).
- **Goal**: Accurately classify music genres and analyze genre similarities/confusion.
- **Location**: [`shazam_for_music_genres/`](shazam_for_music_genres/)
