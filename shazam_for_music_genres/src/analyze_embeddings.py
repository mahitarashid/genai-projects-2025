"""
Embedding analysis and t-SNE visualization for genre classification.

Analyzes the learned representations to understand:
- How genres cluster in embedding space
- Which genres overlap (especially rock)
- Why certain genres have lower accuracy
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import GTZANDataset, GENRE_LABELS, LABEL_TO_GENRE, NUM_LABELS
from src.preprocessing import get_device


# Color palette for genres (colorblind-friendly)
GENRE_COLORS = {
    "blues": "#1f77b4",
    "classical": "#ff7f0e", 
    "country": "#2ca02c",
    "disco": "#d62728",
    "hiphop": "#9467bd",
    "jazz": "#8c564b",
    "metal": "#e377c2",
    "pop": "#7f7f7f",
    "reggae": "#bcbd22",
    "rock": "#17becf"  # Cyan for rock - highlighted
}


def extract_embeddings(
    model: ASTForAudioClassification,
    dataloader: DataLoader,
    dataset: "GTZANDataset",
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract embeddings from the model's penultimate layer.
    
    Uses a hook to capture the output before the classification head.
    
    Args:
        model: Trained AST model
        dataloader: DataLoader for the data
        dataset: Dataset to get file paths from
        device: Device to run inference on
        
    Returns:
        Tuple of (embeddings, labels, file_paths)
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_labels = []
    all_file_paths = []
    
    # Hook to capture embeddings before classifier
    embeddings_buffer = []
    
    def hook_fn(module, input, output):
        # The input to the classifier is the embedding we want
        embeddings_buffer.append(input[0].detach().cpu())
    
    # Register hook on the classifier layer
    hook = model.classifier.dense.register_forward_hook(hook_fn)
    
    # Track sample indices
    sample_idx = 0
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                input_values = batch["input_values"].to(device)
                labels = batch["labels"]
                batch_size = len(labels)
                
                # Forward pass triggers the hook
                _ = model(input_values)
                
                all_labels.extend(labels.numpy())
                
                # Get file paths for this batch
                for i in range(batch_size):
                    all_file_paths.append(dataset.file_paths[sample_idx + i])
                sample_idx += batch_size
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings_buffer, dim=0).numpy()
        
    finally:
        hook.remove()
    
    return all_embeddings, np.array(all_labels), all_file_paths


def compute_pca(
    embeddings: np.ndarray,
    n_components: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute PCA projection and explained variance.
    
    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of PCA components
        random_state: Random seed
        
    Returns:
        Tuple of (pca_coords, explained_variance_ratio, total_variance_explained)
    """
    # Limit n_components to min(n_samples, n_features)
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    
    print(f"Computing PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_coords = pca.fit_transform(embeddings)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance = np.sum(explained_variance_ratio)
    
    print(f"  Total variance explained by {n_components} components: {total_variance:.1%}")
    print(f"  First 2 components explain: {explained_variance_ratio[0]:.1%} + {explained_variance_ratio[1]:.1%} = {explained_variance_ratio[:2].sum():.1%}")
    
    return pca_coords, explained_variance_ratio, total_variance


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    pca_components: int = 50
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute t-SNE projection with PCA preprocessing.
    
    First reduces dimensionality with PCA (to speed up t-SNE and get variance info),
    then applies t-SNE for final 2D projection.
    
    Args:
        embeddings: High-dimensional embeddings
        perplexity: t-SNE perplexity parameter
        max_iter: Maximum number of iterations
        random_state: Random seed
        pca_components: Number of PCA components for preprocessing
        
    Returns:
        Tuple of (tsne_coords, pca_explained_variance_ratio, total_pca_variance)
    """
    # First apply PCA
    pca_coords, explained_variance_ratio, total_variance = compute_pca(
        embeddings, n_components=pca_components, random_state=random_state
    )
    
    print(f"Computing t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init='pca'
    )
    
    tsne_coords = tsne.fit_transform(pca_coords)
    
    return tsne_coords, explained_variance_ratio, total_variance


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot PCA explained variance (scree plot).
    
    Args:
        explained_variance_ratio: Variance explained by each component
        output_path: Path to save the plot
    """
    n_components = len(explained_variance_ratio)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7, color='steelblue')
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title("Variance Explained by Each Component", fontsize=14)
    ax1.set_xticks(range(1, min(n_components + 1, 11)))
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, n_components + 1), cumulative_variance, 'b-o', markersize=4)
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax2.set_title("Cumulative Variance Explained", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Find components needed for 90% and 95%
    n_90 = np.argmax(cumulative_variance >= 0.9) + 1
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    ax2.annotate(f'{n_90} components\nfor 90%', xy=(n_90, 0.9), xytext=(n_90 + 5, 0.85),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"PCA variance plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tsne(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[Path] = None,
    highlight_genre: Optional[str] = "rock",
    title: str = "t-SNE Visualization of Genre Embeddings",
    pca_variance_info: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot t-SNE visualization with genre labels.
    
    Args:
        tsne_coords: 2D t-SNE coordinates
        labels: Genre labels
        output_path: Path to save the plot
        highlight_genre: Genre to highlight (default: rock)
        title: Plot title
        pca_variance_info: Tuple of (first_2_components_variance, total_pca_variance)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each genre
    for genre_name, label_idx in GENRE_LABELS.items():
        mask = labels == label_idx
        
        # Determine marker size and alpha
        if genre_name == highlight_genre:
            marker_size = 120
            alpha = 1.0
            zorder = 10
            edgecolor = 'black'
            linewidth = 1.5
        else:
            marker_size = 60
            alpha = 0.6
            zorder = 5
            edgecolor = 'none'
            linewidth = 0
        
        ax.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            c=GENRE_COLORS[genre_name],
            label=genre_name.capitalize(),
            s=marker_size,
            alpha=alpha,
            zorder=zorder,
            edgecolors=edgecolor,
            linewidths=linewidth
        )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    
    # Add variance info to title if available
    if pca_variance_info:
        first_2_var, total_var = pca_variance_info
        title = f"{title}\n(PCA preprocessing: {total_var:.1%} total variance retained, first 2 PCs explain {first_2_var:.1%})"
    
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_genre_neighbors(
    embeddings: np.ndarray,
    labels: np.ndarray,
    target_genre: str = "rock",
    n_neighbors: int = 10
) -> Dict[str, float]:
    """
    Analyze which genres are nearest neighbors to a target genre.
    
    Args:
        embeddings: Genre embeddings
        labels: Genre labels
        target_genre: Genre to analyze
        n_neighbors: Number of neighbors to consider
        
    Returns:
        Dictionary mapping genres to proportion of neighbors
    """
    target_label = GENRE_LABELS[target_genre]
    target_mask = labels == target_label
    target_embeddings = embeddings[target_mask]
    
    # Fit nearest neighbors on all embeddings
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine')
    nn.fit(embeddings)
    
    # Find neighbors for each target sample
    neighbor_counts = {genre: 0 for genre in GENRE_LABELS.keys()}
    total_neighbors = 0
    
    for i, emb in enumerate(target_embeddings):
        distances, indices = nn.kneighbors(emb.reshape(1, -1))
        
        # Skip first neighbor (itself)
        for idx in indices[0][1:]:
            neighbor_label = labels[idx]
            neighbor_genre = LABEL_TO_GENRE[neighbor_label]
            neighbor_counts[neighbor_genre] += 1
            total_neighbors += 1
    
    # Convert to proportions
    neighbor_proportions = {
        genre: count / total_neighbors
        for genre, count in neighbor_counts.items()
    }
    
    return neighbor_proportions


def find_confusing_instances(
    embeddings: np.ndarray,
    labels: np.ndarray,
    file_paths: List[str],
    target_genre: str = "rock",
    n_neighbors: int = 10,
    top_k: int = 10
) -> List[Dict]:
    """
    Find the most confusing instances of a target genre.
    
    Confusion score is based on the proportion of neighbors that belong
    to other genres (i.e., not the target genre).
    
    Args:
        embeddings: All embeddings
        labels: Genre labels
        file_paths: File paths for each sample
        target_genre: Genre to analyze
        n_neighbors: Number of neighbors to consider
        top_k: Number of confusing instances to return
        
    Returns:
        List of dictionaries with confusing instance info, sorted by confusion score
    """
    target_label = GENRE_LABELS[target_genre]
    target_mask = labels == target_label
    target_indices = np.where(target_mask)[0]
    target_embeddings = embeddings[target_mask]
    
    # Fit nearest neighbors on all embeddings
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine')
    nn.fit(embeddings)
    
    # Compute confusion score for each target sample
    confusion_results = []
    
    for i, (emb, global_idx) in enumerate(zip(target_embeddings, target_indices)):
        distances, indices = nn.kneighbors(emb.reshape(1, -1))
        
        # Count neighbors by genre (skip first - itself)
        neighbor_genres = {}
        other_genre_count = 0
        
        for idx in indices[0][1:]:
            neighbor_label = labels[idx]
            neighbor_genre = LABEL_TO_GENRE[neighbor_label]
            neighbor_genres[neighbor_genre] = neighbor_genres.get(neighbor_genre, 0) + 1
            
            if neighbor_genre != target_genre:
                other_genre_count += 1
        
        # Confusion score: proportion of neighbors from other genres
        confusion_score = other_genre_count / n_neighbors
        
        # Find the most common confusing genre
        confusing_genres = {g: c for g, c in neighbor_genres.items() if g != target_genre}
        if confusing_genres:
            top_confusing_genre = max(confusing_genres.items(), key=lambda x: x[1])
        else:
            top_confusing_genre = (None, 0)
        
        confusion_results.append({
            "file_path": file_paths[global_idx],
            "confusion_score": confusion_score,
            "neighbor_breakdown": neighbor_genres,
            "top_confusing_genre": top_confusing_genre[0],
            "top_confusing_count": top_confusing_genre[1],
            "same_genre_neighbors": neighbor_genres.get(target_genre, 0),
            "total_neighbors": n_neighbors
        })
    
    # Sort by confusion score (highest first)
    confusion_results.sort(key=lambda x: x["confusion_score"], reverse=True)
    
    return confusion_results[:top_k]


def print_confusing_instances(
    confusing_instances: List[Dict],
    target_genre: str
) -> None:
    """
    Print the most confusing instances for manual review.
    
    Args:
        confusing_instances: List of confusing instance info
        target_genre: The target genre being analyzed
    """
    print("\n" + "=" * 80)
    print(f"TOP {len(confusing_instances)} MOST CONFUSING {target_genre.upper()} INSTANCES")
    print("=" * 80)
    print("\nThese samples are most likely to be misclassified (high confusion score = ")
    print("many neighbors from other genres). Review these files manually.\n")
    
    for i, instance in enumerate(confusing_instances):
        print(f"{i+1:2}. {instance['file_path']}")
        print(f"    Confusion Score: {instance['confusion_score']:.1%}")
        print(f"    Same genre neighbors: {instance['same_genre_neighbors']}/{instance['total_neighbors']}")
        
        if instance['top_confusing_genre']:
            print(f"    Most confused with: {instance['top_confusing_genre']} ({instance['top_confusing_count']} neighbors)")
        
        # Show full breakdown
        breakdown = instance['neighbor_breakdown']
        breakdown_str = ", ".join(f"{g}:{c}" for g, c in sorted(breakdown.items(), key=lambda x: -x[1]))
        print(f"    Neighbor breakdown: {breakdown_str}")
        print()
    
    print("=" * 80)


def plot_neighbor_analysis(
    neighbor_proportions: Dict[str, float],
    target_genre: str = "rock",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart of neighbor genre distribution.
    
    Args:
        neighbor_proportions: Genre to proportion mapping
        target_genre: The analyzed genre
        output_path: Path to save the plot
    """
    # Sort by proportion
    sorted_genres = sorted(
        neighbor_proportions.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    genres = [g[0] for g in sorted_genres]
    proportions = [g[1] for g in sorted_genres]
    colors = [GENRE_COLORS[g] for g in genres]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(genres, proportions, color=colors, edgecolor='black', linewidth=0.5)
    
    # Highlight target genre's bar
    for i, genre in enumerate(genres):
        if genre == target_genre:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)
    
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Proportion of Neighbors", fontsize=12)
    ax.set_title(f"Nearest Neighbors of '{target_genre.capitalize()}' Samples in Embedding Space", fontsize=14)
    ax.set_xticklabels([g.capitalize() for g in genres], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, prop) in enumerate(zip(bars, proportions)):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{prop:.1%}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Neighbor analysis plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def compute_genre_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute centroid for each genre in embedding space.
    
    Args:
        embeddings: All embeddings
        labels: Genre labels
        
    Returns:
        Dictionary mapping genre names to centroid vectors
    """
    centroids = {}
    for genre_name, label_idx in GENRE_LABELS.items():
        mask = labels == label_idx
        genre_embeddings = embeddings[mask]
        centroids[genre_name] = np.mean(genre_embeddings, axis=0)
    
    return centroids


def compute_centroid_distances(
    centroids: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute pairwise distances between genre centroids.
    
    Args:
        centroids: Genre centroids
        
    Returns:
        Distance matrix (genres x genres)
    """
    genre_names = list(GENRE_LABELS.keys())
    n_genres = len(genre_names)
    
    distance_matrix = np.zeros((n_genres, n_genres))
    
    for i, g1 in enumerate(genre_names):
        for j, g2 in enumerate(genre_names):
            # Cosine distance
            c1, c2 = centroids[g1], centroids[g2]
            cosine_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
            distance_matrix[i, j] = 1 - cosine_sim
    
    return distance_matrix


def plot_centroid_distances(
    distance_matrix: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot heatmap of centroid distances between genres.
    
    Args:
        distance_matrix: Pairwise distance matrix
        output_path: Path to save the plot
    """
    genre_names = [g.capitalize() for g in GENRE_LABELS.keys()]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        distance_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        xticklabels=genre_names,
        yticklabels=genre_names,
        square=True,
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_title("Cosine Distance Between Genre Centroids\n(Lower = More Similar)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Centroid distance plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_genre_analysis(
    neighbor_proportions: Dict[str, float],
    distance_matrix: np.ndarray,
    target_genre: str = "rock"
) -> None:
    """
    Print detailed analysis of a target genre's overlaps with other genres.
    
    Args:
        neighbor_proportions: Genre to proportion mapping
        distance_matrix: Pairwise distance matrix between genre centroids
        target_genre: The genre to analyze
    """
    genre_names = list(GENRE_LABELS.keys())
    
    if target_genre not in genre_names:
        print(f"Error: '{target_genre}' is not a valid genre.")
        print(f"Valid genres: {', '.join(genre_names)}")
        return
    
    target_idx = genre_names.index(target_genre)
    target_capitalized = target_genre.capitalize()
    
    print("\n" + "=" * 60)
    print(f"{target_capitalized.upper()} GENRE ANALYSIS")
    print("=" * 60)
    
    # Nearest neighbors analysis
    print("\n1. Nearest Neighbors in Embedding Space:")
    print("-" * 40)
    sorted_neighbors = sorted(
        neighbor_proportions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for genre, prop in sorted_neighbors:
        if genre != target_genre:
            print(f"   {genre:12}: {prop:6.1%}")
    
    # Centroid distance analysis
    print(f"\n2. Distance from {target_capitalized} Centroid (lower = more similar):")
    print("-" * 40)
    target_distances = [(genre_names[i], distance_matrix[target_idx, i])
                        for i in range(len(genre_names)) if i != target_idx]
    target_distances.sort(key=lambda x: x[1])
    
    for genre, dist in target_distances:
        similarity = (1 - dist) * 100
        print(f"   {genre:12}: distance={dist:.3f} (similarity={similarity:.1f}%)")
    
    # Summary
    print(f"\n3. Most Confusable Genres with {target_capitalized}:")
    print("-" * 40)
    top_overlaps = [g for g in sorted_neighbors if g[0] != target_genre][:3]
    for i, (genre, prop) in enumerate(top_overlaps):
        print(f"   {i+1}. {genre.capitalize()} ({prop:.1%} of neighbors)")
    
    print("\n" + "=" * 60)


def analyze(
    model_path: str,
    data_dir: str,
    split: str = "test",
    output_dir: Optional[str] = None,
    batch_size: int = 8,
    perplexity: int = 30,
    target_genre: str = "rock",
    top_k: int = 10,
    n_neighbors: int = 10
) -> Dict:
    """
    Full embedding analysis pipeline.
    
    Args:
        model_path: Path to saved model
        data_dir: Path to data directory
        split: Which split to analyze
        output_dir: Directory to save results
        batch_size: Batch size for inference
        perplexity: t-SNE perplexity
        target_genre: Genre to focus analysis on
        top_k: Number of most confusing instances to report
        n_neighbors: Number of neighbors for confusion analysis
        
    Returns:
        Dictionary with analysis results
    """
    # Validate target genre
    if target_genre not in GENRE_LABELS:
        raise ValueError(
            f"Invalid genre '{target_genre}'. "
            f"Valid options: {', '.join(GENRE_LABELS.keys())}"
        )
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = ASTForAudioClassification.from_pretrained(model_path)
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
    
    # Load dataset
    data_dir = Path(data_dir)
    split_file = data_dir / f"{split}_split.jsonl"
    
    print(f"Loading {split} dataset...")
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
    
    # Extract embeddings
    print(f"\nExtracting embeddings from {len(dataset)} samples...")
    embeddings, labels, file_paths = extract_embeddings(model, dataloader, dataset, device)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute PCA + t-SNE
    tsne_coords, explained_variance_ratio, total_pca_variance = compute_tsne(
        embeddings, perplexity=perplexity
    )
    first_2_variance = explained_variance_ratio[:2].sum()
    
    # Analyze target genre neighbors
    print(f"\nAnalyzing {target_genre} genre neighbors...")
    neighbor_proportions = analyze_genre_neighbors(embeddings, labels, target_genre, n_neighbors)
    
    # Find most confusing instances
    print(f"\nFinding top {top_k} most confusing {target_genre} instances...")
    confusing_instances = find_confusing_instances(
        embeddings, labels, file_paths, target_genre, n_neighbors, top_k
    )
    
    # Compute centroid distances
    print("Computing genre centroid distances...")
    centroids = compute_genre_centroids(embeddings, labels)
    distance_matrix = compute_centroid_distances(centroids)
    
    # Print analysis
    print_genre_analysis(neighbor_proportions, distance_matrix, target_genre)
    print_confusing_instances(confusing_instances, target_genre)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot PCA variance
        plot_pca_variance(
            explained_variance_ratio,
            output_path=output_dir / f"{split}_pca_variance.png"
        )
        
        # Plot t-SNE with variance info
        plot_tsne(
            tsne_coords, labels,
            output_path=output_dir / f"{split}_tsne_{target_genre}.png",
            highlight_genre=target_genre,
            pca_variance_info=(first_2_variance, total_pca_variance)
        )
        
        # Plot neighbor analysis
        plot_neighbor_analysis(
            neighbor_proportions,
            target_genre=target_genre,
            output_path=output_dir / f"{split}_{target_genre}_neighbors.png"
        )
        
        # Plot centroid distances
        plot_centroid_distances(
            distance_matrix,
            output_path=output_dir / f"{split}_centroid_distances.png"
        )
        
        # Save embeddings and coordinates for further analysis
        np.save(output_dir / f"{split}_embeddings.npy", embeddings)
        np.save(output_dir / f"{split}_tsne_coords.npy", tsne_coords)
        np.save(output_dir / f"{split}_labels.npy", labels)
        
        # Save analysis results (convert numpy types to Python native types for JSON)
        results = {
            "target_genre": target_genre,
            "neighbor_proportions": {k: float(v) for k, v in neighbor_proportions.items()},
            "distance_matrix": distance_matrix.tolist(),
            "genre_names": list(GENRE_LABELS.keys()),
            "pca_explained_variance_ratio": [float(v) for v in explained_variance_ratio.tolist()],
            "pca_total_variance": float(total_pca_variance),
            "pca_first_2_variance": float(first_2_variance),
            "confusing_instances": confusing_instances
        }
        with open(output_dir / f"{split}_{target_genre}_analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save confusing instances as a separate file for easy access
        confusing_file = output_dir / f"{split}_{target_genre}_confusing_instances.jsonl"
        with open(confusing_file, 'w') as f:
            for instance in confusing_instances:
                f.write(json.dumps(instance) + "\n")
        print(f"Confusing instances saved to {confusing_file}")
        
        print(f"\nAll results saved to {output_dir}")
    
    return {
        "target_genre": target_genre,
        "embeddings": embeddings,
        "tsne_coords": tsne_coords,
        "labels": labels,
        "file_paths": file_paths,
        "neighbor_proportions": neighbor_proportions,
        "distance_matrix": distance_matrix,
        "pca_explained_variance_ratio": explained_variance_ratio,
        "pca_total_variance": total_pca_variance,
        "confusing_instances": confusing_instances
    }


def main():
    """Main entry point for embedding analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze genre embeddings")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved model")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to analyze")
    parser.add_argument("--output-dir", type=str, default="analysis",
                        help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter")
    parser.add_argument("--target-genre", type=str, default="rock",
                        choices=list(GENRE_LABELS.keys()),
                        help="Genre to focus analysis on (default: rock)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of most confusing instances to report (default: 10)")
    parser.add_argument("--n-neighbors", type=int, default=10,
                        help="Number of neighbors for confusion analysis (default: 10)")
    
    args = parser.parse_args()
    
    analyze(
        model_path=args.model_path,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        perplexity=args.perplexity,
        target_genre=args.target_genre,
        top_k=args.top_k,
        n_neighbors=args.n_neighbors
    )


if __name__ == "__main__":
    main()