"""
RAG-based retrieval and classification for glaucoma screening
Retrieves similar images from database and aggregates their labels
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from build_rag_database import FeatureExtractor
from utils import ensure_dir, save_json


class RAGClassifier:
    """
    Retrieval-Augmented Classification for glaucoma screening
    """
    
    def __init__(self, index_path, metadata_path, k=10, aggregation='weighted_vote'):
        """
        Initialize RAG classifier
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to database metadata CSV
            k: Number of nearest neighbors to retrieve
            aggregation: Method to aggregate neighbor labels
                - 'majority_vote': Simple majority voting
                - 'weighted_vote': Weight by similarity (inverse distance)
                - 'mean_prob': Mean probability from neighbors
        """
        print("\n" + "="*60)
        print("Initializing RAG Classifier")
        print("="*60)
        
        self.k = k
        self.aggregation = aggregation
        
        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        self.metadata = pd.read_csv(metadata_path)
        print(f"✓ Loaded metadata for {len(self.metadata)} images")
        
        # Initialize feature extractor
        print("Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor(model_name='facebook/dinov3-vitl16-pretrain-lvd1689m')
        print("✓ Feature extractor ready")
        
        print("="*60 + "\n")
    
    def retrieve_neighbors(self, query_features, k=None):
        """
        Retrieve k nearest neighbors for query features
        
        Args:
            query_features: Feature vector for query image
            k: Number of neighbors (uses self.k if None)
        
        Returns:
            distances: Array of distances to neighbors
            indices: Array of neighbor indices in database
        """
        if k is None:
            k = self.k
        
        # Ensure query is 2D array
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_features.astype('float32'), k)
        
        return distances[0], indices[0]
    
    def aggregate_labels_majority(self, neighbor_indices):
        """Simple majority vote aggregation"""
        neighbor_labels = self.metadata.iloc[neighbor_indices]['label'].values
        # Count votes
        votes = np.bincount(neighbor_labels, minlength=2)
        # Probability is proportion of glaucoma votes
        prob_glaucoma = votes[1] / len(neighbor_labels)
        return prob_glaucoma
    
    def aggregate_labels_weighted(self, neighbor_indices, distances):
        """Weighted vote by similarity (inverse distance)"""
        neighbor_labels = self.metadata.iloc[neighbor_indices]['label'].values
        
        # Convert distances to similarities (inverse with small epsilon)
        epsilon = 1e-6
        similarities = 1 / (distances + epsilon)
        
        # Weighted vote
        weight_glaucoma = np.sum(similarities[neighbor_labels == 1])
        weight_total = np.sum(similarities)
        
        prob_glaucoma = weight_glaucoma / weight_total if weight_total > 0 else 0.5
        return prob_glaucoma
    
    def aggregate_labels_mean(self, neighbor_indices):
        """Mean probability from neighbors (treat labels as probabilities)"""
        neighbor_labels = self.metadata.iloc[neighbor_indices]['label'].values
        prob_glaucoma = np.mean(neighbor_labels)
        return prob_glaucoma
    
    def classify_image(self, image_path):
        """
        Classify a single image using RAG
        
        Args:
            image_path: Path to query image
        
        Returns:
            prob_glaucoma: Probability of glaucoma (0-1)
            neighbor_info: Dictionary with neighbor information
        """
        # Extract features from query image
        query_features = self.feature_extractor.extract_features(image_path)
        
        if query_features is None:
            return 0.5, None  # Return neutral probability on error
        
        # Retrieve neighbors
        distances, indices = self.retrieve_neighbors(query_features)
        
        # Aggregate labels based on chosen method
        if self.aggregation == 'majority_vote':
            prob_glaucoma = self.aggregate_labels_majority(indices)
        elif self.aggregation == 'weighted_vote':
            prob_glaucoma = self.aggregate_labels_weighted(indices, distances)
        elif self.aggregation == 'mean_prob':
            prob_glaucoma = self.aggregate_labels_mean(indices)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Gather neighbor information
        neighbor_info = {
            'distances': distances.tolist(),
            'indices': indices.tolist(),
            'labels': self.metadata.iloc[indices]['label'].tolist(),
            'datasets': self.metadata.iloc[indices]['dataset'].tolist() if 'dataset' in self.metadata.columns else []
        }
        
        return float(prob_glaucoma), neighbor_info
    
    def classify_batch(self, image_paths, return_neighbor_info=False):
        """
        Classify multiple images
        
        Args:
            image_paths: List of image paths
            return_neighbor_info: Whether to return detailed neighbor information
        
        Returns:
            probabilities: Array of glaucoma probabilities
            neighbor_infos: List of neighbor info dicts (if return_neighbor_info=True)
        """
        probabilities = []
        neighbor_infos = [] if return_neighbor_info else None
        
        for img_path in tqdm(image_paths, desc="Classifying images"):
            prob, info = self.classify_image(img_path)
            probabilities.append(prob)
            
            if return_neighbor_info:
                neighbor_infos.append(info)
        
        return np.array(probabilities), neighbor_infos


def evaluate_rag_classifier(classifier, test_csv, output_dir=None):
    """
    Evaluate RAG classifier on test set
    
    Args:
        classifier: RAGClassifier instance
        test_csv: Path to test CSV file
        output_dir: Directory to save results (optional)
    
    Returns:
        results: Dictionary with predictions and metrics
    """
    print("\n" + "="*60)
    print("Evaluating RAG Classifier")
    print("="*60)
    
    # Load test data
    print(f"Loading test data from {test_csv}...")
    df_test = pd.read_csv(test_csv)
    print(f"✓ Loaded {len(df_test)} test images")
    
    # Classify all test images
    print("\nClassifying test images...")
    image_paths = df_test['path'].tolist()
    true_labels = df_test['label'].values
    
    probabilities, neighbor_infos = classifier.classify_batch(
        image_paths, 
        return_neighbor_info=True
    )
    
    # Compile results
    results = {
        'image_paths': image_paths,
        'true_labels': true_labels.tolist(),
        'predicted_probabilities': probabilities.tolist(),
        'neighbor_infos': neighbor_infos,
        'k': classifier.k,
        'aggregation_method': classifier.aggregation
    }
    
    # Save results if output directory specified
    if output_dir:
        ensure_dir(output_dir)
        results_path = os.path.join(output_dir, 'rag_predictions.json')
        
        # Save without neighbor_infos to keep file size manageable
        results_summary = {
            'image_paths': image_paths,
            'true_labels': true_labels.tolist(),
            'predicted_probabilities': probabilities.tolist(),
            'k': classifier.k,
            'aggregation_method': classifier.aggregation
        }
        save_json(results_summary, results_path)
        
        # Also save as CSV for easy analysis
        df_results = pd.DataFrame({
            'image_path': image_paths,
            'true_label': true_labels,
            'predicted_probability': probabilities,
            'predicted_class': (probabilities >= 0.5).astype(int)
        })
        csv_path = os.path.join(output_dir, 'rag_predictions.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\n✓ Saved predictions to {output_dir}")
    
    print("="*60 + "\n")
    
    return results


def main():
    """Main RAG classification pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG-based glaucoma classification')
    parser.add_argument('--index', type=str, default='./rag_database/faiss_index.bin',
                       help='Path to FAISS index')
    parser.add_argument('--metadata', type=str, default='./rag_database/database_metadata.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--test-csv', type=str, default='./data/chaksu_test_labeled.csv',
                       help='Path to test CSV')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of neighbors to retrieve')
    parser.add_argument('--aggregation', type=str, default='weighted_vote',
                       choices=['majority_vote', 'weighted_vote', 'mean_prob'],
                       help='Label aggregation method')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check for Vast.ai environment
    if os.path.exists('/workspace'):
        args.index = '/workspace/rag_database/faiss_index.bin'
        args.metadata = '/workspace/rag_database/database_metadata.csv'
        args.test_csv = '/workspace/data/chaksu_test_labeled.csv'
        args.output_dir = '/workspace/rag_results'
    
    # Initialize classifier
    classifier = RAGClassifier(
        index_path=args.index,
        metadata_path=args.metadata,
        k=args.k,
        aggregation=args.aggregation
    )
    
    # Evaluate on test set
    results = evaluate_rag_classifier(
        classifier,
        test_csv=args.test_csv,
        output_dir=args.output_dir
    )
    
    print(f"\nClassified {len(results['true_labels'])} images")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
