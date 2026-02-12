"""
Build RAG database by extracting features from all images using DINOv3
Features are stored in a FAISS index for efficient similarity search
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import faiss
from transformers import AutoModel, AutoImageProcessor
from utils import ensure_dir, save_json, get_project_root


class FeatureExtractor:
    """Extract features from images using DINOv3"""
    
    def __init__(self, model_name='facebook/dinov3-vitl16-pretrain-lvd1689m', device=None):
        print(f"Loading model: {model_name}")
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as image representation
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return features.squeeze()
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_batch_features(self, image_paths, batch_size=32):
        """Extract features from a batch of images"""
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), 
                      desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            # Load images in batch
            for idx, img_path in enumerate(batch_paths):
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(image)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    all_features.append(None)
            
            if not batch_images:
                continue
            
            # Process batch
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Add features in correct positions
                for idx, feat in zip(valid_indices, features):
                    all_features.append(feat)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                for _ in valid_indices:
                    all_features.append(None)
        
        return all_features


def build_database(csv_files, output_dir, batch_size=32):
    """
    Build RAG database from multiple CSV files
    
    Args:
        csv_files: List of CSV file paths containing path and label columns
        output_dir: Directory to save the database
        batch_size: Batch size for feature extraction
    """
    print("\n" + "="*60)
    print("Building RAG Database")
    print("="*60)
    
    ensure_dir(output_dir)
    
    # Load all CSVs
    all_data = []
    for csv_file in csv_files:
        print(f"Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        # Only include labeled data in database
        df_labeled = df[df['label'] != -1]
        all_data.append(df_labeled)
        print(f"  Loaded {len(df_labeled)} labeled images")
    
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal images in database: {len(df_combined)}")
    print(f"  Normal: {len(df_combined[df_combined['label']==0])}")
    print(f"  Glaucoma: {len(df_combined[df_combined['label']==1])}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    print("\nExtracting features from all images...")
    image_paths = df_combined['path'].tolist()
    features = extractor.extract_batch_features(image_paths, batch_size=batch_size)
    
    # Filter out failed extractions
    valid_features = []
    valid_indices = []
    for idx, feat in enumerate(features):
        if feat is not None:
            valid_features.append(feat)
            valid_indices.append(idx)
    
    features_array = np.array(valid_features).astype('float32')
    df_valid = df_combined.iloc[valid_indices].reset_index(drop=True)
    
    print(f"\n✓ Successfully extracted features from {len(valid_features)}/{len(df_combined)} images")
    print(f"Feature dimension: {features_array.shape[1]}")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = features_array.shape[1]
    
    # Use L2 distance (can also try Inner Product for cosine similarity)
    index = faiss.IndexFlatL2(dimension)
    index.add(features_array)
    
    print(f"✓ FAISS index built with {index.ntotal} vectors")
    
    # Save everything
    index_path = os.path.join(output_dir, 'faiss_index.bin')
    faiss.write_index(index, index_path)
    print(f"✓ Saved FAISS index to {index_path}")
    
    metadata_path = os.path.join(output_dir, 'database_metadata.csv')
    df_valid.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata to {metadata_path}")
    
    # Save database statistics
    stats = {
        'total_images': int(len(df_valid)),
        'normal_count': int(len(df_valid[df_valid['label']==0])),
        'glaucoma_count': int(len(df_valid[df_valid['label']==1])),
        'feature_dimension': int(dimension),
        'model_used': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'datasets_included': df_valid['dataset'].unique().tolist() if 'dataset' in df_valid.columns else []
    }
    
    stats_path = os.path.join(output_dir, 'database_stats.json')
    save_json(stats, stats_path)
    
    print("\n" + "="*60)
    print("Database Build Complete")
    print("="*60)
    print(f"Files saved in: {output_dir}")
    print(f"  - faiss_index.bin (feature index)")
    print(f"  - database_metadata.csv (image metadata)")
    print(f"  - database_stats.json (database statistics)")
    print("="*60 + "\n")
    
    return index_path, metadata_path, stats_path


def main():
    """Main database building pipeline"""
    # Determine data directory
    if os.path.exists('/workspace/data'):
        data_dir = '/workspace/data'
        output_dir = '/workspace/rag_database'
    else:
        data_dir = './data'
        output_dir = './rag_database'
    
    # CSV files to include in database (AIROGS ONLY - pure zero-shot test)
    csv_files = [
        os.path.join(data_dir, 'airogs_train.csv'),
        os.path.join(data_dir, 'airogs_test.csv'),
    ]
    
    # Verify all files exist
    missing_files = [f for f in csv_files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: Missing CSV files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run prepare_data.py first!")
        sys.exit(1)
    
    # Build database
    build_database(csv_files, output_dir, batch_size=32)


if __name__ == '__main__':
    main()
