"""
Data preparation for RAG-based glaucoma screening
Creates CSV files with image paths and labels for AIROGS and Chákṣu datasets
"""
import os
import pandas as pd
from pathlib import Path


def create_airogs_csvs(data_dir):
    """Create train and test CSVs for AIROGS dataset"""
    print("\n" + "="*60)
    print("Processing AIROGS Dataset")
    print("="*60)
    
    # Check if dataset is in subdirectory
    airogs_base = os.path.join(data_dir, 'AIROGS')
    if os.path.exists(os.path.join(airogs_base, 'eyepac-light-v2-512-jpg')):
        airogs_base = os.path.join(airogs_base, 'eyepac-light-v2-512-jpg')
    
    airogs_train_dir = os.path.join(airogs_base, 'train')
    airogs_test_dir = os.path.join(airogs_base, 'test')
    
    # Process training set
    train_data = []
    for label_dir in ['NRG', 'RG']:
        label = 0 if label_dir == 'NRG' else 1
        class_name = 'normal' if label == 0 else 'glaucoma'
        dir_path = os.path.join(airogs_train_dir, label_dir)
        
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images:
                train_data.append({
                    'image_path': os.path.join(dir_path, img),
                    'label': label,
                    'class': class_name,
                    'dataset': 'airogs'
                })
    
    df_train = pd.DataFrame(train_data)
    train_csv_path = os.path.join(data_dir, 'airogs_train.csv')
    df_train.to_csv(train_csv_path, index=False)
    print(f"✓ Created {train_csv_path}")
    print(f"  Total: {len(df_train)} images")
    print(f"  Normal: {len(df_train[df_train['label']==0])}")
    print(f"  Glaucoma: {len(df_train[df_train['label']==1])}")
    
    # Process test set
    test_data = []
    for label_dir in ['NRG', 'RG']:
        label = 0 if label_dir == 'NRG' else 1
        class_name = 'normal' if label == 0 else 'glaucoma'
        dir_path = os.path.join(airogs_test_dir, label_dir)
        
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images:
                test_data.append({
                    'image_path': os.path.join(dir_path, img),
                    'label': label,
                    'class': class_name,
                    'dataset': 'airogs'
                })
    
    df_test = pd.DataFrame(test_data)
    test_csv_path = os.path.join(data_dir, 'airogs_test.csv')
    df_test.to_csv(test_csv_path, index=False)
    print(f"✓ Created {test_csv_path}")
    print(f"  Total: {len(df_test)} images")
    print(f"  Normal: {len(df_test[df_test['label']==0])}")
    print(f"  Glaucoma: {len(df_test[df_test['label']==1])}")
    
    return train_csv_path, test_csv_path


def create_chakshu_csvs(data_dir):
    """Create train (labeled + unlabeled) and test CSVs for Chákṣu dataset"""
    print("\n" + "="*60)
    print("Processing Chákṣu Dataset")
    print("="*60)
    
    chakshu_dir = os.path.join(data_dir, 'CHAKSHU')
    
    # Process labeled training set
    train_labeled_data = []
    train_labeled_dir = os.path.join(chakshu_dir, 'train')
    for label_dir in ['NRG', 'RG']:
        label = 0 if label_dir == 'NRG' else 1
        class_name = 'normal' if label == 0 else 'glaucoma'
        dir_path = os.path.join(train_labeled_dir, label_dir)
        
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images:
                train_labeled_data.append({
                    'image_path': os.path.join(dir_path, img),
                    'label': label,
                    'class': class_name,
                    'dataset': 'chakshu',
                    'split': 'train_labeled'
                })
    
    df_train_labeled = pd.DataFrame(train_labeled_data)
    train_labeled_csv = os.path.join(data_dir, 'chaksu_train_labeled.csv')
    df_train_labeled.to_csv(train_labeled_csv, index=False)
    print(f"✓ Created {train_labeled_csv}")
    print(f"  Total: {len(df_train_labeled)} images")
    print(f"  Normal: {len(df_train_labeled[df_train_labeled['label']==0])}")
    print(f"  Glaucoma: {len(df_train_labeled[df_train_labeled['label']==1])}")
    
    # Process unlabeled training set
    train_unlabeled_data = []
    train_unlabeled_dir = os.path.join(chakshu_dir, 'train_unlabelled')
    
    if os.path.exists(train_unlabeled_dir):
        images = [f for f in os.listdir(train_unlabeled_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img in images:
            train_unlabeled_data.append({
                'image_path': os.path.join(train_unlabeled_dir, img),
                'label': -1,  # No label
                'class': 'unknown',
                'dataset': 'chakshu',
                'split': 'train_unlabeled'
            })
    
    df_train_unlabeled = pd.DataFrame(train_unlabeled_data)
    train_unlabeled_csv = os.path.join(data_dir, 'chaksu_train_unlabeled.csv')
    df_train_unlabeled.to_csv(train_unlabeled_csv, index=False)
    print(f"✓ Created {train_unlabeled_csv}")
    print(f"  Total: {len(df_train_unlabeled)} images (unlabeled)")
    
    # Process test set
    test_data = []
    test_dir = os.path.join(chakshu_dir, 'test')
    for label_dir in ['NRG', 'RG']:
        label = 0 if label_dir == 'NRG' else 1
        class_name = 'normal' if label == 0 else 'glaucoma'
        dir_path = os.path.join(test_dir, label_dir)
        
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images:
                test_data.append({
                    'image_path': os.path.join(dir_path, img),
                    'label': label,
                    'class': class_name,
                    'dataset': 'chakshu',
                    'split': 'test'
                })
    
    df_test = pd.DataFrame(test_data)
    test_csv_path = os.path.join(data_dir, 'chaksu_test_labeled.csv')
    df_test.to_csv(test_csv_path, index=False)
    print(f"✓ Created {test_csv_path}")
    print(f"  Total: {len(df_test)} images")
    print(f"  Normal: {len(df_test[df_test['label']==0])}")
    print(f"  Glaucoma: {len(df_test[df_test['label']==1])}")
    
    return train_labeled_csv, train_unlabeled_csv, test_csv_path


def main():
    """Main data preparation pipeline"""
    print("\n" + "="*60)
    print("RAG-Based Glaucoma Screening - Data Preparation")
    print("="*60)
    
    # Determine data directory
    if os.path.exists('/workspace/data'):
        data_dir = '/workspace/data'
        print("✓ Running on Vast.ai server")
    else:
        data_dir = './data'
        print("✓ Running locally")
    
    print(f"Data directory: {data_dir}")
    
    # Create CSVs for both datasets
    airogs_train_csv, airogs_test_csv = create_airogs_csvs(data_dir)
    chakshu_train_labeled_csv, chakshu_train_unlabeled_csv, chakshu_test_csv = create_chakshu_csvs(data_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Data Preparation Complete")
    print("="*60)
    print("CSV files created:")
    print(f"  1. {airogs_train_csv}")
    print(f"  2. {airogs_test_csv}")
    print(f"  3. {chakshu_train_labeled_csv}")
    print(f"  4. {chakshu_train_unlabeled_csv}")
    print(f"  5. {chakshu_test_csv}")
    print("\nRAG Database will be built from:")
    print("  - AIROGS train + test (all labeled)")
    print("  - Chákṣu train labeled")
    print("\nTesting will be performed on:")
    print("  - Chákṣu test labeled")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
