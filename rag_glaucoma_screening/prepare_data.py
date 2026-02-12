"""
prepare_data.py - Data Preparation for RAG-based Glaucoma Screening
Adapted from proven Netra-Adapt code

Handles:
1. AIROGS (Kaggle) - eyepac-light-v2-512-jpg folder with RG/NRG structure
2. Chákṣu (Figshare) - Complex nested structure with Train/Test splits
   - 1.0_Original_Fundus_Images/[Bosch|Forus|Remidio]
   - 6.0_Glaucoma_Decision/Glaucoma_decision_comparision/*_majority.csv
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def find_folder(base_path, folder_name, recursive=True):
    """Find a folder anywhere within base_path."""
    if recursive:
        for root, dirs, files in os.walk(base_path):
            if folder_name in dirs:
                return os.path.join(root, folder_name)
    direct = os.path.join(base_path, folder_name)
    if os.path.exists(direct):
        return direct
    return None


def prepare_airogs(data_dir, airogs_dir, csv_out_dir):
    """Process AIROGS dataset with RG/NRG folder structure.
    
    Creates TWO CSVs:
    - airogs_train.csv: Training set (80%)
    - airogs_test.csv: Test set (20%)
    """
    print(f"--- Processing AIROGS ---")
    records = []
    
    # Find RG and NRG folders (may be nested after unzip)
    rg_dir = find_folder(airogs_dir, "RG")
    nrg_dir = find_folder(airogs_dir, "NRG")
    
    if rg_dir:
        rg_files = glob.glob(os.path.join(rg_dir, "*.jpg"))
        for f in rg_files:
            records.append({"path": f, "label": 1})
        print(f"  Found {len(rg_files)} RG (glaucoma) images")
    
    if nrg_dir:
        nrg_files = glob.glob(os.path.join(nrg_dir, "*.jpg"))
        for f in nrg_files:
            records.append({"path": f, "label": 0})
        print(f"  Found {len(nrg_files)} NRG (normal) images")
        
    if records:
        df = pd.DataFrame(records)
        
        # Shuffle and split 80-20
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        
        df_train = df[:split_idx]
        df_test = df[split_idx:]
        
        # Save train
        train_path = os.path.join(csv_out_dir, "airogs_train.csv")
        df_train.to_csv(train_path, index=False)
        print(f"  ✓ Saved {train_path} ({len(df_train)} images)")
        
        # Save test
        test_path = os.path.join(csv_out_dir, "airogs_test.csv")
        df_test.to_csv(test_path, index=False)
        print(f"  ✓ Saved {test_path} ({len(df_test)} images)")
    else:
        print("[ERROR] No AIROGS images found!")
        print(f"  Expected structure: {airogs_dir}/RG/*.jpg and {airogs_dir}/NRG/*.jpg")


def parse_chaksu_labels(data_dir, chaksu_dir, csv_out_dir):
    """
    Process Chákṣu dataset with complex nested structure.
    
    Figshare download structure:
    CHAKSHU/
    ├── Train/
    │   ├── 1.0_Original_Fundus_Images/
    │   │   ├── Bosch/
    │   │   ├── Forus/
    │   │   └── Remidio/
    │   └── 6.0_Glaucoma_Decision/
    │       └── Glaucoma_decision_comparision/
    │           └── *_majority.csv
    └── Test/
        └── (same structure)
    """
    print(f"\n--- Processing Chákṣu ---")
    
    # Dictionary to store filename -> label mapping
    label_map = {}
    
    # Step 1: Find all majority decision CSV files (in Train and Test)
    csv_patterns = [
        os.path.join(chaksu_dir, "**", "6.0_Glaucoma_Decision", "**", "*majority*.csv"),
        os.path.join(chaksu_dir, "**", "Glaucoma_decision*", "*majority*.csv"),
        os.path.join(chaksu_dir, "6.0_Glaucoma_Decision", "**", "*majority*.csv"),
    ]
    
    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))
    csv_files = list(set(csv_files))  # Remove duplicates
    
    print(f"  Found {len(csv_files)} label CSV files")
    
    # Step 2: Parse each CSV to build label_map
    for csv_file in csv_files:
        try:
            # Try different encodings and delimiters
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except:
                df = pd.read_csv(csv_file, encoding='latin-1')
            
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            
            # Find the image and decision columns
            img_col = None
            dec_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'image' in col_lower:
                    img_col = col
                if 'majority' in col_lower or 'decision' in col_lower:
                    dec_col = col
            
            if img_col is None or dec_col is None:
                print(f"    [SKIP] {os.path.basename(csv_file)} - columns not found")
                continue
            
            # Parse rows
            for idx, row in df.iterrows():
                raw_name = str(row[img_col]).strip()
                decision = str(row[dec_col]).upper().strip()
                
                # Clean filename: handle formats like "Image101.jpg-Image101-1.jpg"
                # Extract the core image name
                parts = raw_name.replace('\\', '/').split('/')
                fname = parts[-1]  # Get last part if path
                
                # Handle hyphenated names
                if '-' in fname and fname.count('-') > 0:
                    # Take first part: "Image101.jpg" from "Image101.jpg-Image101-1.jpg"
                    fname = fname.split('-')[0]
                
                # Ensure extension
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fname += ".jpg"
                
                # Normalize case for matching
                fname_lower = fname.lower()
                
                # Map decision to label
                if "NORMAL" in decision:
                    label = 0
                elif "GLAUCOMA" in decision or "SUSPECT" in decision:
                    label = 1
                else:
                    continue  # Skip unclear labels
                
                # Store with lowercase key for case-insensitive matching
                label_map[fname_lower] = label
                label_map[fname] = label  # Also store original case
                
            print(f"    [OK] {os.path.basename(csv_file)} - {len(df)} entries")
                
        except Exception as e:
            print(f"    [ERROR] {os.path.basename(csv_file)}: {e}")
    
    print(f"  Total labels parsed: {len(label_map)}")
    
    # Step 3: Find all images (in Train and Test, under 1.0_Original_Fundus_Images)
    image_patterns = [
        os.path.join(chaksu_dir, "**", "1.0_Original*", "Bosch", "*"),
        os.path.join(chaksu_dir, "**", "1.0_Original*", "Forus", "*"),
        os.path.join(chaksu_dir, "**", "1.0_Original*", "Remidio", "*"),
        os.path.join(chaksu_dir, "Bosch", "*"),
        os.path.join(chaksu_dir, "Forus", "*"),
        os.path.join(chaksu_dir, "Remidio", "*"),
    ]
    
    all_images = []
    for pattern in image_patterns:
        found = glob.glob(pattern, recursive=True)
        # Filter to actual image files
        found = [f for f in found if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(f)]
        all_images.extend(found)
    all_images = list(set(all_images))  # Remove duplicates
    
    print(f"  Found {len(all_images)} image files")
    
    # Step 4: Match images with labels
    labeled_records = []
    unlabeled_records = []
    
    for img_path in all_images:
        fname = os.path.basename(img_path)
        fname_lower = fname.lower()
        
        # Try exact match (case-insensitive)
        if fname_lower in label_map:
            labeled_records.append({"path": img_path, "label": label_map[fname_lower]})
        elif fname in label_map:
            labeled_records.append({"path": img_path, "label": label_map[fname]})
        else:
            # Try fuzzy match - find any key that contains this filename or vice versa
            matched = False
            for key, lbl in label_map.items():
                if key.lower() in fname_lower or fname_lower in key.lower():
                    labeled_records.append({"path": img_path, "label": lbl})
                    matched = True
                    break
            if not matched:
                unlabeled_records.append({"path": img_path, "label": -1})
    
    print(f"  Matched: {len(labeled_records)} labeled, {len(unlabeled_records)} unlabeled")
    
    # Step 5: Separate Train vs Test based on folder structure
    train_labeled = []
    test_labeled = []
    train_all = []
    
    for rec in labeled_records:
        if '/Train/' in rec['path'] or '\\Train\\' in rec['path']:
            train_labeled.append(rec)
            train_all.append(rec)
        elif '/Test/' in rec['path'] or '\\Test\\' in rec['path']:
            test_labeled.append(rec)
        else:
            # If no Train/Test in path, add to train by default
            train_labeled.append(rec)
            train_all.append(rec)
    
    for rec in unlabeled_records:
        if '/Train/' in rec['path'] or '\\Train\\' in rec['path']:
            train_all.append(rec)
    
    print(f"  Train split: {len(train_labeled)} labeled, {len(train_all)} total")
    print(f"  Test split: {len(test_labeled)} labeled")
    
    # Step 6: Save CSVs
    if train_labeled:
        df_train = pd.DataFrame(train_labeled)
        n_glaucoma = sum(1 for r in train_labeled if r['label'] == 1)
        n_normal = sum(1 for r in train_labeled if r['label'] == 0)
        print(f"  Train class distribution: Normal={n_normal}, Glaucoma={n_glaucoma}")
        
        out_path = os.path.join(csv_out_dir, "chaksu_train_labeled.csv")
        df_train.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df_train)} images)")
    
    if test_labeled:
        df_test = pd.DataFrame(test_labeled)
        n_glaucoma = sum(1 for r in test_labeled if r['label'] == 1)
        n_normal = sum(1 for r in test_labeled if r['label'] == 0)
        print(f"  Test class distribution: Normal={n_normal}, Glaucoma={n_glaucoma}")
        
        out_path = os.path.join(csv_out_dir, "chaksu_test_labeled.csv")
        df_test.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df_test)} images) - FOR RAG EVALUATION")
        
    # Save ALL training images (labels ignored for unlabeled set)
    if train_all:
        df_all = pd.DataFrame(train_all)
        df_all['label'] = -1  # Force unlabeled
        out_path = os.path.join(csv_out_dir, "chaksu_train_unlabeled.csv")
        df_all.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df_all)} images)")


def validate_data(csv_out_dir):
    """Quick validation of prepared data."""
    print(f"\n--- Validation ---")
    
    expected_csvs = [
        "airogs_train.csv",
        "airogs_test.csv",
        "chaksu_train_labeled.csv",
        "chaksu_test_labeled.csv",
        "chaksu_train_unlabeled.csv"
    ]
    
    for csv_name in expected_csvs:
        csv_path = os.path.join(csv_out_dir, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Validate that paths exist
            valid = sum(1 for p in df['path'] if os.path.exists(p))
            print(f"  {csv_name}: {len(df)} entries, {valid} valid paths")
        else:
            print(f"  {csv_name}: NOT FOUND")


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
    
    # Set up paths
    airogs_dir = os.path.join(data_dir, "AIROGS")
    chaksu_dir = os.path.join(data_dir, "CHAKSHU")
    csv_out_dir = data_dir  # Save CSVs directly in data directory
    
    # Process datasets
    prepare_airogs(data_dir, airogs_dir, csv_out_dir)
    parse_chaksu_labels(data_dir, chaksu_dir, csv_out_dir)
    validate_data(csv_out_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Data Preparation Complete")
    print("="*60)
    print("CSV files created in:", csv_out_dir)
    print("\nRAG Database will be built from:")
    print("  - AIROGS train + test (pure Western data)")
    print("\nRAG Evaluation will use:")
    print("  - Chákṣu test set (pure zero-shot cross-ethnic test)")


if __name__ == "__main__":
    main()
