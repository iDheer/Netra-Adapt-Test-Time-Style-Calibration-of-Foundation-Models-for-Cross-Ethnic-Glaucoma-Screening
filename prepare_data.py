"""
prepare_data.py - Intelligent Data Preparation for Netra-Adapt

Handles:
1. AIROGS (Kaggle) - Simple RG/NRG folder structure
2. Chákṣu (Figshare) - Complex nested structure with Train/Test splits
   - 1.0_Original_Fundus_Images/[Bosch|Forus|Remidio]
   - 6.0_Glaucoma_Decision/Glaucoma_decision_comparision/*_majority.csv
   
The Figshare download creates a nested structure that we handle automatically.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
BASE_DIR = "/workspace/Netra_Adapt/data"
AIROGS_DIR = os.path.join(BASE_DIR, "raw_airogs")
CHAKSU_DIR = os.path.join(BASE_DIR, "raw_chaksu")
CSV_OUT_DIR = os.path.join(BASE_DIR, "processed_csvs")


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


def prepare_airogs():
    """Process AIROGS dataset with RG/NRG folder structure."""
    print(f"--- Processing AIROGS ---")
    records = []
    
    # Find RG and NRG folders (may be nested after unzip)
    rg_dir = find_folder(AIROGS_DIR, "RG")
    nrg_dir = find_folder(AIROGS_DIR, "NRG")
    
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
        out_path = os.path.join(CSV_OUT_DIR, "airogs_train.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df)} images)")
    else:
        print("[ERROR] No AIROGS images found!")
        print(f"  Expected structure: {AIROGS_DIR}/RG/*.jpg and {AIROGS_DIR}/NRG/*.jpg")


def parse_chaksu_labels():
    """
    Process Chákṣu dataset with complex nested structure.
    
    Figshare download structure:
    raw_chaksu/
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
        os.path.join(CHAKSU_DIR, "**", "6.0_Glaucoma_Decision", "**", "*majority*.csv"),
        os.path.join(CHAKSU_DIR, "**", "Glaucoma_decision*", "*majority*.csv"),
        os.path.join(CHAKSU_DIR, "6.0_Glaucoma_Decision", "**", "*majority*.csv"),
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
        os.path.join(CHAKSU_DIR, "**", "1.0_Original*", "Bosch", "*"),
        os.path.join(CHAKSU_DIR, "**", "1.0_Original*", "Forus", "*"),
        os.path.join(CHAKSU_DIR, "**", "1.0_Original*", "Remidio", "*"),
        os.path.join(CHAKSU_DIR, "Bosch", "*"),
        os.path.join(CHAKSU_DIR, "Forus", "*"),
        os.path.join(CHAKSU_DIR, "Remidio", "*"),
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
    
    # Step 5: Save CSVs
    if labeled_records:
        df_lab = pd.DataFrame(labeled_records)
        
        # Count class distribution
        n_glaucoma = sum(1 for r in labeled_records if r['label'] == 1)
        n_normal = sum(1 for r in labeled_records if r['label'] == 0)
        print(f"  Class distribution: Normal={n_normal}, Glaucoma={n_glaucoma}")
        
        out_path = os.path.join(CSV_OUT_DIR, "chaksu_labeled.csv")
        df_lab.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df_lab)} images) - FOR PHASE B (ORACLE) & EVALUATION")
        
    # For Phase C (Adaptation), we use ALL images (labels ignored by adaptation algorithm)
    if labeled_records or unlabeled_records:
        all_recs = labeled_records + unlabeled_records
        df_all = pd.DataFrame(all_recs)
        df_all['label'] = -1  # Force unlabeled (SFDA doesn't use labels)
        out_path = os.path.join(CSV_OUT_DIR, "chaksu_unlabeled.csv")
        df_all.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df_all)} images) - FOR PHASE C (ADAPT)")


def validate_data():
    """Quick validation of prepared data."""
    print(f"\n--- Validation ---")
    
    for csv_name in ["airogs_train.csv", "chaksu_labeled.csv", "chaksu_unlabeled.csv"]:
        csv_path = os.path.join(CSV_OUT_DIR, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Validate that paths exist
            valid = sum(1 for p in df['path'] if os.path.exists(p))
            print(f"  {csv_name}: {len(df)} entries, {valid} valid paths")
        else:
            print(f"  {csv_name}: NOT FOUND")


if __name__ == "__main__":
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    prepare_airogs()
    parse_chaksu_labels()
    validate_data()