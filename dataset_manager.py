"""Dataset management and automatic downloading.

Automatically fetches AIROGS (Kaggle), Chákṣu (Figshare), and handles
file extraction for the Netra-Adapt project.
"""

import os
import requests
import zipfile
import opendatasets as od
import pandas as pd
from tqdm import tqdm
from config import cfg


class DatasetManager:
    """Manages dataset downloading and setup for AIROGS and Chákṣu."""
    def __init__(self):
        self.airogs_light_path = os.path.join(cfg.DATA_ROOT, "eyepacs-airogs-light-v2")
        self.airogs_full_path = os.path.join(cfg.DATA_ROOT, "airogs-full")
        self.chakshu_path = os.path.join(cfg.DATA_ROOT, "chakshu")

    def _download_file(self, url, dest_path):
        """Download with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

    def _unzip_file(self, zip_path, extract_to):
        print(f"📦 Unzipping {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    def setup_airogs_light(self):
        if os.path.exists(self.airogs_light_path):
            print("✅ AIROGS Light v2 found.")
            return

        print("⬇️ Downloading AIROGS Light v2 (Kaggle)...")
        # Ensure kaggle.json is in ~/.kaggle/
        try:
            od.download("https://www.kaggle.com/datasets/deathtrooper/eyepacs-airogs-light-v2", data_dir=cfg.DATA_ROOT)
        except Exception as e:
            print(f"❌ Kaggle Download Failed: {e}")
            print("Please ensure kaggle.json is configured.")

    def setup_chakshu(self):
        if os.path.exists(self.chakshu_path):
            print("✅ Chákṣu dataset found.")
            return
        
        print("⬇️ Downloading Chákṣu (Figshare)...")
        os.makedirs(self.chakshu_path, exist_ok=True)
        # Direct link to Chákṣu Version 2
        url = "https://figshare.com/ndownloader/articles/20123135/versions/2"
        zip_dest = os.path.join(cfg.DATA_ROOT, "chakshu.zip")
        
        self._download_file(url, zip_dest)
        self._unzip_file(zip_dest, self.chakshu_path)
        os.remove(zip_dest)
        print("✅ Chákṣu Setup Complete.")

    def get_dfs(self):
        # 1. Load Chákṣu
        chakshu_csv = None
        for root, dirs, files in os.walk(self.chakshu_path):
            if "Glaucoma_Decision_Comparison_Remedio_majority.csv" in files:
                chakshu_csv = os.path.join(root, "Glaucoma_Decision_Comparison_Remedio_majority.csv")
                break
        
        if chakshu_csv:
            df_chakshu = pd.read_csv(chakshu_csv)
            # Create standardized columns
            # Chákṣu CSV 'Image' column usually lacks extension or full path
            df_chakshu['path'] = df_chakshu['Image']
            df_chakshu = df_chakshu.rename(columns={'Glaucoma Decision': 'label'})
        else:
             print("⚠️ Chákṣu CSV not found automatically. Using placeholder.")
             df_chakshu = pd.DataFrame(columns=['path', 'label'])

        # 2. Load AIROGS
        if cfg.DATASET_MODE == 'full':
            # Simplified logic for Full AIROGS
            pass 
        else:
            # Light V2
            airogs_csv = os.path.join(self.airogs_light_path, "train_labels.csv")
            if not os.path.exists(airogs_csv):
                 for root, _, files in os.walk(self.airogs_light_path):
                     for f in files:
                         if f.endswith(".csv") and "train" in f:
                             airogs_csv = os.path.join(root, f)
            
            df_airogs = pd.read_csv(airogs_csv)
            if 'filename' in df_airogs.columns:
                df_airogs = df_airogs.rename(columns={'filename': 'path', 'class': 'label'})

        return df_airogs, df_chakshu