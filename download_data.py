import os
import requests
import tarfile
import pandas as pd
from tqdm import tqdm

def download_dataset(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

def extract_and_load(tar_path, extract_dir):
    if not os.path.exists(extract_dir):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(extract_dir))
        print("Extraction complete.")
    
    data = []
    for label in ['pos', 'neg']:
        dir_name = os.path.join(extract_dir, label)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    data.append({'review': f.read(), 'sentiment': 1 if label == 'pos' else 0})
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    TAR_PATH = "aclImdb_v1.tar.gz"
    EXTRACT_DIR = "aclImdb/train" # Focus on training data first
    OUTPUT_CSV = "imdb_reviews.csv"

    if not os.path.exists(OUTPUT_CSV):
        download_dataset(DATA_URL, TAR_PATH)
        # Extract and load both train and test to get the full 50k
        print("Processing training data...")
        df_train = extract_and_load(TAR_PATH, "aclImdb/train")
        print("Processing test data...")
        df_test = extract_and_load(TAR_PATH, "aclImdb/test")
        
        df_full = pd.concat([df_train, df_test], ignore_index=True)
        df_full.to_csv(OUTPUT_CSV, index=False)
        print(f"Dataset saved to {OUTPUT_CSV}. Total reviews: {len(df_full)}")
    else:
        print(f"{OUTPUT_CSV} already exists.")
