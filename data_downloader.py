import os
import requests
import zipfile
from tqdm import tqdm

def download_movielens_100k():
    """Download MovieLens 100K dataset"""
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "ml-100k.zip"
    extract_path = "."
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    print("Downloading MovieLens 100K dataset...")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    print("Extracting dataset...")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Move files to data directory
    if os.path.exists("ml-100k"):
        import shutil
        for file in os.listdir("ml-100k"):
            shutil.move(f"ml-100k/{file}", f"data/{file}")
        os.rmdir("ml-100k")
    
    # Clean up zip file
    os.remove(zip_path)
    
    print("Dataset downloaded and extracted successfully!")
    print("Files available in the 'data' directory:")

if __name__ == "__main__":
    download_movielens_100k() 