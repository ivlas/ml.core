import os
import requests

def download_datasets(datasets_dict, data_dir="data/"):
    """
    Download datasets from provided URLs and save them to the specified directory.
    
    Parameters:
    datasets_dict (dict): Dictionary with filename as key and URL as value
    data_dir (str): Directory to save the downloaded files, defaults to "data/"
    
    Returns:
    list: List of paths to successfully downloaded files
    """
    os.makedirs(data_dir, exist_ok=True)
    
    downloaded_files = []
    
    for filename, url in datasets_dict.items():
        file_path = os.path.join(data_dir, filename)
        
        try:
            print(f"Downloading {filename} from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded {filename} to {file_path}")
            downloaded_files.append(file_path)
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    return downloaded_files

if __name__ == "__main__":
    datasets = {
        "names.txt" : "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt",
        "shakespeare.txt" : "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt",
    }
    
    download_datasets(datasets)