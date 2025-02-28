import kagglehub

def download_dataset():
    # Download the latest version of the dataset
    path = kagglehub.dataset_download("kishanyadav/inshort-news")
    return path

if __name__ == "__main__":
    dataset_path = download_dataset()
    print("Path to dataset files:", dataset_path)