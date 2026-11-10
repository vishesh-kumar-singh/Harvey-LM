import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login

def download_phase3_data():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))
    load_dotenv(dotenv_path=env_path)
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables. Please check your .env file.")
        return
        
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)
    
    print("Downloading Phase 3 data from Hugging Face...")
    local_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    os.makedirs(local_data_dir, exist_ok=True)
    
    download_path = snapshot_download(
        repo_id="Exploration-Lab/CS-781-Capstone",
        repo_type="dataset",
        allow_patterns="Phase_3_RR/*",
        token=hf_token,
        local_dir=local_data_dir
    )
    print(f"\nData successfully downloaded to: {download_path}")
    print("Downloaded files:")
    for root, dirs, files in os.walk(download_path):
        for file in files:
            p = os.path.join(root, file)
            print(f"  {p} ({os.path.getsize(p)} bytes)")

if __name__ == "__main__":
    download_phase3_data()
