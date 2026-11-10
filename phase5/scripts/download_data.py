import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, login

def download_phase5_data():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))
    load_dotenv(dotenv_path=env_path)
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables. Please check your .env file.")
        sys.exit(1)
        
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)
    
    api = HfApi()
    print("Listing dataset repository files in Exploration-Lab/CS-781-Capstone...")
    repo_files = api.list_repo_files(repo_id="Exploration-Lab/CS-781-Capstone", repo_type="dataset", token=hf_token)
    phase5_files = [f for f in repo_files if "phase_5" in f.lower() or "phase5" in f.lower() or "lsi" in f.lower()]
    print(f"Discovered {len(phase5_files)} Phase 5 files in repo:")
    for f in phase5_files:
        print(f"  - {f}")
        
    if not phase5_files:
        print("\nAll files in repo:")
        for f in repo_files:
            print(f"  * {f}")
        return

    # Determine prefix
    first_file = phase5_files[0]
    prefix = first_file.split("/")[0] + "/*"
    print(f"\nDownloading pattern '{prefix}' from Hugging Face...")
    
    local_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    os.makedirs(local_data_dir, exist_ok=True)
    
    download_path = snapshot_download(
        repo_id="Exploration-Lab/CS-781-Capstone",
        repo_type="dataset",
        allow_patterns=prefix,
        token=hf_token,
        local_dir=local_data_dir
    )
    print(f"\nData successfully downloaded to: {download_path}")
    print("Downloaded files:")
    for root, dirs, files in os.walk(download_path):
        for file in files:
            p = os.path.join(root, file)
            print(f"  {p} ({os.path.getsize(p):,} bytes)")

if __name__ == "__main__":
    download_phase5_data()
