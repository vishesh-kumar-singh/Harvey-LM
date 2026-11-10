#!/usr/bin/env python3
"""
Download Phase 6 Abstractive Legal Summarization dataset from Hugging Face Hub.
Dataset repo: Exploration-Lab/CS-781-Capstone
Target path: Phase_6_Summ/ -> code/phase6/data/raw/Phase_6_Summ
"""

import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, login

def download_phase6_data():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))
    load_dotenv(dotenv_path=env_path)
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables. Please check your .env file.")
        sys.exit(1)
        
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)
    
    api = HfApi()
    print("Connecting to Hugging Face: Exploration-Lab/CS-781-Capstone...")
    repo_files = api.list_repo_files(repo_id="Exploration-Lab/CS-781-Capstone", repo_type="dataset", token=hf_token)
    phase6_files = [f for f in repo_files if "phase_6" in f.lower() or "phase6" in f.lower() or "summ" in f.lower()]
    print(f"Discovered {len(phase6_files)} Phase 6 files in repository:")
    for f in phase6_files:
        print(f"  - {f}")
        
    if not phase6_files:
        print("Error: No Phase 6 files found!")
        return

    prefix = "Phase_6_Summ/*"
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
    print("\nVerified Downloaded Files:")
    for root, dirs, files in os.walk(os.path.join(local_data_dir, "Phase_6_Summ")):
        for file in files:
            p = os.path.join(root, file)
            print(f"  {p} ({os.path.getsize(p):,} bytes)")

if __name__ == "__main__":
    download_phase6_data()
