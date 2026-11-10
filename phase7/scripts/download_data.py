#!/usr/bin/env python3
"""
Download Phase 7 Prior Case Retrieval (PCR) dataset from Hugging Face Hub.
Dataset repo: Exploration-Lab/CS-781-Capstone
Target path: Phase_7_PCR/ -> code/phase7/data/raw/Phase_7_PCR
"""

import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, login


def download_phase7_data():
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
    phase7_files = [f for f in repo_files if "phase_7" in f.lower() or "phase7" in f.lower() or "pcr" in f.lower()]
    print(f"Discovered {len(phase7_files)} Phase 7 files in repository:")
    for f in phase7_files:
        print(f"  - {f}")

    if not phase7_files:
        print("Error: No Phase 7 files found!")
        return

    prefix = "Phase_7_PCR/*"
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
    for root, dirs, files in os.walk(os.path.join(local_data_dir, "Phase_7_PCR")):
        for file in files:
            p = os.path.join(root, file)
            print(f"  {p} ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    download_phase7_data()
