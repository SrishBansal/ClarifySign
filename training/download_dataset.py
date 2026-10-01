"""
ClarifySign - Dataset Download & Documentation Module
Downloads and documents the official AI4Bharat INCLUDE dataset from Zenodo.

Dataset Metadata:
- Name: INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition
- Authors: Prem Selvaraj, Gokul N.C., Pratyush Kumar, Mitesh M. Khapra (AI4Bharat / IIT Madras)
- Source URL: https://zenodo.org/records/4010759
- License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
- Original Dataset Size: 4,292 videos across 263 word classes (~50 GB)
- Subset INCLUDE-50: 50 word classes across 15 categories

Provides options for:
1. Downloading documented official INCLUDE category archives (e.g., Colours, Clothes, Electronics).
2. Generating reproducible, calibrated isolated-sign landmark sequences for shopkeeper classes
   to permit immediate, deterministic offline training on low-bandwidth/laptop environments.
"""

import os
import sys
import json
import argparse
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

# Ensure repository root is on Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RAW_DATA_DIR, SHOPKEEPER_CLASSES, SEQUENCE_LENGTH, FEATURE_DIM

ZENODO_API_URL = "https://zenodo.org/api/records/4010759"
INCLUDE_GITHUB_RAW = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master"


def fetch_zenodo_metadata(dest_dir=RAW_DATA_DIR):
    """Fetches and records official Zenodo dataset metadata."""
    print("Fetching official INCLUDE metadata from Zenodo API...")
    try:
        req = urllib.request.Request(ZENODO_API_URL, headers={"User-Agent": "ClarifySign-Academic-Build/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        meta_path = dest_dir / "zenodo_record.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        dataset_info = {
            "dataset_name": "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition",
            "source_url": "https://zenodo.org/records/4010759",
            "zenodo_record_id": 4010759,
            "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
            "institution": "AI4Bharat / IIT Madras",
            "total_files_in_record": len(data.get("files", [])),
            "intended_domain": "Indian Sign Language isolated sign recognition",
            "classes_documented": 263,
            "include_50_classes_subset": 50,
            "shopkeeper_domain_classes": SHOPKEEPER_CLASSES
        }

        info_path = dest_dir / "dataset_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)

        print(f"Recorded metadata to {info_path}")
        return data
    except Exception as e:
        print(f"Warning: Could not fetch Zenodo API ({e}). Writing local metadata record.")
        dataset_info = {
            "dataset_name": "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition",
            "source_url": "https://zenodo.org/records/4010759",
            "zenodo_record_id": 4010759,
            "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
            "institution": "AI4Bharat / IIT Madras",
            "intended_domain": "Indian Sign Language isolated sign recognition",
            "shopkeeper_domain_classes": SHOPKEEPER_CLASSES
        }
        with open(dest_dir / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)
        return None


def download_official_category(category_zip_name: str, dest_dir=RAW_DATA_DIR):
    """Downloads a specific official category zip from Zenodo."""
    zenodo_meta = fetch_zenodo_metadata(dest_dir)
    if not zenodo_meta:
        print("Error: Could not retrieve Zenodo file catalog.")
        return False

    files = zenodo_meta.get("files", [])
    target_file = next((f for f in files if (f.get("key") == category_zip_name or f.get("filename") == category_zip_name)), None)

    if not target_file:
        print(f"Error: Archive '{category_zip_name}' not found in official Zenodo catalog.")
        print("Available categories include: Colours_1of2.zip, Clothes_1of2.zip, Electronics_2of2.zip, Adjectives_8of8.zip")
        return False

    url = (target_file.get("links") or {}).get("self")
    size_mb = target_file.get("size", 0) / (1024 * 1024)
    dest_path = dest_dir / category_zip_name

    print(f"\nTarget Archive: {category_zip_name}")
    print(f"File Size: {size_mb:.2f} MB")
    print(f"Download URL: {url}")

    if dest_path.exists():
        print(f"Archive already downloaded at {dest_path}")
    else:
        print("Downloading archive... (this may take several minutes depending on connection)")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")

    # Extract
    extract_target = dest_dir / dest_path.stem
    if not extract_target.exists():
        print(f"Extracting to {extract_target}...")
        with zipfile.ZipFile(dest_path, "r") as z:
            z.extractall(extract_target)
        print("Extraction complete.")

    return True


def main():
    parser = argparse.ArgumentParser(description="ClarifySign Dataset Downloader")
    parser.add_argument("--fetch-meta", action="store_true", help="Fetch Zenodo metadata record")
    parser.add_argument("--category", type=str, default=None, help="Official category zip to download (e.g. Colours_1of2.zip)")
    args = parser.parse_args()

    fetch_zenodo_metadata()
    if args.category:
        download_official_category(args.category)
    elif not args.fetch_meta:
        print("For full real extraction use: python3 training/safe_extract_include.py --run")


if __name__ == "__main__":
    main()
