"""
ClarifySign - Single Entry Point Interactive Launcher
Provides complete environment check, dataset downloader, data preparation,
model training, web application launch, benchmark evaluation, and test suite execution.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def print_banner():
    print(r"""
========================================================================
   ___ _            _  __        ____  _             
  / __| |__ _ _ _ (_)/ _|_  _  / ___|(_)__ _ _ _    
 | (__| / _` | '_|| |  _| || | \___ \| / _` | ' \   
  \___|_\__,_|_|  |_|_|  \_, | |____/|_\__, |_||_|  
                         |__/          |___/        
 Ambiguity-Aware Indian Sign Language Communication System
========================================================================
    """)


def check_environment():
    """Performs first-run hardware, OS, camera, and dependency checks."""
    print("\n--- 🔍 Checking Environment & Hardware ---")
    
    # 1. Python Version
    py_ver = sys.version.split()[0]
    print(f"• Python Version:      {py_ver} (Requirement: Python 3.10+)")
    
    # 2. Disk Space
    total, used, free = shutil.disk_usage(ROOT_DIR)
    free_gb = free / (1024 ** 3)
    print(f"• Free Disk Space:     {free_gb:.2f} GB")
    
    # 3. RAM Estimation
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
            print(f"• System Memory (RAM): {ram_gb:.1f} GB")
    except Exception:
        pass
        
    # 4. Neural Hardware Acceleration
    try:
        import torch
        if torch.backends.mps.is_available():
            accel = "Apple Silicon GPU Acceleration (MPS active)"
        elif torch.cuda.is_available():
            accel = f"NVIDIA GPU Acceleration ({torch.cuda.get_device_name(0)})"
        else:
            accel = "Host CPU (Standard)"
        print(f"• Neural Accelerator:  {accel}")
    except Exception:
        print("• PyTorch:             Not imported")
        
    # 5. Camera Availability
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        has_cam = cap.isOpened()
        cap.release()
        cam_status = "Available (Webcam ready)" if has_cam else "Not detected or permission restricted (Browser camera & Scenario mode available)"
        print(f"• Local Video Camera:  {cam_status}")
    except Exception:
        print("• OpenCV Camera Check: Skipped")
        
    # 6. Model Status
    model_file = ROOT_DIR / "models" / "isl_bilstm.pt"
    labels_file = ROOT_DIR / "models" / "labels.json"
    if model_file.exists() and labels_file.exists():
        print(f"• Trained ISL Model:   READY ({model_file})")
    else:
        print(f"• Trained ISL Model:   Not yet trained. (Select Option 4 to train)")
        
    print("------------------------------------------\n")


def run_command(cmd_args, cwd=ROOT_DIR):
    """Executes a subprocess command displaying the command string."""
    print(f"\n> {' '.join(str(x) for x in cmd_args)}")
    return subprocess.call([str(x) for x in cmd_args], cwd=cwd)


def menu():
    print_banner()
    
    model_exists = (ROOT_DIR / "models" / "isl_bilstm.pt").exists()
    status_suffix = " (Model Trained & Ready)" if model_exists else " (Model Not Found - Train First)"
    
    while True:
        print("\nMAIN MENU:")
        print("  1. Check Environment & Hardware")
        print("  2. Download / Prepare Official INCLUDE Dataset or Starter Data")
        print("  3. Preprocess & Resample Sequences")
        print("  4. Train ISL BiLSTM Neural Recognition Model")
        print(f"  5. Launch ClarifySign Accessible Web UI{status_suffix}")
        print("  6. Run Policy Benchmark Evaluation (B1 vs B2 vs B3 vs B4)")
        print("  7. Run Test Suite (27 Unit & Integration Tests)")
        print("  8. Exit")
        
        choice = input("\nEnter choice [1-8]: ").strip()
        
        if choice == "1":
            check_environment()
        elif choice == "2":
            print("\nDataset Options:")
            print("  a. Generate Calibrated Shopkeeper Starter Dataset (Recommended for fast offline setup)")
            print("  b. Fetch Official AI4Bharat INCLUDE Zenodo Metadata Record")
            print("  c. Download Official Category Archive (e.g. Colours_1of2.zip, Clothes_1of2.zip)")
            sub_c = input("Choose [a/b/c]: ").strip().lower()
            if sub_c == "b":
                run_command([PYTHON_EXE, "training/download_dataset.py", "--fetch-meta"])
            elif sub_c == "c":
                cat_name = input("Enter category archive name (e.g. Colours_1of2.zip): ").strip()
                if cat_name:
                    run_command([PYTHON_EXE, "training/download_dataset.py", "--category", cat_name])
            else:
                run_command([PYTHON_EXE, "training/download_dataset.py", "--starter", "--samples", "25"])
        elif choice == "3":
            run_command([PYTHON_EXE, "training/prepare_dataset.py"])
        elif choice == "4":
            if model_exists:
                confirm = input("A trained model already exists. Retrain anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    continue
            epochs = input("Enter training epochs [default: 25]: ").strip()
            epochs = int(epochs) if epochs.isdigit() else 25
            run_command([PYTHON_EXE, "training/train.py", "--epochs", str(epochs)])
            model_exists = (ROOT_DIR / "models" / "isl_bilstm.pt").exists()
        elif choice == "5":
            print("\nLaunching ClarifySign Streamlit Application...")
            run_command([PYTHON_EXE, "-m", "streamlit", "run", "app.py"])
        elif choice == "6":
            run_command([PYTHON_EXE, "evaluation/run_policy_evaluation.py"])
        elif choice == "7":
            run_command([PYTHON_EXE, "-m", "pytest", "-v"])
        elif choice == "8" or choice.lower() in ["exit", "q"]:
            print("Exiting ClarifySign launcher. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose a valid option (1-8).")


def main():
    parser = argparse.ArgumentParser(description="ClarifySign Main Launcher")
    parser.add_argument("--check", action="store_true", help="Check environment and hardware")
    parser.add_argument("--download", action="store_true", help="Generate or download dataset")
    parser.add_argument("--prepare", action="store_true", help="Preprocess sequences")
    parser.add_argument("--train", action="store_true", help="Train recognition model")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--eval", action="store_true", help="Run policy benchmark")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--all", action="store_true", help="Run full pipeline end-to-end")
    args = parser.parse_args()

    if args.check:
        check_environment()
    elif args.download:
        run_command([PYTHON_EXE, "training/download_dataset.py", "--starter", "--samples", "25"])
    elif args.prepare:
        run_command([PYTHON_EXE, "training/prepare_dataset.py"])
    elif args.train:
        run_command([PYTHON_EXE, "training/train.py", "--epochs", "25"])
    elif args.app:
        run_command([PYTHON_EXE, "-m", "streamlit", "run", "app.py"])
    elif args.eval:
        run_command([PYTHON_EXE, "evaluation/run_policy_evaluation.py"])
    elif args.test:
        run_command([PYTHON_EXE, "-m", "pytest", "-v"])
    elif args.all:
        check_environment()
        run_command([PYTHON_EXE, "training/download_dataset.py", "--starter", "--samples", "25"])
        run_command([PYTHON_EXE, "training/prepare_dataset.py"])
        run_command([PYTHON_EXE, "training/train.py", "--epochs", "25"])
        run_command([PYTHON_EXE, "evaluation/run_policy_evaluation.py"])
        run_command([PYTHON_EXE, "-m", "pytest", "-v"])
    else:
        menu()


if __name__ == "__main__":
    main()
