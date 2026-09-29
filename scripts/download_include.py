"""
ClarifySign - Backward Compatibility Module for scripts/download_include.py
Delegates to training.download_dataset.
"""
from training.download_dataset import main

if __name__ == "__main__":
    main()
