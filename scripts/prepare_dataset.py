"""
ClarifySign - Backward Compatibility Module for scripts/prepare_dataset.py
Delegates to training.prepare_dataset.
"""
from training.prepare_dataset import main

if __name__ == "__main__":
    main()
