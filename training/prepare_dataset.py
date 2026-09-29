"""
ClarifySign - Dataset Preprocessing Pipeline
Processes raw video files (.mp4, .avi, .mov) or landmark sequences (.npy),
resamples temporal sequences to exactly SEQUENCE_LENGTH=48 frames,
and saves standardized feature matrices into data/processed/<class_name>/.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCE_LENGTH,
    FEATURE_DIM,
    PREPROCESSING_CONFIG_PATH
)
from core.landmarks import HolisticExtractor, resample_sequence


def prepare_from_numpy(source_dir, output_dir, target_length=SEQUENCE_LENGTH):
    """Prepares existing .npy landmark sequence directories."""
    count = 0
    class_dirs = [d for d in Path(source_dir).iterdir() if d.is_dir()]

    for c_dir in sorted(class_dirs):
        class_name = c_dir.name
        out_cls_dir = Path(output_dir) / class_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        npy_files = list(c_dir.glob("*.npy"))
        for npy_path in npy_files:
            try:
                seq = np.load(npy_path)
                resampled = resample_sequence(seq, target_length=target_length)
                dest = out_cls_dir / npy_path.name
                np.save(dest, resampled)
                count += 1
            except Exception as e:
                print(f"Warning: Failed processing {npy_path}: {e}")

    return count


def prepare_from_videos(source_dir, output_dir, target_length=SEQUENCE_LENGTH):
    """Extracts MediaPipe holistic landmarks from video files and resamples them."""
    extractor = HolisticExtractor()
    count = 0

    video_exts = (".mp4", ".avi", ".mov", ".mkv")

    for root, _, files in os.walk(source_dir):
        rel_path = Path(root).relative_to(source_dir)
        class_name = rel_path.parts[0] if rel_path.parts else "unknown"

        video_files = [f for f in files if f.lower().endswith(video_exts)]
        if not video_files:
            continue

        out_cls_dir = Path(output_dir) / class_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        for v_file in video_files:
            video_path = Path(root) / v_file
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                feat = extractor(frame)
                frames.append(feat)

            cap.release()

            if not frames:
                continue

            seq = resample_sequence(frames, target_length=target_length)
            stem = video_path.stem
            dest = out_cls_dir / f"{stem}.npy"
            np.save(dest, seq)
            count += 1

    extractor.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare ClarifySign ISL landmark dataset")
    parser.add_argument("--input", default=str(RAW_DATA_DIR), help="Input raw data directory")
    parser.add_argument("--output", default=str(PROCESSED_DATA_DIR), help="Output processed directory")
    parser.add_argument("--length", type=int, default=SEQUENCE_LENGTH, help="Target sequence length")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for data in: {input_path}")

    # Check for starter numpy landmark directory first
    starter_candidate = input_path / "include_shopkeeper_starter"
    total_prepared = 0

    if starter_candidate.exists():
        print(f"Found starter landmark dataset at {starter_candidate}")
        total_prepared = prepare_from_numpy(starter_candidate, output_path, target_length=args.length)
    else:
        # Check if there are .npy files directly
        any_npy = list(input_path.rglob("*.npy"))
        if any_npy:
            total_prepared = prepare_from_numpy(input_path, output_path, target_length=args.length)
        else:
            # Video extraction
            total_prepared = prepare_from_videos(input_path, output_path, target_length=args.length)

    print(f"\nPreprocessing Complete. Prepared {total_prepared} sequences.")

    # Save preprocessing metadata configuration
    prep_config = {
        "sequence_length": args.length,
        "feature_dim": FEATURE_DIM,
        "normalization": "root_zero_centered_wrist_pose",
        "temporal_resampling": "linear_index_interpolation",
        "processed_samples_count": total_prepared,
        "classes_prepared": sorted([d.name for d in output_path.iterdir() if d.is_dir()])
    }

    with open(PREPROCESSING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(prep_config, f, indent=2)

    print(f"Saved preprocessing config to {PREPROCESSING_CONFIG_PATH}")


if __name__ == "__main__":
    main()
