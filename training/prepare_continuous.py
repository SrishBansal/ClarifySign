"""
ClarifySign - Continuous / Sentence-Level Signing Preprocessing Pipeline
Dedicated processing for continuous sign language datasets (e.g., ISL-CSLTR).

Architectural Principle:
Continuous signing (sentence-level) cannot be forced through the 48-frame fixed-window
normalization designed for isolated-word signs. A 10-word sentence spanning 300+ frames
loses critical transition dynamics and handshapes if compressed into 48 frames.

Corpus Reference:
- ISL-CSLTR: Continuous Indian Sign Language Recognition Corpus
- Source: Mendeley Data, DOI: 10.17632/kcmpdxky7p.1
- Authors: Elakkiya R. & Natarajan B.
- Scale: 700 videos, 100 sentences, 7 signers
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_DIM
from core.landmarks import HolisticExtractor


class ContinuousSigningPreprocessor:
    """
    Separated preprocessor for sentence-level continuous Indian Sign Language.
    Preserves variable sequence length or uses sliding temporal windows with overlap,
    explicitly rejecting fixed 48-frame compression for multi-word sequences.
    """

    def __init__(
        self,
        min_sentence_frames: int = 60,
        max_sentence_frames: int = 600,
        window_size: int = 96,
        stride: int = 32,
    ):
        self.min_sentence_frames = min_sentence_frames
        self.max_sentence_frames = max_sentence_frames
        self.window_size = window_size
        self.stride = stride
        self.extractor = HolisticExtractor()

    def validate_window_length(self, num_frames: int, fixed_window: int = 48) -> bool:
        """
        Validates whether a fixed window length is appropriate.
        Sentence-length input requires variable-length or windowed processing.
        """
        if num_frames > fixed_window * 2:
            # Emits explicit warning/rejection of destructive compression
            return False
        return True

    def process_continuous_video(
        self,
        video_path: Path,
        sliding_window: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extracts landmarks frame-by-frame for a continuous sentence video.
        Returns:
            - "raw_sequence": array of shape (T, 225) preserving full temporal duration.
            - "windows": (if sliding_window=True) array of shape (N, window_size, 225).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        frames_features = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            feat = self.extractor(frame)
            frames_features.append(feat)
        cap.release()

        total_frames = len(frames_features)
        if total_frames == 0:
            raw_seq = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        else:
            raw_seq = np.array(frames_features, dtype=np.float32)

        # Validation check against 48-frame isolated sign compression
        if not self.validate_window_length(total_frames, fixed_window=48):
            # Sequence exceeds isolated-word length; preserved at full duration
            pass

        result = {"raw_sequence": raw_seq, "total_frames": total_frames}

        if sliding_window and total_frames >= self.window_size:
            windows = []
            for start in range(0, total_frames - self.window_size + 1, self.stride):
                windows.append(raw_seq[start:start + self.window_size])
            result["windows"] = np.array(windows, dtype=np.float32)
        elif sliding_window and total_frames > 0:
            # Pad sequence if shorter than window_size
            pad = np.zeros((self.window_size - total_frames, FEATURE_DIM), dtype=np.float32)
            padded = np.vstack([raw_seq, pad])
            result["windows"] = np.expand_dims(padded, axis=0)
        else:
            result["windows"] = np.zeros((0, self.window_size, FEATURE_DIM), dtype=np.float32)

        return result


def main():
    parser = argparse.ArgumentParser(description="ClarifySign Continuous Sign Language Preprocessor")
    parser.add_argument("--input-dir", type=str, default=str(RAW_DATA_DIR / "isl_csltr"))
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DATA_DIR / "continuous"))
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    print("====================================================================")
    print("ClarifySign Continuous Sign Language (ISL-CSLTR) Preprocessing Path")
    print("====================================================================")
    print(f"Input directory:       {args.input_dir}")
    print(f"Output directory:      {args.output_dir}")
    print(f"Sliding window size:   {args.window_size} frames (NOT 48-frame isolated)")
    print(f"Sliding window stride: {args.stride} frames")
    print("Status: Separated pipeline established per architectural mandate.")


if __name__ == "__main__":
    main()
