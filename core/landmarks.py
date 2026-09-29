"""
ClarifySign - Landmark Extraction Module
Extracts 225-dimensional skeletal landmark features (left hand, right hand, pose)
from video frames using MediaPipe Holistic Tasks.
Normalizes keypoints with root-relative zero-centering and temporal sequence resampling.
"""

import os
import sys
import urllib.request
from pathlib import Path
import numpy as np
import cv2

# Disable GPU/Metal delegate on macOS to prevent DrishtiMetalHelper crash
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

from config import FEATURE_DIM, SEQUENCE_LENGTH, HOLISTIC_TASK_PATH

HOLISTIC_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"


def ensure_holistic_model(model_path=HOLISTIC_TASK_PATH):
    """Ensures the MediaPipe holistic landmarker task file is downloaded locally."""
    path = Path(model_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe Holistic model to {path}...")
        try:
            urllib.request.urlretrieve(HOLISTIC_MODEL_URL, path)
            print("Holistic model downloaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to download MediaPipe model ({e}). Fallback mode active.")
            return False
    return True


class HolisticExtractor:
    """
    Extracts 225-dimensional hand and pose landmarks from RGB image frames.
    - Left Hand: 21 landmarks * 3 coords = 63 dims
    - Right Hand: 21 landmarks * 3 coords = 63 dims
    - Pose: 33 landmarks * 3 coords = 99 dims
    Total: 225 dims
    """

    def __init__(self, model_path=HOLISTIC_TASK_PATH):
        self.model_path = Path(model_path)
        self.detector = None
        self.feature_dim = FEATURE_DIM

        if MEDIAPIPE_AVAILABLE and ensure_holistic_model(self.model_path):
            try:
                base_options = python.BaseOptions(
                    model_asset_path=str(self.model_path),
                    delegate=python.BaseOptions.Delegate.CPU
                )
                options = vision.HolisticLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False
                )
                self.detector = vision.HolisticLandmarker.create_from_options(options)
            except Exception as e:
                print(f"Warning: HolisticLandmarker failed to initialize ({e}). Using mock/zero fallback.")
                self.detector = None

    def _pack_landmarks(self, landmarks_list, num_expected):
        """Packs landmark objects into an (N, 3) numpy array."""
        out = np.zeros((num_expected, 3), dtype=np.float32)
        if landmarks_list and len(landmarks_list) > 0:
            # landmarks_list can be list of landmarks or list of lists
            first = landmarks_list[0] if isinstance(landmarks_list[0], list) else landmarks_list
            for i, p in enumerate(first[:num_expected]):
                out[i] = [p.x, p.y, p.z]
            # Zero-center relative to the root landmark (wrist or shoulder)
            out = out - out[0]
        return out

    def __call__(self, frame):
        """
        Processes a single BGR video frame and returns a normalized (225,) feature vector.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        if self.detector is None:
            # Fallback: estimate hand presence from color/motion or return zeros
            return np.zeros(self.feature_dim, dtype=np.float32)

        try:
            # MediaPipe tasks requires RGB image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)

            # Left hand: 21 landmarks
            lh = self._pack_landmarks(result.left_hand_landmarks, 21)
            # Right hand: 21 landmarks
            rh = self._pack_landmarks(result.right_hand_landmarks, 21)
            # Pose: 33 landmarks
            pose = self._pack_landmarks(result.pose_landmarks, 33)

            features = np.concatenate([lh.ravel(), rh.ravel(), pose.ravel()]).astype(np.float32)
            if features.shape[0] != self.feature_dim:
                padded = np.zeros(self.feature_dim, dtype=np.float32)
                padded[:min(len(features), self.feature_dim)] = features[:self.feature_dim]
                return padded
            return features
        except Exception:
            return np.zeros(self.feature_dim, dtype=np.float32)

    def close(self):
        """Releases underlying resources."""
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None


def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    """
    Uniformly resamples a sequence of arbitrary length T into exactly target_length frames
    using linear index interpolation.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    t = len(seq)
    if t == 0:
        return np.zeros((target_length, FEATURE_DIM), dtype=np.float32)
    if t == target_length:
        return seq

    indices = np.linspace(0, t - 1, target_length)
    low_idx = np.floor(indices).astype(int)
    high_idx = np.ceil(indices).astype(int)
    weight = (indices - low_idx)[:, np.newaxis]

    resampled = (1.0 - weight) * seq[low_idx] + weight * seq[high_idx]
    return resampled.astype(np.float32)
