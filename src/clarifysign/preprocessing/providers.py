from __future__ import annotations

from typing import Sequence


class IdentityPosePreprocessor:
    """Deterministic local adapter used until a pose-extraction provider is selected."""

    def transform(self, pose_sequence: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        if not pose_sequence:
            raise ValueError("pose_sequence cannot be empty")
        return tuple(tuple(float(value) for value in frame) for frame in pose_sequence)
