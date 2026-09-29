from __future__ import annotations

from typing import Protocol, Sequence


class PosePreprocessor(Protocol):
    def transform(self, pose_sequence: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]: ...
