from __future__ import annotations

from typing import Protocol, Sequence


class AvatarProvider(Protocol):
    def render(self, motion_plan: Sequence[str]) -> str: ...
