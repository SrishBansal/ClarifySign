from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from clarifysign.sign_generation.schemas import AvatarRender, DictionaryVideoAsset, PoseSequenceAsset


@dataclass(frozen=True, slots=True)
class MockAvatarProvider:
    def render(self, motion_plan: Sequence[str]) -> str:
        return f"mock-avatar:{'|'.join(motion_plan)}"


@dataclass(frozen=True, slots=True)
class MockAvatarRenderer:
    """Test renderer that returns a render handle without embedding any media asset."""

    def render_asset(self, asset: DictionaryVideoAsset | PoseSequenceAsset) -> AvatarRender:
        uri = asset.playback_uri if isinstance(asset, DictionaryVideoAsset) else asset.pose_uri
        return AvatarRender(render_id=f"mock-render:{asset.attribution.asset_id}", renderer_version="mock-avatar-v1", source_uri=uri)
