from __future__ import annotations

from dataclasses import dataclass

from clarifysign.data.schemas import SemanticIntent
from clarifysign.sign_generation.schemas import AssetAttribution, ISLGlossPlan, PoseSequenceAsset, SemanticRepresentation


@dataclass(frozen=True, slots=True)
class MockSignGenerationProvider:
    def generate(self, intent: SemanticIntent) -> tuple[str, ...]:
        return ("neutral", f"intent:{intent.intent}", "neutral")


@dataclass(frozen=True, slots=True)
class MockSemanticNormalizer:
    """Test normalizer that carries text as an intent; it creates no ISL grammar."""

    def normalize(self, text: str, selected_language: str) -> SemanticRepresentation:
        return SemanticRepresentation(source_text=text, selected_language=selected_language, intent="unresolved_text")


@dataclass(frozen=True, slots=True)
class ApprovedMockGlossPlanner:
    """Returns a plan only for an explicitly supplied semantic intent."""

    intent: str = "known_intent"
    glosses: tuple[str, ...] = ("KNOWN",)

    def plan(self, semantic: SemanticRepresentation) -> ISLGlossPlan | None:
        return ISLGlossPlan(glosses=self.glosses, planner_version="mock-approved-v1") if semantic.intent == self.intent else None


@dataclass(frozen=True, slots=True)
class MockPoseSequenceGenerator:
    """Synthetic pose output for tests; no model, asset, or download is used."""

    supports_intent: str = "generated_intent"

    def generate_pose(self, semantic: SemanticRepresentation, plan: ISLGlossPlan | None) -> PoseSequenceAsset | None:
        if semantic.intent != self.supports_intent:
            return None
        return PoseSequenceAsset(
            pose_uri="mock://pose/generated", frame_count=12,
            attribution=AssetAttribution(asset_id="mock-pose", source_url="https://example.test/mock", license="test-only", attribution="ClarifySign mock", terms_version="v1"),
        )


@dataclass(frozen=True, slots=True)
class MockFingerspellingProvider:
    def spell(self, proper_name: str) -> PoseSequenceAsset | None:
        if not proper_name.strip():
            return None
        return PoseSequenceAsset(
            pose_uri=f"mock://fingerspell/{proper_name}", frame_count=max(1, len(proper_name)),
            attribution=AssetAttribution(asset_id="mock-fingerspelling", source_url="https://example.test/mock", license="test-only", attribution="ClarifySign mock", terms_version="v1"),
        )
