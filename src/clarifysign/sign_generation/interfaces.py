from __future__ import annotations

from typing import Protocol, Sequence

from clarifysign.data.schemas import SemanticIntent
from clarifysign.sign_generation.schemas import (
    AvatarRender,
    DictionaryVideoAsset,
    ISLGlossPlan,
    PoseSequenceAsset,
    SemanticRepresentation,
)


class SignGenerationProvider(Protocol):
    """Creates a sign-motion plan; it does not select a static video."""

    def generate(self, intent: SemanticIntent) -> tuple[str, ...]: ...


class SemanticNormalizer(Protocol):
    def normalize(self, text: str, selected_language: str) -> SemanticRepresentation: ...


class ISLGlossPlanner(Protocol):
    def plan(self, semantic: SemanticRepresentation) -> ISLGlossPlan | None: ...


class DictionaryVideoRetriever(Protocol):
    def retrieve_phrase(self, semantic: SemanticRepresentation) -> DictionaryVideoAsset | None: ...

    def retrieve_gloss_plan(self, plan: ISLGlossPlan) -> DictionaryVideoAsset | None: ...


class PoseSequenceGenerator(Protocol):
    def generate_pose(self, semantic: SemanticRepresentation, plan: ISLGlossPlan | None) -> PoseSequenceAsset | None: ...


class FingerspellingProvider(Protocol):
    def spell(self, proper_name: str) -> PoseSequenceAsset | None: ...


class AvatarRenderer(Protocol):
    def render_asset(self, asset: DictionaryVideoAsset | PoseSequenceAsset) -> AvatarRender: ...
