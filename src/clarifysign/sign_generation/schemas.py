"""Pydantic schemas for provenance-aware ISL generation and playback."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SemanticRepresentation(_Schema):
    source_text: str = Field(min_length=1)
    selected_language: str = Field(min_length=2)
    intent: str = Field(min_length=1)
    slots: dict[str, str] = Field(default_factory=dict)
    proper_names: tuple[str, ...] = ()


class ISLGlossPlan(_Schema):
    """An approved ISL plan, never an arbitrary English-token sequence."""

    glosses: tuple[str, ...] = Field(min_length=1)
    planner_version: str = Field(min_length=1)
    approved: bool = True

    @model_validator(mode="after")
    def requires_approved_plan(self) -> "ISLGlossPlan":
        if not self.approved:
            raise ValueError("only approved ISL gloss plans may be rendered")
        return self


class AssetAttribution(_Schema):
    asset_id: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https?://")
    license: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    terms_version: str = Field(min_length=1)


class DictionaryVideoAsset(_Schema):
    playback_uri: str = Field(min_length=1)
    attribution: AssetAttribution
    glosses: tuple[str, ...] = Field(min_length=1)
    is_phrase_asset: bool = False


class PoseSequenceAsset(_Schema):
    pose_uri: str = Field(min_length=1)
    attribution: AssetAttribution
    frame_count: int = Field(gt=0)


class ClarificationAlternative(_Schema):
    alternative_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    action: Literal["retry", "show_text", "rephrase", "choose_candidate"]


class PlaybackControls(_Schema):
    replay_enabled: bool = True
    pause_enabled: bool = True
    slow_playback_enabled: bool = True
    alternatives: tuple[ClarificationAlternative, ...] = ()


class AvatarRender(_Schema):
    render_id: str = Field(min_length=1)
    renderer_version: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)


class SignGenerationResult(_Schema):
    status: Literal["ready", "fallback_text", "rephrase_required"]
    provenance: Literal["dictionary_based", "retrieved_phrase", "generated_pose", "fallback_text"]
    semantic: SemanticRepresentation
    gloss_plan: ISLGlossPlan | None = None
    dictionary_asset: DictionaryVideoAsset | None = None
    pose_asset: PoseSequenceAsset | None = None
    avatar_render: AvatarRender | None = None
    visible_text: str | None = None
    rephrase_prompt: str | None = None
    controls: PlaybackControls

    @model_validator(mode="after")
    def result_has_safe_rendering_state(self) -> "SignGenerationResult":
        if self.status == "ready" and self.avatar_render is None:
            raise ValueError("ready playback requires an avatar render")
        if self.status == "fallback_text" and not self.visible_text:
            raise ValueError("fallback_text requires visible text")
        if self.status == "rephrase_required" and not self.rephrase_prompt:
            raise ValueError("rephrase_required requires a prompt")
        return self
