"""Portable JSONL manifest types; raw data is always external to the repository."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal


Split = Literal["train", "validation", "test", "unassigned"]
DATASET_NAMES = frozenset({"isign", "isltranslate", "cislr", "approved_isl_dictionary"})


def _relative_artifact(value: str | None) -> str | None:
    if value is None:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact paths must be relative to CLARIFYSIGN_DATA_ROOT")
    return value


@dataclass(frozen=True, slots=True)
class DatasetManifestRecord:
    source_dataset: str
    license: str
    original_sample_id: str
    source_url: str
    split: Split = "unassigned"
    language: str = "isl"
    processing_version: str = "v1"
    signer_id: str | None = None
    raw_video: str | None = None
    extracted_pose: str | None = None
    english_text: str | None = None
    hindi_text: str | None = None
    marathi_text: str | None = None
    gloss: str | None = None
    quality_score: float | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if self.source_dataset not in DATASET_NAMES:
            raise ValueError(f"unsupported dataset source: {self.source_dataset}")
        if not self.license.strip() or not self.original_sample_id.strip() or not self.source_url.startswith("http"):
            raise ValueError("license, original_sample_id, and an http(s) source_url are required")
        if self.split not in {"train", "validation", "test", "unassigned"}:
            raise ValueError("invalid split")
        for field_name in ("raw_video", "extracted_pose"):
            object.__setattr__(self, field_name, _relative_artifact(getattr(self, field_name)))
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")

    @property
    def source_key(self) -> str:
        return f"{self.source_dataset}:{self.original_sample_id}"

    def canonical_content_hash(self) -> str:
        if self.content_hash:
            return self.content_hash
        payload = {
            "raw_video": self.raw_video,
            "extracted_pose": self.extracted_pose,
            "english_text": self.english_text,
            "hindi_text": self.hindi_text,
            "marathi_text": self.marathi_text,
            "gloss": self.gloss,
        }
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["content_hash"] = self.canonical_content_hash()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifestRecord":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ManifestValidationReport:
    total_records: int
    duplicate_source_ids: tuple[str, ...]
    duplicate_content_hashes: tuple[str, ...]
    signer_split_leaks: tuple[str, ...]
    missing_modalities: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not any((self.duplicate_source_ids, self.duplicate_content_hashes, self.signer_split_leaks))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"valid": self.valid}


def write_jsonl(records: Iterable[DatasetManifestRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(input_path: Path) -> list[DatasetManifestRecord]:
    records: list[DatasetManifestRecord] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(DatasetManifestRecord.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid manifest record at line {line_number}: {error}") from error
    return records
