from pathlib import Path

import pytest

from clarifysign.data.manifest import DatasetManifestRecord, read_jsonl, write_jsonl
from clarifysign.data.splits import assign_signer_independent_splits
from clarifysign.data.validation import validate_manifest


def record(sample_id: str, signer: str = "signer-a", **kwargs: object) -> DatasetManifestRecord:
    defaults: dict[str, object] = {
        "source_dataset": "isign",
        "license": "CC-BY-NC-SA-4.0",
        "original_sample_id": sample_id,
        "source_url": "https://example.test/isign",
        "signer_id": signer,
        "raw_video": f"raw/{sample_id}.mp4",
        "english_text": "synthetic text",
    }
    defaults.update(kwargs)
    return DatasetManifestRecord(**defaults)  # type: ignore[arg-type]


def test_jsonl_round_trip_uses_relative_artifacts_only(tmp_path: Path) -> None:
    output = tmp_path / "manifest.jsonl"
    write_jsonl([record("one")], output)
    restored = read_jsonl(output)
    assert restored[0].original_sample_id == "one"
    assert restored[0].raw_video == "raw/one.mp4"
    with pytest.raises(ValueError, match="relative"):
        record("bad", raw_video="/private/data/video.mp4")


def test_validation_detects_source_and_content_duplicates_and_signer_leakage() -> None:
    first = record("one", split="train", content_hash="same")
    duplicate = record("one", split="test", content_hash="same")
    report = validate_manifest([first, duplicate])
    assert report.duplicate_source_ids == ("isign:one",)
    assert report.duplicate_content_hashes == ("same",)
    assert report.signer_split_leaks == ("signer-a",)
    assert report.valid is False


def test_signer_independent_split_never_divides_a_signer() -> None:
    records = [record("one", "signer-a"), record("two", "signer-a"), record("three", "signer-b")]
    assigned = assign_signer_independent_splits(records)
    signer_a_splits = {item.split for item in assigned if item.signer_id == "signer-a"}
    assert len(signer_a_splits) == 1
    assert validate_manifest(assigned).signer_split_leaks == ()
    with pytest.raises(ValueError, match="signer_id"):
        assign_signer_independent_splits([record("missing", signer=None)])
