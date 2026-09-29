"""Duplicate, modality, and signer-leakage validation for manifests."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from clarifysign.data.manifest import DatasetManifestRecord, ManifestValidationReport


def validate_manifest(records: Iterable[DatasetManifestRecord]) -> ManifestValidationReport:
    items = list(records)
    source_ids: dict[str, int] = defaultdict(int)
    hashes: dict[str, int] = defaultdict(int)
    signer_splits: dict[str, set[str]] = defaultdict(set)
    missing_modalities: list[str] = []
    for record in items:
        source_ids[record.source_key] += 1
        hashes[record.canonical_content_hash()] += 1
        if record.signer_id and record.split != "unassigned":
            signer_splits[record.signer_id].add(record.split)
        if not any((record.raw_video, record.extracted_pose, record.english_text, record.hindi_text, record.marathi_text, record.gloss)):
            missing_modalities.append(record.source_key)
    return ManifestValidationReport(
        total_records=len(items),
        duplicate_source_ids=tuple(sorted(key for key, count in source_ids.items() if count > 1)),
        duplicate_content_hashes=tuple(sorted(key for key, count in hashes.items() if count > 1)),
        signer_split_leaks=tuple(sorted(signer for signer, splits in signer_splits.items() if len(splits) > 1)),
        missing_modalities=tuple(sorted(missing_modalities)),
    )
