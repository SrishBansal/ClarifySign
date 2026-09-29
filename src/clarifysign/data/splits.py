"""Deterministic signer-independent split assignment."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Iterable

from clarifysign.data.manifest import DatasetManifestRecord


def assign_signer_independent_splits(
    records: Iterable[DatasetManifestRecord],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> list[DatasetManifestRecord]:
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("invalid split fractions")
    items = list(records)
    absent = [item.source_key for item in items if not item.signer_id]
    if absent:
        raise ValueError(f"signer-independent split requires signer_id; missing: {', '.join(absent)}")
    signer_to_split: dict[str, str] = {}
    for signer in sorted({item.signer_id for item in items if item.signer_id}):
        bucket = int(sha256(signer.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        signer_to_split[signer] = "train" if bucket < train_fraction else "validation" if bucket < train_fraction + validation_fraction else "test"
    return [replace(item, split=signer_to_split[item.signer_id or ""]) for item in items]
