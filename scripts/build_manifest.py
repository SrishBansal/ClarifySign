"""Build JSONL manifests from user-accessed local metadata; never downloads data."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clarifysign.data.manifest import DatasetManifestRecord, read_jsonl, write_jsonl
from clarifysign.data.validation import validate_manifest


def data_root() -> Path:
    value = os.environ.get("CLARIFYSIGN_DATA_ROOT")
    if not value:
        raise RuntimeError("Set CLARIFYSIGN_DATA_ROOT to a local directory outside Git before building a manifest.")
    return Path(value).expanduser().resolve()


def content_hash_for_artifacts(record: DatasetManifestRecord, root: Path) -> str:
    """Hash local bytes when an approved local artifact is available; otherwise hash metadata."""
    digest = sha256()
    found = False
    for relative_path in (record.raw_video, record.extracted_pose):
        if relative_path:
            path = root / relative_path
            if path.is_file():
                found = True
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest() if found else record.canonical_content_hash()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-redistributable ClarifySign dataset JSONL manifest")
    parser.add_argument("--input", required=True, help="JSONL metadata path relative to CLARIFYSIGN_DATA_ROOT")
    parser.add_argument("--output", required=True, help="manifest output path relative to CLARIFYSIGN_DATA_ROOT")
    args = parser.parse_args()
    root = data_root()
    input_path, output_path = root / args.input, root / args.output
    records = [replace(record, content_hash=content_hash_for_artifacts(record, root)) for record in read_jsonl(input_path)]
    report = validate_manifest(records)
    if report.duplicate_source_ids or report.duplicate_content_hashes:
        raise RuntimeError(json.dumps(report.to_dict(), ensure_ascii=False))
    write_jsonl(records, output_path)
    print(json.dumps(report.to_dict() | {"output": str(output_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
