"""Validate a JSONL manifest using an environment-configured local data root."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clarifysign.data.manifest import read_jsonl
from clarifysign.data.validation import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ClarifySign dataset manifest JSONL")
    parser.add_argument("--manifest", required=True, help="path relative to CLARIFYSIGN_DATA_ROOT")
    parser.add_argument("--report", help="optional JSON report path relative to CLARIFYSIGN_DATA_ROOT")
    args = parser.parse_args()
    root_value = os.environ.get("CLARIFYSIGN_DATA_ROOT")
    if not root_value:
        raise RuntimeError("Set CLARIFYSIGN_DATA_ROOT before validating a manifest.")
    root = Path(root_value).expanduser().resolve()
    report = validate_manifest(read_jsonl(root / args.manifest)).to_dict()
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        output = root / args.report
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
