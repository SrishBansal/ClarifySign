"""Explicitly initiate an approved local dataset retrieval command; never commits data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


SOURCES = {
    "isign": "https://huggingface.co/datasets/Exploration-Lab/iSign",
    "isltranslate": "https://github.com/Exploration-Lab/ISLTranslate",
    "cislr": "https://huggingface.co/datasets/IIT-K/CISLR",
    "approved_isl_dictionary": "set CLARIFYSIGN_ISL_DICTIONARY_URL after legal approval",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Initiate a user-authorized dataset retrieval command")
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--execute", action="store_true", help="run the supplied retrieval command")
    parser.add_argument("--command", nargs="+", help="approved retrieval command, without a shell")
    args = parser.parse_args()
    root_value = os.environ.get("CLARIFYSIGN_DATA_ROOT")
    if not root_value:
        raise RuntimeError("Set CLARIFYSIGN_DATA_ROOT to a private, Git-ignored directory first.")
    root = Path(root_value).expanduser().resolve()
    print(f"Source: {args.source}\nAccess page: {SOURCES[args.source]}\nDestination: {root}")
    if not args.execute:
        print("No download started. Review the dataset terms, obtain access, then supply --execute --command <approved command>.")
        return
    if not args.command:
        raise RuntimeError("--execute requires an explicit approved --command; this script never guesses gated access commands.")
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(args.command, cwd=root, check=True)


if __name__ == "__main__":
    main()
