# Dataset Manifest Schema

ClarifySign records only JSONL metadata manifests in its data pipeline. Raw videos, extracted poses, captions, and gated archives remain in the private directory selected by `CLARIFYSIGN_DATA_ROOT`; they must not be committed or redistributed from this repository.

## Format

One UTF-8 JSON object per line represents one original source sample or source segment. Paths are always relative to `CLARIFYSIGN_DATA_ROOT`, never absolute machine paths.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `source_dataset` | string enum | yes | `isign`, `isltranslate`, `cislr`, or `approved_isl_dictionary`. |
| `license` | string | yes | The exact license/terms supplied for the accessed release. Do not infer a license from a paper. |
| `original_sample_id` | string | yes | Immutable ID from the original provider. |
| `source_url` | HTTPS URL | yes | Dataset card, repository, record, or approved dictionary source. |
| `split` | enum | yes | `train`, `validation`, `test`, or `unassigned`. |
| `language` | string | yes | Primary signed/spoken language context; default `isl`. |
| `processing_version` | string | yes | Version of the local extraction/normalization pipeline. |
| `signer_id` | string/null | no | Original signer/video identity when released. Required for signer-independent splitting. |
| `raw_video` | relative path/null | no | Video relative to the private data root. |
| `extracted_pose` | relative path/null | no | Pose artifact relative to the private data root. |
| `english_text` | string/null | no | English reference text. |
| `hindi_text` | string/null | no | Hindi reference text. |
| `marathi_text` | string/null | no | Marathi reference text. |
| `gloss` | string/null | no | Released or approved gloss annotation. |
| `quality_score` | number/null | no | Local quality value in `[0, 1]`; document its method in `processing_version`. |
| `content_hash` | SHA-256 string/null | no | Digest of local video/pose bytes when available; otherwise deterministic manifest-content digest. |

At least one modality/annotation field should be populated. The current validator reports empty records but does not discard them silently.

## Example

```json
{"content_hash":"<sha256>","english_text":"example","extracted_pose":"poses/segment-001.npz","gloss":"EXAMPLE","hindi_text":null,"language":"isl","license":"provider terms verified locally","marathi_text":null,"original_sample_id":"provider-001","processing_version":"pose-v1","quality_score":0.9,"raw_video":"videos/segment-001.mp4","signer_id":"signer-07","source_dataset":"isign","source_url":"https://huggingface.co/datasets/Exploration-Lab/iSign","split":"train"}
```

## Split and duplicate rules

- `assign_signer_independent_splits` assigns each signer deterministically to exactly one split. It rejects records without `signer_id`; use a source-group split policy only after documenting why signer metadata is unavailable.
- A duplicate source key is `source_dataset:original_sample_id`.
- A duplicate content hash indicates byte-identical local artifacts or equivalent supplied digests. Such records must be reviewed before training/evaluation.
- A signer observed in more than one assigned split is a leakage error.
- `scripts/validate_manifest.py` emits a JSON validation report and exits non-zero for source-ID, content-hash, or signer-split failures.

## Commands

```bash
export CLARIFYSIGN_DATA_ROOT=/private/location/clarifysign-data
python3 scripts/build_manifest.py --input metadata/isign.jsonl --output manifests/isign.jsonl
python3 scripts/validate_manifest.py --manifest manifests/isign.jsonl --report reports/isign-validation.json
```

The repository does not provide the input metadata or any raw samples; the example commands are intentionally local-only.
