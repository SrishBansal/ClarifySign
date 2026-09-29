# Dataset Retrieval

## Non-redistribution rule

Do not add raw videos, poses, captions, archives, access tokens, or gated dataset exports to Git. Set `CLARIFYSIGN_DATA_ROOT` to a private location outside this repository. Each team member must obtain access under the source's own terms.

The retrieval helper only initiates a command that the user explicitly supplies after review; it does not guess gated download endpoints or accept terms on the user's behalf.

```bash
export CLARIFYSIGN_DATA_ROOT=/private/location/clarifysign-data
python3 scripts/retrieve_dataset.py isign
# Review terms and access requirements, then provide the provider-approved command:
python3 scripts/retrieve_dataset.py isign --execute --command <approved-command-and-arguments>
```

`--command` is executed without a shell, in `CLARIFYSIGN_DATA_ROOT`. Keep tokens in the local environment or a credential manager, not in a command checked into this repository.

## Source-specific access checklist

| Source | Access page | Access and licensing action before retrieval | Manifest value |
|---|---|---|---|
| iSign | [Hugging Face dataset card](https://huggingface.co/datasets/Exploration-Lab/iSign) | Complete any gated-access form, confirm the card's CC-BY-NC-SA-4.0/research-only terms and non-redistribution commitment, then use the provider-approved download command. | `source_dataset: "isign"`; preserve provider UID/video ID as `original_sample_id`. |
| ISLTranslate | [Official project repository](https://github.com/Exploration-Lab/ISLTranslate) | Follow the maintainers' current access instructions. Verify the release-specific data license and video redistribution terms before creating a manifest; do not assume the code-repository license applies to data. | `source_dataset: "isltranslate"`; preserve the release's sample/video ID. |
| CISLR | [Hugging Face access page](https://huggingface.co/datasets/IIT-K/CISLR) | Obtain any required account approval/token and verify the current dataset-card terms. The corpus draws on dictionary material, so confirm downstream video/derivative-use rights before processing. | `source_dataset: "cislr"`; preserve corpus/dictionary sample ID and signer ID when supplied. |
| Future ISL dictionary | No default source is approved. Set `CLARIFYSIGN_ISL_DICTIONARY_URL` and `CLARIFYSIGN_ISL_DICTIONARY_LICENSE` only after legal, consent, quality, and redistribution review. | Record the approved source URL and exact license/terms in every row; use `approved_isl_dictionary`. | `source_dataset: "approved_isl_dictionary"`. |

The Indian Sign Language Dictionary is catalogued by India's Open Government Data platform, but a catalog entry alone is not approval to ingest or redistribute every associated media asset. Treat it as a candidate source until the review above is complete.

## Local workflow

1. Obtain source access directly from the provider and save material under the private data root.
2. Create source metadata JSONL using the schema in [DATA_MANIFEST_SCHEMA.md](DATA_MANIFEST_SCHEMA.md). Use relative artifact paths only.
3. Run `scripts/build_manifest.py`; it calculates content hashes from local video/pose files when present and refuses source-ID/content-hash duplicates.
4. Run `scripts/validate_manifest.py` and retain the report alongside the private manifest.
5. Use signer-independent splits when signer IDs exist. If they do not, stop before training and document a source-grouped split policy and leakage risk.

No downloader, manifest builder, or validation script implements training.
