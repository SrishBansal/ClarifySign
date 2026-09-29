# ClarifySign — Removal and Migration Plan

**Status:** planning only. This document authorizes no runtime-code, model, asset, dependency, or dataset change by itself.

**Evidence base:** [CURRENT_REPO_AUDIT.md](CURRENT_REPO_AUDIT.md), especially sections 3–10. Paths and line references below refer to the audited revision `79450a0` unless a later pull request changes them.

## 1. Target boundaries

The production system shall have two explicitly separate directions:

```text
ISL video / pose
  -> sign-to-text model
  -> confidence calibration
  -> clarification engine
  -> multilingual text / TTS

speech / text
  -> ASR
  -> semantic representation
  -> ISL sign generation
  -> interactive avatar
```

The second direction is a future workstream. It must not reuse a label lookup, canned sentence, static video, or an evaluation fixture as evidence of sign generation. No claim in this plan characterizes an avatar or bidirectional translation as novel.

## 2. Action register

Each row has exactly one disposition. Directory globs identify homogeneous generated/binary sets; every file matching the glob receives the stated disposition.

| Path | Action | Exact reason | Destination / gate |
|---|---|---|---|
| `.env.example` | KEEP | Empty `HF_TOKEN` placeholder; no secret committed (audit §5.4). | Keep as the only example env file; add no real value. |
| `.gitignore` | REFACTOR | It ignores `data/raw/` but not `.env`; it does not express the chosen treatment of generated `.npy` or downloaded `.task` bundles (audit §9). | Add `.env`; decide whether reproducible downloads or LFS own production artifacts. |
| `RUN_ME.py` | KEEP | Current supported launcher and test/train/UI dispatcher. | Retain while each command is made provenance-aware. |
| `RUN_WINDOWS.bat` | KEEP | Thin supported Windows forwarding launcher. | Retain. |
| `app.py` | REFACTOR | Mixes model-backed capture with authored Demo Mode; marks UI `LIVE` even when model/extractor is unavailable; has unused imports (`pandas`, `PIL.Image`, `synthesize_audio_bytes`) (audit §8; lines 13, 17, 46, 368–401, 417–419). | Split production inference view from an explicitly labelled evaluation/demo view. |
| `config.py` | REFACTOR | It combines model paths, policy thresholds, language codes, and the legacy 17-class product vocabulary (lines 18–20, 38–50, 90–109). | Separate versioned model schema/config from baseline fixture labels. |
| `requirements.txt` | REFACTOR | One unpinned mixed runtime/training/evaluation dependency list makes optional functionality mandatory. | Split into runtime, training, evaluation, and optional extras only after imports move. |
| `research_spec.md` | REPLACE | Says a checkpoint cannot honestly be bundled while `models/isl_bilstm.pt` and synthetic metrics are bundled (audit §7.2). | A dated research protocol with data provenance and claim boundaries. |
| `AUDIT.md` | REFACTOR | Duplicates audit material and describes the camera path out of date (audit §9). | Replace duplication with links to the current audit and this plan. |
| `CLARIFYSIGN_FINAL_REVIEW.md` | REFACTOR | Its policy figures disagree with committed JSON and it over-associates synthetic data with INCLUDE (audit §7.6). | Regenerate figures from a named, immutable evaluation artifact. |
| `README.md` | REFACTOR | Claims/results conflict with `results/policy_evaluation_results.json`; distinguishes neither synthetic baseline nor production model (audit §7.2–7.6). | State supported scope, data source, split, latency scope, and limitations beside each claim. |
| `core/__init__.py` | KEEP | Package surface used by tests and launch paths. | Maintain a small public API. |
| `core/landmarks.py` | REFACTOR | It returns zero features on detector failure (lines 87–118), which can be passed onward as a prediction. | Return an explicit extraction-status/error result and preserve raw frame consent boundaries. |
| `core/recognizer.py` | REFACTOR | Actual inference is present, but checkpoint/schema provenance and calibration are not separated (audit §5.2). | Load only a model manifest with label set, data version, calibration, and intended scope. |
| `core/model.py` | KEEP | Compatibility alias over the recognizer. | Keep until imports are migrated; then assess removal in a dedicated compatibility PR. |
| `core/infogain.py` | KEEP | Current noisy-answer information-gain implementation is the clarification core. | Keep as policy module with direct unit tests. |
| `core/clarification.py` | REFACTOR | Compatibility adapter and documentation diverge from `core/infogain.py` (audit §9). | Define one policy API and remove duplicate legacy semantics after parity tests. |
| `core/dialogue.py` | KEEP | Independent dialogue state used by the clarification path. | Retain; add persistence/privacy policy before storing user data. |
| `core/uncertainty.py` | REFACTOR | Threshold gating exists but calibrated confidence is not a separately evidenced model artifact. | Consume a fitted calibration artifact; distinguish uncertainty from extraction failure/OOD. |
| `core/translation.py` | REFACTOR | `VERIFIED_SEMANTIC_DATABASE` (20–253), `PHRASE_TEMPLATES` (256–305), contextual sentences (309–538), and inline compounds (586–608) are hard-coded language realization, not general translation. `PHRASE_TEMPLATES` is unused. | Move test/demo lexicons to baselines; production consumes a semantic representation and a versioned translation provider. |
| `core/speech.py` | REFACTOR | Browser speech is used; server gTTS is optional and unused by `app.py` (audit §8). Odia may fall back to Hindi. | Preserve browser TTS; make any cloud TTS an opt-in provider with language/consent/error reporting. |
| `training/__init__.py` | KEEP | Package marker. | Retain. |
| `training/download_dataset.py` | REFACTOR | It labels procedurally generated landmarks with INCLUDE/CC-BY attribution (127–239; audit §6.2) and has an unused GitHub URL. | Split official acquisition metadata from a clearly named synthetic-fixture generator. |
| `training/prepare_dataset.py` | REFACTOR | Supports video and `.npy` ingest but lacks a dataset manifest/consent/license gate. | Emit immutable manifests: source, license, signer split, preprocessing version, and checksums. |
| `training/train.py` | REFACTOR | Signer split safeguard is useful, but it trains current synthetic data and emits claims that need provenance. | Require a validated manifest and save reproducible run metadata/calibration data. |
| `training/evaluate_recognizer.py` | REPLACE | Scores all 425 processed files rather than the held-out partition (audit §5.2, §6.4). | New evaluator accepts a locked test manifest only and rejects overlap with train/validation. |
| `scripts/download_include.py` | KEEP | Compatibility wrapper. | Keep until CLI migration; test parity. |
| `scripts/prepare_dataset.py` | KEEP | Compatibility wrapper. | Keep until CLI migration; test parity. |
| `scripts/train.py` | KEEP | Compatibility wrapper. | Keep until CLI migration; test parity. |
| `scripts/policy_stress_test.py` | MOVE_TO_BASELINE | Wrapper launches a synthetic policy stress test, not production sign inference. | `baselines/policy_simulation/` after directory is introduced. |
| `evaluation/run_policy_evaluation.py` | MOVE_TO_BASELINE | Operates on authored candidate distributions, so it is a policy simulation, not recognizer validation. | `baselines/policy_simulation/`; retain provenance banner. |
| `evaluation/scenarios.json` | MOVE_TO_BASELINE | Ten hand-authored probabilities include non-model labels (`green`, `shirt`, `medium`) (audit §3.5). | `baselines/policy_simulation/fixtures/`; never load from production UI. |
| `tests/test_uncertainty.py` | KEEP | Covers uncertainty utility behavior. | Extend with calibration/extraction-status contract tests. |
| `tests/test_clarification.py` | KEEP | Core clarification coverage. | Maintain as policy parity tests. |
| `tests/test_dialogue.py` | KEEP | Dialogue state coverage. | Keep. |
| `tests/test_smoke.py` | REFACTOR | Its translation/recognizer smoke assumptions should select an explicit fixture/model. | Use a labelled test fixture, not production checkpoint claims. |
| `tests/test_policy.py` | REFACTOR | Two minimal one-line tests (audit §9). | Expand policy fixture coverage or merge after equivalent coverage is demonstrated. |
| `models/isl_bilstm.pt` | REPLACE | Current binary is trained/evaluated on synthetic kinematics but is presented alongside INCLUDE claims (audit §6.2, §10.6). | A real-data model only with manifest, license review, held-out results, calibration, and reproducible hash. |
| `models/labels.json` | REFACTOR | Duplicates labels from config/metadata. | Generate from the model manifest; keep only the manifest-owned form. |
| `models/preprocessing_config.json` | REFACTOR | Static declaration of synthetic processed count without full provenance. | Replace with versioned data/model manifest. |
| `models/training_metadata.json` | REPLACE | Records 100% synthetic-set statistics as model metadata (audit §5.1). | New run metadata tied to code/data/model hashes and split manifest. |
| `models/holistic_landmarker.task` | REFACTOR | Needed by current extractor but redistributed without a notice/license review (audit §10.3). | Download/cache from documented upstream with checksum and redistribution decision. |
| `models/hand_landmarker.task` | REMOVE | No Python reference; ~7.5 MB (audit §4). | Remove only after confirming release artifact and fresh clone do not require it. |
| `models/.gitkeep` | KEEP | Keeps model directory structure when binaries become fetched artifacts. | Retain. |
| `data/processed/.gitkeep` | KEEP | Keeps generated-data location. | Retain. |
| `data/processed/**/*.npy` (425 files) | MOVE_TO_BASELINE | Generated sine/Gaussian landmark fixtures, not INCLUDE-derived samples (audit §1.2, §6.2). | `baselines/synthetic_landmarks/` or generated test fixture package; never call production data. |
| `data/raw/**` (untracked) | KEEP | Raw video is correctly gitignored; it may contain identifiable people. | Keep local/private; use manifests, consent/license review, and no commit. |
| `results/.gitkeep` | KEEP | Directory structure. | Retain. |
| `results/policy_evaluation_results.json` | MOVE_TO_BASELINE | Results derive from synthetic scenarios; README claims conflict with it (audit §7.6). | `baselines/policy_simulation/results/` with script/version/seed. |
| `results/recognizer_evaluation.json` | REPLACE | Full-set evaluation and local absolute path reveal inaccurate provenance and a local path (audit §6.4). | Fresh held-out-only report with relative identifiers/checksums. |
| `docs/ARCHITECTURE.md` | REFACTOR | High-level flow needs the target directional boundary and baseline/production separation. | Update with §1 diagram and status labels. |
| `docs/CURRENT_REPO_AUDIT.md` | KEEP | Source evidence for this plan. | Preserve with audit revision/date. |
| `docs/DATASET.md` | REFACTOR | Conflates upstream INCLUDE with the synthetic starter. | Separate upstream citation, downloaded raw data, generated fixtures, and redistribution terms. |
| `docs/DATASET_SOURCES.md` | REFACTOR | iSign/ISLTranslate are citations only; wording must not imply availability. | Add “not integrated/not downloaded” status. |
| `docs/EVALUATION.md` | REPLACE | Tables conflict with committed policy JSON and overstate recognizer evidence (audit §7). | Report distinct recognizer, calibration, policy-simulation, and end-to-end studies. |
| `docs/FINAL_SUBMISSION_README.md` | REFACTOR | Checkpoint/accuracy statements contradict repository state (audit §7.2). | Align to actual artifact/manifests. |
| `docs/LIMITATIONS.md` | REFACTOR | Good scope caveats, but its perfect-oracle IG description is stale (audit §10.6). | Align to noisy-answer model and future sign-generation boundaries. |
| `docs/PROJECT_STATUS.md` | REMOVE | Stale Keras and synthetic-results statements contradict checked-in artifacts (audit §9). | Delete after the replacement status summary is in README/architecture. |
| `docs/RUNBOOK.md` | REFACTOR | Describes camera/cloud flow incompletely relative to WebRTC-first UI (audit §9). | Document supported capture modes and privacy/network behavior. |
| `docs/VIVA_EXPLANATION.md` | REFACTOR | Preserve its explanatory use, but align EIG semantics and evidence boundaries. | Cite current evaluator/manifests. |
| `docs/REMOVAL_AND_MIGRATION_PLAN.md` | KEEP | This staged plan. | Update only through reviewed plan changes. |

## 3. Hard-coded mappings: baseline, not production logic

There are no committed word-to-video, word-to-sign video, sentence-to-sign video, avatar, Lottie, GLB, GIF, MP4, or WebM mappings (audit §3.1 and §4). Therefore there is no hidden video lookup to preserve as production behavior.

The following hard-coded material must be baseline/demo fixture data, not production decision logic:

| Mapping / fixture | Current location | Required treatment |
|---|---|---|
| 17 English word-to-class labels | `config.py:90–109`, `models/labels.json:1–19`, preprocessing/training metadata | Treat as legacy synthetic-model vocabulary. A production label map comes only from its signed model manifest. |
| Concept-to-10-language word/phrase dictionary | `core/translation.py:20–253` | Move to `baselines/lexicon_fixture/`; retain for deterministic tests and UI demonstrations only. |
| Canned question templates | `core/translation.py:256–305` | Baseline fixture; currently unused. Do not promote as generated dialogue. |
| Concept-to-canned-sentence realizations | `core/translation.py:309–538`, `620–639` | Baseline fixture; replace production use with a semantic representation plus versioned realization provider. |
| Inline color/garment strings | `core/translation.py:586–608` | Baseline fixture; remove duplication by using the same test lexicon. |
| Ten synthetic posterior distributions | `evaluation/scenarios.json:1–147` | Policy simulation fixtures only, moved with the evaluation runner. |
| Synthetic gesture class signatures | `training/download_dataset.py:127–239`; `data/processed/**/*.npy` | Baseline/generated fixtures only; never call them “INCLUDE samples” or use their accuracy as live-model performance. |

Language-code maps in `config.py:38–50` and `core/speech.py:11–37` are configuration, not sign mappings. Retain them only after verifying individual language/TTS support and accurately reporting fallback behavior.

## 4. Dependency retirement and packaging

No dependency is removed in the documentation PR. Retire only after the code path and tests named below are removed or replaced.

| Dependency | Current evidence | Removal candidate? | Preconditions |
|---|---|---|---|
| `pandas` | Imported only by `app.py:14`; audit found no use. | Yes. | Remove import; run UI import and full tests. |
| `Pillow` | `PIL.Image` imported only by `app.py:17`; audit found no use. | Yes. | Remove import; verify no Streamlit transitive packaging reliance is assumed; test clean environment. |
| `matplotlib` | Listed in `requirements.txt:13`; no repository import. | Yes. | Confirm no documentation/notebook command depends on it; test all commands. |
| `gTTS` | Optional function in `core/speech.py:101–115`; app imports it but does not call it. | Yes, from default runtime. | First make server TTS an extra or remove it; retain browser TTS tests. |
| `transformers`, `sentencepiece` | Optional NLLB loading in `core/translation.py:555–650`. | Yes, from default runtime. | Move neural translation to `.[translation]` extra or retire it; no production path may silently claim neural translation without the extra. |
| `streamlit-webrtc`, `av` | Used together in `app.py:425–462`. | Conditional. | Remove only if the UI intentionally standardizes on `st.camera_input` or a replacement capture layer; verify camera regression tests. |
| `scipy` | `core/infogain.py:128` imports optimizer. | Not yet. | Replace optimizer and prove numerical/policy parity. |
| `scikit-learn` | Training labels/split and evaluation metrics. | Not yet. | Replace all training/evaluation uses and regression-test split/metrics. |
| `opencv-python`, `mediapipe`, `torch`, `numpy`, `streamlit`, `pytest` | Current capture/extraction/model/UI/test paths. | No, in this migration stage. | Reassess only with an approved architecture replacement. |

Target packaging after migration: a minimal production runtime; `training` extra; `evaluation` extra; optional `translation` and `server-tts` extras. Pin exact versions and record model/provider licenses in the release manifest.

## 5. Dataset and asset migration risks

1. **Provenance:** rename/move the 425 arrays without a manifest can further hide that they are generated fixtures. Preserve generator version, seed, class list, shape, checksum, and “synthetic” marker before moving them.
2. **Licensing:** do not attach INCLUDE’s CC-BY-4.0 notice to generated data merely because labels are INCLUDE-aligned. Record original license for original code/data separately; obtain a project source license and third-party notices.
3. **PII:** downloaded INCLUDE/raw recordings may show identifiable people. Keep `data/raw/` private and ignored; do not commit videos, thumbnails, or derived metadata that identifies a signer without permission.
4. **Model compatibility:** `isl_bilstm.pt`, `labels.json`, preprocessing settings, and feature dimensions are coupled. Never move/replace one independently; validate manifest hash, class order, `(48,225)` feature schema, and calibration artifact together.
5. **MediaPipe distribution:** `holistic_landmarker.task` is currently needed; move to verified on-demand fetch or documented vendor artifact only after checksum and terms review. Do not remove it in the same PR as camera refactoring.
6. **No legacy-video recovery assumption:** the branch named `archive/hardcoded-demo-before-rebuild` points to the same commit as `main` (audit Appendix A). It contains no recoverable previous demo asset tree.
7. **Future avatar assets:** accept no avatar, signing clip, or generated motion asset without signer/performer consent, ownership/license, versioned gloss/semantic alignment, and an evaluation protocol. A visual avatar is not evidence of linguistic correctness.

## 6. Small, reversible pull requests

| PR | Scope and reversible change | Must pass before merge |
|---|---|---|
| PR-0: Evidence freeze | Add this plan; record current commit, asset hashes, dataset checksums, and existing result hashes. No runtime behavior changes. | `python -m pytest -q`; `python RUN_ME.py --check`; documentation link check. |
| PR-1: Claim and UI labelling | Mark Research Scenario as simulation; make UI status distinguish camera, landmark extractor, model, and demo; correct docs to distinguish synthetic fixtures from real data. | Existing pytest suite; scripted UI import/smoke; `python evaluation/run_policy_evaluation.py` output schema check. |
| PR-2: Baseline quarantine | Introduce `baselines/`; move scenario runner/fixtures, synthetic generator, and 425 arrays using `git mv`; update paths without changing bytes. | Existing pytest suite; policy simulator snapshot test; data count = 425; file checksums unchanged. |
| PR-3: Dataset manifests | Add schemas/manifests and make preparation/training read them; preserve legacy read path temporarily. | Existing pytest suite; manifest validation; signer-disjoint split test; legacy CLI smoke. |
| PR-4: Honest recognizer evaluation | Replace full-set evaluator with held-out-manifest evaluator; remove local absolute result path; regenerate only clearly labelled reports. | Existing pytest suite; overlap-rejection test; deterministic evaluation fixture test; report schema test. |
| PR-5: Model and confidence contract | Add model manifest plus calibration artifact; make extraction failure non-predictive; maintain old checkpoint loader behind an explicit `legacy-synthetic` flag. | Existing pytest suite; checkpoint/schema mismatch rejection; extraction-failure test; calibration metrics test. |
| PR-6: Translation/TTS provider boundary | Move dictionaries/templates to baseline fixtures; add semantic-representation interface; make neural translation/server TTS extras and report fallback. | Existing pytest suite; ten-language fixture tests; provider-unavailable test; browser TTS rendering test. |
| PR-7: Dependency and asset cleanup | Remove unused imports/dependencies, delete unused hand landmarker only after fresh-clone verification; establish notices/LICENSE. | Existing pytest suite in a clean minimal environment; UI capture smoke; `RUN_ME.py --check`; asset download checksum test. |
| PR-8: Future reverse-direction foundation | Add interface-only ASR → semantic representation → sign-generation → avatar contracts and fixtures; no capability claim. | Existing pytest suite; contract tests using fixtures; no production UI route until a licensed/evaluated generator exists. |

Every PR must be independently revertible with a single `git revert` and must avoid mixing file moves with behavioral changes where practical.

## 7. Rollback strategy

1. Before PR-1, tag the audited state (for example `pre-migration-audit-79450a0`) and archive release checksums for `models/`, `data/processed/`, and `results/`.
2. Create each PR branch from the previous tagged/merged commit and merge only after its table-row gates pass.
3. For a regression, revert that one merged PR; do not reset shared history and do not restore generated data manually.
4. The branch `archive/hardcoded-demo-before-rebuild` is a **name only**, not an independent legacy snapshot: audit §1 and Appendix A show it points to `79450a0`, the same commit as `main`. It can serve as a convenient reference/tag for the current audited baseline but **cannot** restore a missing hard-coded video demo.
5. If a moved baseline must be restored temporarily, revert the `git mv` PR or use `git restore --source=<pre-migration-tag> -- <explicit paths>` in a new recovery PR. Do not cherry-pick undocumented binaries or raw videos.
6. Roll back model changes as an atomic set: checkpoint, model manifest, labels, preprocessing schema, calibration artifact, and report. A checkpoint-only rollback is invalid.

## 8. Mandatory test matrix after every PR

Run the full existing suite plus the applicable PR-specific checks:

```bash
python -m pytest -q
python RUN_ME.py --check
```

Current test files that must continue to pass until superseded by equivalent tests are:

- `tests/test_uncertainty.py`
- `tests/test_clarification.py`
- `tests/test_dialogue.py`
- `tests/test_smoke.py`
- `tests/test_policy.py`

Additional invariant tests introduced during migration must cover: no train/test manifest overlap; no model load on schema/hash mismatch; extraction failure cannot emit a normal sign prediction; all baseline scenarios are excluded from production UI; no local absolute paths/secrets in generated results; correct labelling of synthetic fixtures; and provider fallback/absence behavior.

## 9. Acceptance criteria for the target architecture

- Production ISL-to-text uses a data/model/calibration manifest and reports confidence provenance.
- Clarification consumes calibrated candidates, not authored scenario probabilities.
- Production multilingual output is explicitly identified as template, translation provider, or TTS provider; supported language behavior and fallbacks are testable.
- Baselines are isolated from production imports and UI routes.
- Speech/text-to-ISL remains unavailable until ASR, semantic representation, sign generation, avatar assets, licensing, consent, and linguistic/user evaluation are implemented and documented.
- Documentation makes no accuracy, real-time, language-coverage, avatar, bidirectional-translation, or commercial-use claim beyond the corresponding reproducible evidence.
