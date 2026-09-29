# ClarifySign — Current Repository Audit

**Scope:** Read-only inspection of GitHub repository [SrishBansal/ClarifySign](https://github.com/SrishBansal/ClarifySign) as checked out locally at `/Users/srishbansal/Desktop/ClarifySign_FINAL_SUBMISSION`.

| Field | Value |
|---|---|
| Remote | `https://github.com/SrishBansal/ClarifySign.git` |
| Default branch | `origin/main` |
| Audited commit | `79450a025cc640eaddb3e06ffa17db74b7c067e7` (“Initial commit”) |
| Parent commit | `545ffba` — contains **only** `README.md` |
| Working tree | Clean relative to `HEAD` |
| Local extra branches | `feat/clarifyisl-rebuild`, `archive/hardcoded-demo-before-rebuild` — both point at the **same** commit as `main` (the older hardcoded-demo tree is **not** retained as a distinct git object) |
| Audit date | 2026-09-29 |
| This file | New documentation only. No other files were edited. |

---

## 1. Full tree

No Jupyter notebooks exist. No `.mp4` / `.gif` / `.webm` / raster UI assets exist. Binary checkpoints and 425 processed `.npy` sequences **are** committed. Git LFS is **not** configured (no `.gitattributes`).

`data/raw/` is gitignored (`.gitignore` lines 3–4). The working copy may contain extra starter `.npy` under `data/raw/include_shopkeeper_starter/` (425 files observed locally); those are **not** in Git.

### 1.1 Source, config, launchers, tests, docs (every non-`.npy` tracked file)

```text
ClarifySign/
├── .env.example
├── .gitignore
├── AUDIT.md
├── CLARIFYSIGN_FINAL_REVIEW.md
├── README.md
├── RUN_ME.py
├── RUN_WINDOWS.bat
├── app.py
├── config.py
├── requirements.txt
├── research_spec.md
├── core/
│   ├── __init__.py
│   ├── clarification.py
│   ├── dialogue.py
│   ├── infogain.py
│   ├── landmarks.py
│   ├── model.py
│   ├── recognizer.py
│   ├── speech.py
│   ├── translation.py
│   └── uncertainty.py
├── training/
│   ├── __init__.py
│   ├── download_dataset.py
│   ├── evaluate_recognizer.py
│   ├── prepare_dataset.py
│   └── train.py
├── scripts/
│   ├── download_include.py          # 8-line wrapper → training.download_dataset
│   ├── prepare_dataset.py           # 8-line wrapper → training.prepare_dataset
│   ├── train.py                     # 8-line wrapper → training.train
│   └── policy_stress_test.py        # 9-line wrapper → evaluation.run_policy_evaluation
├── evaluation/
│   ├── run_policy_evaluation.py
│   └── scenarios.json
├── tests/
│   ├── test_clarification.py
│   ├── test_dialogue.py
│   ├── test_policy.py
│   ├── test_smoke.py
│   └── test_uncertainty.py
├── models/
│   ├── .gitkeep
│   ├── isl_bilstm.pt
│   ├── labels.json
│   ├── preprocessing_config.json
│   ├── training_metadata.json
│   ├── holistic_landmarker.task
│   └── hand_landmarker.task         # committed; not referenced by Python
├── results/
│   ├── .gitkeep
│   ├── policy_evaluation_results.json
│   └── recognizer_evaluation.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── DATASET_SOURCES.md
│   ├── EVALUATION.md
│   ├── FINAL_SUBMISSION_README.md
│   ├── LIMITATIONS.md
│   ├── PROJECT_STATUS.md
│   ├── RUNBOOK.md
│   └── VIVA_EXPLANATION.md
└── data/
    ├── processed/.gitkeep
    └── processed/<17 class dirs>/<25 .npy each>   # 425 files; see §1.2
```

**Gitignored / not in Git (present or expected on disk):**

- `data/raw/` including `dataset_info.json`, `zenodo_record.json`, `include_shopkeeper_starter/` (`.gitignore` lines 3–4)
- `__pycache__/`, `.venv/`, `data/landmarks/`, `models/*.keras` (`.gitignore` lines 1–5)
- No `LICENSE` file anywhere in the tree
- No `.env` file (only `.env.example`)

### 1.2 Datasets (processed landmarks committed)

17 INCLUDE-named classes × 25 sequences = **425** NumPy arrays tracked in Git, each intended shape `(48, 225)`:

| Class directory | Tracked files | Naming |
|---|---|---|
| `data/processed/bank/` | 25 | `bank_Signer{A,B,C}_NNN.npy` |
| `data/processed/biglarge/` | 25 | same pattern |
| `data/processed/black/` | 25 | |
| `data/processed/blue/` | 25 | |
| `data/processed/cellphone/` | 25 | |
| `data/processed/good/` | 25 | |
| `data/processed/hello/` | 25 | |
| `data/processed/hot/` | 25 | |
| `data/processed/new/` | 25 | |
| `data/processed/pen/` | 25 | |
| `data/processed/red/` | 25 | |
| `data/processed/shoes/` | 25 | |
| `data/processed/smalllittle/` | 25 | |
| `data/processed/storeorshop/` | 25 | |
| `data/processed/thankyou/` | 25 | |
| `data/processed/tshirt/` | 25 | |
| `data/processed/white/` | 25 | |

Signer split encoded in filenames: SignerA indices `{000,003,006,009,012,015,018,021,024}` (9), SignerB `{001,004,…,022}` (8), SignerC `{002,005,…,023}` (8).

These arrays are produced by `training/download_dataset.py` `generate_calibrated_starter_dataset()` (lines 127–239): **procedural sine/Gaussian kinematics**, not MediaPipe extraction from INCLUDE videos.

### 1.3 Tests and documentation inventory

| Tests | 27 `def test_*` functions: uncertainty 10, clarification 7, dialogue 5, smoke 3, policy 2 |
| Docs | 9 files under `docs/` plus root `README.md`, `AUDIT.md`, `CLARIFYSIGN_FINAL_REVIEW.md`, `research_spec.md` |

---

## 2. Application entry points

| Entry | Path | Lines | What it does |
|---|---|---|---|
| Primary CLI | `RUN_ME.py` | 93–151 (menu), 153–187 (`main`), 190–191 | Interactive menu and flags `--check/--download/--prepare/--train/--app/--eval/--test/--all` |
| Windows | `RUN_WINDOWS.bat` | 1–7 | `python RUN_ME.py` |
| Web UI | `app.py` | 1–47 imports; 49–54 `st.set_page_config`; 286–742 Streamlit body | `streamlit run app.py` (also `RUN_ME.py` 139–141, 173–174) |
| Train | `training/train.py` | `main` via argparse | Called from `RUN_ME.py` 137, 172 |
| Download | `training/download_dataset.py` | 242– | Called from `RUN_ME.py` 121–127, 168, 181 |
| Prepare | `training/prepare_dataset.py` | | `RUN_ME.py` 129, 170 |
| Policy eval | `evaluation/run_policy_evaluation.py` | | `RUN_ME.py` 143, 176 |
| Tests | `pytest` | | `RUN_ME.py` 145, 178 |
| Compat shims | `scripts/*.py` | 1–8/9 | Re-export `main` from `training/` or `evaluation/` |

There is **no** Flask/FastAPI/Next.js server. The product UI is Streamlit.

`config.py` lines 18–20 create `data/raw`, `data/processed`, `models`, `results` on import.

---

## 3. Hard-coded word-to-video, word-to-sign, sentence-to-sign, and language mappings

### 3.1 Word-to-video / sentence-to-sign video lookup

**None.** Repository search found no `.mp4`/`.gif`/`.webm` assets and no dict mapping glosses to video URLs or files.

`training/prepare_dataset.py` (video extensions at ~lines 54–59) can **extract** landmarks from videos if the user downloads them; it is not a playback dictionary.

### 3.2 Word-to-class (recognizer vocabulary)

`config.py` lines 90–109 `SHOPKEEPER_CLASSES` — 17 English INCLUDE-style labels.

Duplicated in:

- `models/labels.json` lines 1–19 (JSON array, same 17 strings)
- `models/preprocessing_config.json` lines 7–25 `classes_prepared`
- `models/training_metadata.json` lines 6–24 `classes`

These are **classifier class names**, not a sign-animation library.

### 3.3 Language code maps

`config.py` lines 38–50 `LANGUAGES`: Hindi/Marathi/Bengali/Gujarati/Tamil/Telugu/Kannada/Malayalam/Punjabi/Odia → Flores-200 / NLLB codes (`hin_Deva`, …, `ory_Orya`).

`core/speech.py` lines 11–23 `SPEECH_LANG_CODES` (gTTS / short BCP-47).  
`core/speech.py` lines 25–37 `WEB_SPEECH_VOICE_TAGS` (`hi-IN`, …, `or-IN`, `en-IN`).

`app.py` lines 349–351 UI language list adds `"English"` on top of the ten Indic names.

### 3.4 Word / concept → native-script phrase (offline “translation”)

All in `core/translation.py`:

| Structure | Lines | Role |
|---|---|---|
| `VERIFIED_SEMANTIC_DATABASE` | 20–253 | Concept key → 10-language **word/short phrase** (`shirt`, `tshirt`, `shoes`, `hat`, colors, sizes, `bank`, `storeorshop`, `cellphone`, `pen`, `hello`, `good`, `new`, `thankyou`) |
| `PHRASE_TEMPLATES` | 256–305 | Shopkeeper-facing templates (`ask_color_blue`, `ask_color_black`, `ask_tshirt`, `ask_shoes`). **Never referenced** by other Python modules. |
| `CONTEXTUAL_REALIZATIONS` | 309–538 | Concept → full first-person **sentences** in 10 languages (includes extra keys `shirt`, `hat` not in `SHOPKEEPER_CLASSES`) |
| Compound Hindi/Marathi/Bengali/Tamil color+garment strings | 586–608 | Inline dicts inside `translate_semantic` |
| Template wrappers around dictionary words | 620–639 | `"मुझे {native_word} चाहिए।"` etc. |

Neural fallback: `MultilingualTranslator.__init__` default `facebook/nllb-200-distilled-600M` (line 548); `try_load_neural` lines 555–571 uses `HF_TOKEN`; generate path lines 641–650.

### 3.5 Synthetic evaluation “signs”

`evaluation/scenarios.json` lines 1–~147: ten scenarios inject **hand-written** `(label, probability)` lists, including labels **outside** the 17-class model (`green`, `shirt`, `medium`, etc.). This is a policy simulator, not video mapping.

---

## 4. Static gesture / video / animation assets and references

| Asset | In Git? | Referenced by |
|---|---|---|
| Gesture **videos** | No | — |
| UI images / Lottie / GLB | No | — |
| Google Fonts CSS | External | `app.py` line 59 `@import url('https://fonts.googleapis.com/css2?family=Inter...')` |
| Camera overlay | Drawn in code | `app.py` lines 440–442 `cv2.rectangle` on live frames |
| `models/holistic_landmarker.task` (~13 MB) | Yes | `config.py` line 27; `core/landmarks.py` lines 26–38, 55–70 |
| `models/hand_landmarker.task` (~7.5 MB) | Yes | **No Python import.** Mentioned only in `AUDIT.md` (~line 97) |
| Processed `.npy` | Yes (425) | `training/train.py`, `training/evaluate_recognizer.py`, `training/prepare_dataset.py` |
| `isl_bilstm.pt` | Yes | `config.py` 23; `core/recognizer.py`; `app.py` 292; `RUN_ME.py` 77–80 |

Live camera is **not** a static asset: WebRTC (`app.py` 423–462) or `st.camera_input` fallback (478–498).

---

## 5. ML models, checkpoints, inference, dependencies, keys, endpoints

### 5.1 Checkpoints and bundles

| File | Role |
|---|---|
| `models/isl_bilstm.pt` | Project BiLSTM weights (`ISLBiLSTM` in `core/recognizer.py` 28–85; load `ISLRecognizer` 88–120+) |
| `models/holistic_landmarker.task` | MediaPipe Holistic Tasks bundle |
| `models/hand_landmarker.task` | MediaPipe Hands bundle, unused |
| `models/labels.json` | Class index order |
| `models/preprocessing_config.json` | Declares `processed_samples_count`: 425 |
| `models/training_metadata.json` | Claims test top-1/top-3 = 1.0, 9.01 s train on MPS |

`research_spec.md` line 16 and `docs/PROJECT_STATUS.md` lines 7–8 still mention `models/isl_bilstm.keras` — **that file is not in the repo** (and `models/*.keras` is gitignored).

### 5.2 Inference / training scripts

| Script | Lines (approx.) | Function |
|---|---|---|
| `core/landmarks.py` | 87–118 `__call__` | Per-frame 225-D features; zeros if detector missing (94–96) |
| `core/recognizer.py` | `predict` | Softmax over 17 classes |
| `app.py` | 444–475, 478–498 | Buffer frames → `resample_sequence` → `recognizer.predict` |
| `training/train.py` | 425 lines | Fit BiLSTM on `data/processed` |
| `training/evaluate_recognizer.py` | 29–70 | Loads **every** `data/processed/**/*.npy` (not held-out SignerC only) |
| `training/prepare_dataset.py` | video + npy ingest | |
| `core/translation.py` 555–650 | Optional HF seq2seq | Not the default path for known concepts |

### 5.3 Python dependencies (`requirements.txt` lines 1–16)

`streamlit`, `opencv-python`, `mediapipe`, `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `gTTS`, `transformers`, `sentencepiece`, `pytest`, `matplotlib`, `Pillow`, `streamlit-webrtc`, `av`.

### 5.4 API keys / env

`.env.example` lines 1–18:

- `HF_TOKEN=` empty placeholder (lines 2–3); consumed in `core/translation.py` 561–565
- Path/device and policy hyperparameters (`MODEL_PATH`, `CONFIDENCE_THRESHOLD`, …)

**No live API keys, tokens, or `.env` committed.**

### 5.5 Cloud / network endpoints

| URL / host | File:lines | Use |
|---|---|---|
| `https://zenodo.org/api/records/4010759` | `training/download_dataset.py` 35, 43 | INCLUDE metadata |
| `https://zenodo.org/records/4010759` | same file 8, 53–54, 74–75, 231; docs | Citation |
| `https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master` | `training/download_dataset.py` 36 | **Assigned, never used** |
| Zenodo file `links.self` | `download_official_category` 101–113 | Category ZIP download |
| `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task` | `core/landmarks.py` 28, 38 | Download holistic `.task` if missing |
| `stun:stun.l.google.com:19302` | `app.py` 428–429 | WebRTC ICE |
| fonts.googleapis.com | `app.py` 59 | Fonts |
| Hugging Face Hub | `transformers` `from_pretrained` | Optional NLLB weights |
| Google Translate TTS | `gTTS` in `core/speech.py` 106–113 | Server-side MP3 (imported in `app.py` 46 but **never called** in `app.py`) |
| Streamlit Community Cloud | `docs/RUNBOOK.md` ~87 | Documented deploy target, not a code URL |

---

## 6. Dataset sources, download locations, licenses, Git presence

### 6.1 Claimed official corpus (INCLUDE)

Documented in `training/download_dataset.py` 5–11 and `docs/DATASET.md` 3–16:

- **Name:** INCLUDE (AI4Bharat / IIT Madras)
- **Download:** Zenodo record `4010759` — `https://zenodo.org/records/4010759`
- **GitHub:** `https://github.com/AI4Bharat/INCLUDE` (`docs/DATASET.md` line 14)
- **License claimed:** CC-BY-4.0
- **Scale claimed:** 4,292 videos, 263 classes, ~50 GB (`docs/DATASET.md` 22–26)

**Raw INCLUDE videos are not in Git.** `.gitignore` line 3 excludes `data/raw/`.

### 6.2 What is actually committed

425 **synthetic** landmark sequences plus starter generator (`training/download_dataset.py` 127–239). Starter metadata writes `"license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)"` and `"source_reference": "https://zenodo.org/records/4010759"` (lines 221–231) even though samples are **not** INCLUDE video frames.

`docs/DATASET.md` 79–84 describes this starter as “calibrated landmark” data for offline/viva use.

### 6.3 Mentioned but not implemented

`docs/DATASET_SOURCES.md` lines 6–7: **iSign / ISLTranslate** (~31k sentence pairs) — citation only; no downloader, no data files.

### 6.4 Evaluation artifacts

`results/recognizer_evaluation.json` — `dataset_path` line 2 is a **machine-local absolute path** (`/Users/srishbansal/Downloads/ClarifySign_FINAL_SUBMISSION/data/processed`). `total_evaluated_samples` 425 with `top1_accuracy` 1.0 (lines 3–23) matches **full processed set**, not the 136-sample test split in `training_metadata.json` lines 26–28.

`evaluation/scenarios.json` — synthetic; `"synthetic": true` on each scenario (e.g. lines 15, 29).

---

## 7. Claims in README / documentation

Citations are to the files as they exist at commit `79450a0`.

### 7.1 Novelty

- `README.md` 12–19: “ClarifySign introduces an interactive, clarification-driven communication protocol” (camera, own BiLSTM, entropy/EIG, dialogue, 10 languages).
- `docs/ARCHITECTURE.md` 35–40: **does not** claim first ISL translator; novelty is “active, information-theoretic clarification”.
- `docs/FINAL_SUBMISSION_README.md` 4–14: “not an application wrapper”; contribution is uncertainty-aware clarification.
- `CLARIFYSIGN_FINAL_REVIEW.md` ~11+: retail communication / not greedy top-1.

### 7.2 Accuracy

- `models/training_metadata.json` 38–43: `test_top1_accuracy` / `test_top3_accuracy` **1.0**, `test_loss` 0.0014, train 9.01 s.
- `results/recognizer_evaluation.json` 23–24: top-1/top-3 **1.0** on **425** samples.
- `docs/EVALUATION.md` 27–34: “100.00%” on 136 Signer C samples; 36–40 caveats closed vocab / landmarks.
- `README.md` 169–171: “No Fake Accuracy” / measured from execution.
- Tension: `research_spec.md` 16–19 and `docs/FINAL_SUBMISSION_README.md` 16–17 say not to bundle a fabricated checkpoint / invent accuracy; the repo **does** bundle `isl_bilstm.pt` and 100% metrics on **synthetic** kinematics.

### 7.3 Real-time performance

- `app.py` 1–5, 49–50, 417–419, 512–514: “Real-Time”, “LIVE”, “Continuous real-time tracking”.
- `README.md` 147–148: B4 “**0.04 ms decision latency**”.
- `docs/EVALUATION.md` 75–89: splits MediaPipe 15–30 ms/frame vs policy 0.04 ms; NLLB 250–600 ms; gTTS 300–800 ms.
- `results/policy_evaluation_results.json` B4 `avg_latency_ms` 0.3398 (lines 80) vs README table 0.04 ms — **numbers disagree**.
- WebRTC buffer eval gated by 2.0 s (`app.py` 467), not continuous per-frame classification.

### 7.4 Language coverage

- `README.md` 19, 152–165: **10 official Indian languages** + sample Hindi…Odia strings.
- `config.py` 38 comment: “10 Official Indian Languages + English”.
- `docs/LIMITATIONS.md` 43–48: default is dictionary templates; NLLB optional; Odia TTS may fall back to Hindi (`core/speech.py` 21, 111–113).
- Coverage is **17 retail concepts + canned sentences**, not open-domain ISL or 22 scheduled languages.

### 7.5 Commercial / field use

- `README.md` 3–4, 10: retail/shopkeeper environments.
- `docs/LIMITATIONS.md` 54–55: **no** human-subject / deaf–shopkeeper field study.
- `docs/FINAL_SUBMISSION_README.md` 20–22: INCLUDE is isolated-sign, Chennai collection, not full regional ISL.
- No commercial-license grant for **this** repository (no `LICENSE` file). INCLUDE’s CC-BY-4.0 is cited for the **upstream** dataset.

### 7.6 Policy benchmark (README vs JSON)

`README.md` 128–148 table: B4 Success **100.0%**, IG **1.50b**, latency **0.04ms**.

`results/policy_evaluation_results.json` 66–80 (`B4_rho95`): success **90.0%**, `avg_information_gain` **1.157**, `avg_latency_ms` **0.3398**. Same file’s B2/B3 remain 100% (lines 31–51).

`docs/EVALUATION.md` 50–56 still prints B4 **100.0%** and 1.50 bits — **out of date vs JSON**.

`README.md` 48: “27 Unit & Integration Tests” — matches 27 `test_*` functions.

---

## 8. UI that is a static / simulated demo vs model-backed inference

| UI piece | File:lines | Backed by model? |
|---|---|---|
| **Demo Mode / Research Scenario** | `app.py` 320–321, 368–401 | **No.** Loads `evaluation/scenarios.json` and sets `latest_candidates` from authored tuples (394–397). Bypasses camera and BiLSTM. |
| Welcome empty-state copy | `app.py` 561–571 | Static HTML |
| CSS / branding / “LIVE” badge | `app.py` 57–283, 411–419 | Static; badge does not reflect model health |
| Language `<select>` | 348–354 | Controls dictionary lookup language |
| Clarification buttons | 627–661 | Policy + user click; in demo mode posterior is synthetic |
| Shopkeeper text form | 667–681 | Typed text, **not** ISL recognition |
| Research expander metrics | 688–742 | Uses whatever `latest_candidates` is (live or injected) |
| WebRTC / `st.camera_input` | 423–498 | **Yes, if** `ISLRecognizer.available` and landmarks non-zero. If MediaPipe fails, `HolisticExtractor` returns zeros (`core/landmarks.py` 94–96) → garbage/zero features, not a honest “no hands” UI |
| Browser TTS button | `get_browser_speech_html` via 611–613 | Speaks **dictionary** text, not a sign video |
| gTTS `synthesize_audio_bytes` | imported `app.py` 46 | **Unused** in the UI |

Live path is model-backed in structure; **Demo Mode is a static policy demo**. Translation of committed labels is dictionary-backed (`verified_multilingual_semantic`), not NLLB, for in-vocab concepts (`core/translation.py` 610–614).

---

## 9. File-by-file recommendation

Exactly one of: `KEEP`, `REFACTOR`, `MOVE_TO_BASELINE`, `REMOVE`, `REPLACE`.

| Path | Rec | Why |
|---|---|---|
| `RUN_ME.py` | KEEP | Real launcher; fix README “27 tests” if tests change |
| `RUN_WINDOWS.bat` | KEEP | Thin Windows entry |
| `app.py` | REFACTOR | Dead imports (lines 45–46); Demo Mode vs live mixed; WebRTC `session_state` mutation from worker is fragile; label UI “LIVE” honestly |
| `config.py` | KEEP | Single source for classes/hyperparams |
| `.env.example` | KEEP | Documents `HF_TOKEN` without secrets |
| `.gitignore` | REFACTOR | Ignore `data/processed/*.npy` or document why 425 arrays are committed; ignore `models/*.task` if re-downloaded; add `.env` |
| `requirements.txt` | KEEP | Pin versions before submission freeze |
| `research_spec.md` | REPLACE | Still forbids bundling a checkpoint (`16–19`) while `isl_bilstm.pt` exists; keras reference is stale |
| `core/__init__.py` | KEEP | Package exports |
| `core/landmarks.py` | KEEP | Real MediaPipe path + documented zero fallback |
| `core/recognizer.py` | KEEP | Actual `ISLBiLSTM` + inference |
| `core/model.py` | KEEP | Compat alias |
| `core/infogain.py` | KEEP | Noisy-answer IG implementation |
| `core/clarification.py` | REFACTOR | Adapter over infogain; keep tests working; `docs/LIMITATIONS.md` 32–36 still describes **perfect-oracle** IG |
| `core/uncertainty.py` | KEEP | Thin wrapper |
| `core/dialogue.py` | KEEP | Decay γ=0.85 at line 50 |
| `core/translation.py` | REFACTOR | Drop unused `PHRASE_TEMPLATES` (256–305) or wire them; do not call dictionary “neural”; extra keys `hat`/`shirt` |
| `core/speech.py` | KEEP | Web Speech used; gTTS optional |
| `training/__init__.py` | KEEP | |
| `training/download_dataset.py` | REFACTOR | Unused `INCLUDE_GITHUB_RAW` (line 36); do **not** stamp CC-BY-4.0 + Zenodo URL on synthetic npy (221–231) |
| `training/prepare_dataset.py` | KEEP | Video/npy ingest |
| `training/train.py` | KEEP | Real training; signer-split warning ~135 |
| `training/evaluate_recognizer.py` | REFACTOR | Currently scores **all** 425 files (44–49); must evaluate **test split only** and stop writing 100% on train+val |
| `scripts/download_include.py` | KEEP | Shim |
| `scripts/prepare_dataset.py` | KEEP | Shim |
| `scripts/train.py` | KEEP | Shim |
| `scripts/policy_stress_test.py` | KEEP | Shim |
| `evaluation/run_policy_evaluation.py` | KEEP | Honest provenance strings; align README with this output |
| `evaluation/scenarios.json` | MOVE_TO_BASELINE | Synthetic policy fixtures, not recognizer eval |
| `tests/test_uncertainty.py` | KEEP | |
| `tests/test_clarification.py` | KEEP | |
| `tests/test_dialogue.py` | KEEP | |
| `tests/test_smoke.py` | KEEP | |
| `tests/test_policy.py` | REFACTOR | Two one-line tests (lines 1–3); expand or merge into `test_clarification.py` |
| `models/isl_bilstm.pt` | REPLACE | Retrain on **real INCLUDE** landmarks or label explicitly as synthetic-only toy weights |
| `models/labels.json` | KEEP | |
| `models/preprocessing_config.json` | KEEP | |
| `models/training_metadata.json` | REPLACE | 100% / 9 s on synthetic sine classes is not a scientific test number |
| `models/holistic_landmarker.task` | KEEP | Required for landmarks; confirm MediaPipe redistribution terms |
| `models/hand_landmarker.task` | REMOVE | Unreferenced ~7.5 MB |
| `models/.gitkeep` | KEEP | |
| `data/processed/.gitkeep` | KEEP | |
| `data/processed/**/*.npy` (425 files) | MOVE_TO_BASELINE | Synthetic kinematics; do not present as INCLUDE videos. Prefer generate-on-setup or a clearly named `baseline_synthetic/` |
| `results/.gitkeep` | KEEP | |
| `results/policy_evaluation_results.json` | REFACTOR | Keep JSON; **replace** README/EVALUATION tables that still show B4 100% / 1.50 bits |
| `results/recognizer_evaluation.json` | REPLACE | Full-set 100% + local absolute `dataset_path` (line 2) |
| `README.md` | REFACTOR | Lines 128–148 contradict JSON; line 181 stray `# ClarifySign`; overstates live INCLUDE training |
| `AUDIT.md` | REFACTOR | Overlaps this audit; trim or point here; WebRTC vs `st.camera_input` description is stale vs `app.py` 423–478 |
| `CLARIFYSIGN_FINAL_REVIEW.md` | REFACTOR | Align policy numbers with `results/policy_evaluation_results.json` |
| `docs/ARCHITECTURE.md` | KEEP | Pipeline description is accurate at a high level |
| `docs/DATASET.md` | REFACTOR | INCLUDE facts vs “shopkeeper classes curated from INCLUDE” vs synthetic starter must be inseparable |
| `docs/DATASET_SOURCES.md` | KEEP | Short; iSign is future-work only |
| `docs/EVALUATION.md` | REPLACE | Section 3 table (50–56) and ablations (67–71) do not match committed JSON |
| `docs/FINAL_SUBMISSION_README.md` | REFACTOR | “Do not invent accuracy” vs bundled 100%; keras-era “ZIP cannot contain checkpoint” vs `isl_bilstm.pt` |
| `docs/LIMITATIONS.md` | REFACTOR | Strong honesty (17 signs, no field study); update EIG oracle paragraph (32–36) to match `core/infogain.py` |
| `docs/PROJECT_STATUS.md` | REMOVE | Stale keras / “must not present synthetic results” while results are committed |
| `docs/RUNBOOK.md` | REFACTOR | Cloud/`st.camera_input` story incomplete given WebRTC-first `app.py` |
| `docs/VIVA_EXPLANATION.md` | KEEP | Viva Q&A; sync IG assumptions with infogain.py |
| `docs/CURRENT_REPO_AUDIT.md` | KEEP | This file |
| Untracked `data/raw/**` | KEEP (local only) | Must stay gitignored; do not commit INCLUDE zips without license review |

---

## 10. Risks

### 10.1 Secrets

- No committed `.env` or non-empty `HF_TOKEN`.
- `results/recognizer_evaluation.json` line 2 leaks a **local filesystem username/path**.

### 10.2 Personally identifiable data

- No names, emails, or webcam recordings in Git.
- Synthetic “SignerA/B/C” are labels, not people.
- If INCLUDE videos are downloaded into `data/raw/`, those contain **identifiable signers**; they must not be committed (currently gitignored).

### 10.3 Copyrighted / third-party binaries

- MediaPipe `.task` files (Google) redistributed in `models/` without a project `LICENSE` or NOTICE.
- Inter/JetBrains Mono loaded from Google Fonts at runtime (`app.py` 59).
- NLLB-200 and gTTS are dependency licenses (Meta / Google), not bundled weights (NLLB downloaded on demand).

### 10.4 Missing licenses

- **No `LICENSE` for ClarifySign source.**
- INCLUDE CC-BY-4.0 is documented (`docs/DATASET.md` 15; `training/download_dataset.py` 9) but **does not** automatically license synthetic npy or this code.
- Starter metadata **copies INCLUDE’s CC-BY-4.0** onto generated arrays (`training/download_dataset.py` 230) — legally misleading.

### 10.5 Dataset redistribution

- 425 `.npy` in Git: if reviewers treat them as INCLUDE derivatives, redistribution/attribution is unclear because they are **not** extracted from Zenodo videos.
- Full INCLUDE (~50 GB) is correctly not vendored.

### 10.6 Misleading claims (high)

1. **100% ISL accuracy** on a 17-way sine-parameterized synthetic set (`training_metadata.json`, `recognizer_evaluation.json`, `docs/EVALUATION.md` 27–34) will not transfer to real INCLUDE or live webcam.
2. **`training/evaluate_recognizer.py` evaluates the training set** (all 425 files) while metadata talks about a 136-sample Signer C test.
3. **README policy table (128–148)** claims B4 100% success and 1.50 bits; **committed JSON** reports 90% and 1.157 bits for `B4_rho95`.
4. **“Our own trained neural recognizer” on INCLUDE** (`README.md` 14) vs generator that never reads INCLUDE frames.
5. **Real-time / LIVE** UI (`app.py` 417–419) vs 2-second commit throttle and possible zero-vector landmarks.
6. **10-language “translation engine”** is mostly `VERIFIED_SEMANTIC_DATABASE` / `CONTEXTUAL_REALIZATIONS`; NLLB is optional and unused for in-vocab keys.
7. **`docs/PROJECT_STATUS.md` / `research_spec.md`** contradict the bundled checkpoint.
8. **`LIMITATIONS.md` EIG** still describes H=0 on confirm; **`core/infogain.py` 1–27** says that formulation was replaced.
9. Scenario labels `green`/`shirt`/`medium` (`evaluation/scenarios.json` 10, 24, 37) are **not** model classes — fine for policy sim, easy to misread as recognizer output.
10. Public GitHub copy of MediaPipe models + synthetic data **without** a repo license complicates any commercial or course-reuse story.

---

## Appendix A — Git first commit

`545ffba` tree is solely `README.md` (same narrative as current README). There is **no** recoverable hardcoded word-to-video demo in git history at these two commits, despite the branch name `archive/hardcoded-demo-before-rebuild`.

## Appendix B — Test count

| File | `test_*` count |
|---|---|
| `tests/test_uncertainty.py` | 10 |
| `tests/test_clarification.py` | 7 |
| `tests/test_dialogue.py` | 5 |
| `tests/test_smoke.py` | 3 |
| `tests/test_policy.py` | 2 |
| **Total** | **27** |
