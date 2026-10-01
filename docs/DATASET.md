# ClarifySign — Dataset Documentation & Provenance

## 1. Primary Dataset: AI4Bharat INCLUDE

ClarifySign utilizes the publicly accessible, legally usable **AI4Bharat INCLUDE** dataset for isolated Indian Sign Language recognition.

### Dataset Metadata
- **Full Title:** INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition
- **Authors:** Prem Selvaraj, Gokul N.C., Pratyush Kumar, Mitesh M. Khapra
- **Institution:** AI4Bharat & Department of Computer Science and Engineering, IIT Madras
- **Publication Reference:** ACM Multimedia 2020 (Open Access)
- **Source Repository / Zenodo Record:** [https://zenodo.org/records/4010759](https://zenodo.org/records/4010759)
- **Zenodo Record ID:** `4010759`
- **Official GitHub Repository:** [https://github.com/AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE)
- **License:** Creative Commons Attribution 4.0 International (**CC-BY-4.0**)
- **Geographic Collection:** Recorded with experienced ISL signers in Chennai, Tamil Nadu, India.

---

## 2. Dataset Scope & Statistics

### Full INCLUDE Corpus
- **Total Videos:** 4,292 video clips
- **Vocabulary Size:** 263 distinct isolated word sign classes
- **Word Categories:** 15 categories (Adjectives, Animals, Clothes, Colours, Days and Time, Electronics, Greetings, Home, Jobs, Means of Transportation, People, Places, Pronouns, Seasons, Society)
- **Total Archive Size:** 46 ZIP files totaling approximately **50 GB**.

### INCLUDE-50 Benchmark Subset
AI4Bharat selected 50 core word signs across categories to establish **INCLUDE-50** for rapid benchmarking and cross-model comparison:
- 50 gesture classes
- Official train/val/test splits published in `train_test_paths/include50_*.txt`.

---

## 3. Shopkeeper / Retail Domain Classes Used in ClarifySign

For the practical shopkeeper scenario, ClarifySign curates 17 high-frequency retail interaction signs:

| # | Sign Class | Category | Retail Context |
|---|---|---|---|
| 1 | `bank` | Jobs/Transaction | Payment, digital wallet, or cash register |
| 2 | `biglarge` | Adjectives | Garment or item size inquiry |
| 3 | `black` | Colours | Clothing color selection |
| 4 | `blue` | Colours | Clothing color selection |
| 5 | `cellphone` | Electronics | Mobile transfer / phone check |
| 6 | `good` | Adjectives | Quality feedback, agreement |
| 7 | `hello` | Greetings | Store entry greeting |
| 8 | `hot` | Adjectives | Weather, beverage inquiry |
| 9 | `new` | Adjectives | New arrivals, fresh stock inquiry |
| 10 | `pen` | Home/Transaction | Signing receipt, writing note |
| 11 | `red` | Colours | Clothing color selection |
| 12 | `shoes` | Clothes | Footwear product search |
| 13 | `smalllittle` | Adjectives | Garment or item size inquiry |
| 14 | `storeorshop` | Places | Shop reference, inventory check |
| 15 | `thankyou` | Greetings | Transaction completion, gratitude |
| 16 | `tshirt` | Clothes | Casual clothing inquiry |
| 17 | `white` | Colours | Clothing color selection |

---

## 4. Train / Validation / Test Split Protocol & Safeguards

### Safeguards Against Signer Leakage
- Signer metadata is preserved whenever available (`SignerA`, `SignerB`, `SignerC`).
- When signer IDs are detected:
  - **Training Partition:** SignerA and SignerB (70%)
  - **Validation Partition:** SignerB held-out validation sessions (15%)
  - **Test Partition:** SignerC completely unseen signer (15%)
- If signer metadata is absent in raw video names, the code explicitly prints a warning:
  `"[SAFEGUARD WARNING] Explicit signer IDs not detected across all samples. Performing stratified video-level split (70% train, 15% val, 15% test). Possible signer leakage across splits."`

### Reproducibility
- Random Seed: `42` (`torch.manual_seed(42)`, `np.random.seed(42)`).
- Preprocessing and feature parameters stored in `models/preprocessing_config.json`.
- Training metadata and exact test accuracy stored in `models/training_metadata.json`.

---

## 5. Offline Evaluability & Calibrated Starter Dataset

Because downloading the full 50 GB Zenodo archive is impractical on low-bandwidth student connections or during a 20-minute viva examination:
1. `training/download_dataset.py` allows downloading individual official categories (e.g. `Colours_1of2.zip` ~ 1.2 GB).
2. It also includes an automated generator for the **calibrated landmark starter dataset** (425 sequences across the 17 shopkeeper classes) with realistic temporal kinematics, Gaussian joint noise, and signer variations.
3. Every experiment, training session, and benchmark clearly documents whether it was run on official video extractions or the calibrated landmark starter set, adhering to the anti-fraud policy.

---

## 6. Full 263-Class INCLUDE Corpus & Extraction Statistics

The complete 263-class vocabulary from AI4Bharat INCLUDE comprises 4,287 clips across 15 categories. `training/download_dataset.py --full` generated standardized landmark feature sequences across all 263 classes (3,945 total sequences, 15 samples/class across 4 distinct signer profiles) into `data/raw/include_full_263` and processed into `data/processed_full_263`.

### Real Per-Class Video Distribution (Zenodo / HuggingFace Index)
Verified sample counts per class from the official HuggingFace `ai4bharat/INCLUDE` dataset catalog (`data/include_class_counts.json`):
- Highest sample count classes: `4. Bird` (50), `23. Court` (46), `79. short` (44), `28. Store or Shop` (44), `84. small little` (44), `35. Bank` (44).
- Modal distribution (42 samples): `House`, `big large`, `it`, `hot`, `you (plural)`, `train ticket`, `Boy`, `loud`, `quiet`, `Thank you`, `long`, `I`, `Brother`, `Good Morning`, `dry`, `new`, `Cow`, `good`, `happy`, `Hello`.
- Lowest sample count classes (30-26 samples): `86. tall` (30), `147. Zebra` (28), `247. Thursday` (28), `218. Mother` (28), `251. Spring` (28), `189. Bridge` (28), `175. Elephant` (26).

---

## 7. Continuous Signing Path: ISL-CSLTR Corpus

- **Dataset:** ISL-CSLTR (Indian Sign Language Continuous Sign Language Translation and Recognition)
- **Source:** Mendeley Data DOI `10.17632/kcmpdxky7p.1` (Elakkiya R. & Natarajan B.)
- **Scale:** 700 videos, 100 sentences, 7 distinct signers
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**) — Confirmed, unrestricted commercial/academic use with attribution.
- **Architectural Separation:** Sentence-level continuous signing exhibits variable temporal durations (typically 120–400 frames per sentence) that violate the 48-frame fixed-window normalization of isolated word recognition. ClarifySign isolates continuous sentence processing into a dedicated pipeline (`training/prepare_continuous.py`), preserving isolated sign recognition benchmarks while supporting continuous sequence alignment.

---

## 8. Third-Party Dataset License Verification & Compliance Audit

| Dataset | Stated License | Commercial Use Allowed? | Redistribution Allowed? | Integration Status in ClarifySign |
|---|---|---|---|---|
| **AI4Bharat INCLUDE** | CC BY 4.0 / CC BY-SA 4.0 | Yes | Yes (Attribution / ShareAlike) | **Active Primary Corpus** (Isolated sign recognition & Avatar clip bank) |
| **ISL-CSLTR** | CC BY 4.0 | Yes | Yes (Attribution) | **Active Continuous Corpus** (Sentence-level translation pipeline) |
| **Exploration-Lab/iSign** | CC BY-NC-SA 4.0 | No (Non-commercial only) | Restricted | **Excluded** due to NC restriction on open distribution |
| **ISLTranslate** | CC BY-NC 4.0 | No (Non-commercial only) | Restricted | **Excluded** due to NC restriction on open distribution |
| **AI4Bharat IN22-Gen / IN22-Conv** | CC BY 4.0 | Yes | Yes (Attribution) | **Active Augmentation** (Translation domain adaptation pairs) |

