# ClarifySign — Technical Limitations & System Boundaries

This document provides a candid, transparent analysis of the current engineering and scientific boundaries of the ClarifySign prototype.

---

## 1. Vocabulary & Dataset Scope

- **Curated 17-Sign Subset**: The current model is trained on a 17-sign retail domain subset (`bank`, `biglarge`, `black`, `blue`, `cellphone`, `good`, `hello`, `hot`, `new`, `pen`, `red`, `shoes`, `smalllittle`, `storeorshop`, `thankyou`, `tshirt`, `white`). It does not recognize full Indian Sign Language (~3000+ words) or complex sentences.
- **Sample Size**: The dataset contains 425 total landmark sequences across 3 distinct signers (Signer A, B, C). While the signer-independent split confirms generalizability across these three signers, broader demographic variation (age, hand morphology, regional dialects) remains untested.
- **Isolated Sign Recognition**: The current pipeline recognizes single isolated sign gestures resampled to 48 frames. It does not perform continuous sign language recognition (CSLR) or automatic temporal boundary segmentation (spotting sign onset/offset in continuous streams).

---

## 2. Computer Vision & Landmark Extraction

- **MediaPipe Failure Modes**: Landmark extraction degrades under severe conditions:
  - Signer hands moving rapidly outside the camera field of view.
  - Severe occlusions (e.g. one hand crossing in front of the other at identical depth).
  - Low lighting or heavy motion blur where hand keypoints cannot be localized.
- **Static Facial Dynamics**: Current feature extraction focuses on upper body pose (33 keypoints) and hands (21 + 21 keypoints). Subtle facial expressions and mouthings (critical non-manual markers in ISL grammar) are not heavily modeled.

---

## 3. Probability Calibration

- **Temperature Scaling Parameter**: The model includes a configurable temperature hyperparameter $T$ to scale output logits ($p_i = \text{softmax}(z_i / T)$). However, this is **temperature-scalable probability output**, not an empirically calibrated model fitted via Platt scaling or temperature calibration on an out-of-distribution calibration set.
- **Heuristic Ambiguity Gate**: The ambiguity thresholds ($p_1 < 0.78$, margin $< 0.20$, entropy $> 1.25$ bits) were engineered for retail dialogue sensitivity and are not guaranteed to be globally optimal for all noise regimes.

---

## 4. Expected Information Gain (EIG) Simplifying Assumptions

- **Candidate Resolution Assumption**: In the current EIG formulation:
  $$\text{IG}(q) = H(Y|X) - \sum_a P(a|q, X) \cdot H(Y|X, q, a)$$
  When the user selects a candidate option $a = c_i$, the conditional entropy $H(Y|X, q, a=c_i)$ is assumed to be $0$ (the sign identity is fully resolved). When the user selects "Other / Neither", the residual entropy over unqueried classes is calculated. While mathematically well-behaved, this assumes the user never makes a mistake when clicking/tapping the clarification option.
- **Single-Turn Clarification**: The policy evaluates questions for a single immediate disambiguation turn rather than multi-step tree search (e.g., Monte Carlo Tree Search over infinite turn horizons).

---

## 5. Multilingual Translation & Speech

- **Translation Backends**:
  - The primary operational backend for the 17 vocabulary signs is the **Verified Semantic Database & Phrase Templates**, which provides deterministic, grammatically correct translations in 10 Indian languages.
  - Neural machine translation (e.g. HuggingFace NLLB-200) is supported as an optional backend for arbitrary sentences when internet access and API tokens are available. It is not the default for isolated retail concepts.
- **Speech Synthesis (TTS)**:
  - Client-side Web Speech API depends on the host browser and operating system's installed voices.
  - Server-side gTTS requires active internet access. For languages without native Google TTS voices (such as Odia), speech synthesis falls back to Hindi voice output.

---

## 6. Evaluation Scope

- **No Human-Subject Field Study**: All policy benchmark evaluations were conducted using a **deterministic controlled benchmark** with 10 synthetic scenario distributions. No clinical or real-world deaf customer / shopkeeper field trials have been conducted to date.
- **Simulation Assumption**: In the scenario benchmark, the simulated user responds accurately when their intended ground truth is present in the presented question options.
