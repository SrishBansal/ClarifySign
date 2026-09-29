# ClarifySign — System Architecture Documentation

## 1. Executive Summary

ClarifySign is an ambiguity-aware communication system designed for Indian Sign Language (ISL) interactions in retail/shopkeeper environments. 

Rather than prematurely committing to greedy Top-1 predictions when gestures exhibit perceptual or environmental ambiguity, ClarifySign implements a **clarification-driven communication protocol**:

```
Webcam Video Capture
         ↓
MediaPipe Holistic Skeletal Extraction (225 features)
         ↓
Temporal Normalization & Resampling (48 frames)
         ↓
Deep Bidirectional LSTM Neural Model (Our Trained ISL Engine)
         ↓
Softmax Candidate Probability Distribution P(Y|X)
         ↓
Information-Theoretic Uncertainty Engine (Top-1, Margin, Entropy)
         ↓
Expected Information Gain (EIG) Question Utility Optimization
         ↓
Customer-Facing Clarification Card (Refusal to Guess)
         ↓
Dialogue State Context Update (Temporal Decay & Resolution)
         ↓
Multilingual Translation Engine (10 Official Indian Languages)
         ↓
Speech Synthesis Output (Web Speech API + gTTS)
```

---

## 2. Core Research Novelty & Hypothesis

### Novelty Position
ClarifySign does **not** claim to be the first sign language translator or the first multilingual ISL system.

The core research contribution is the **active, information-theoretic clarification communication protocol**:
When ISL recognition is ambiguous, the system quantitatively computes the **Expected Information Gain (EIG)** of candidate clarification questions, balances this against conversational context and interruption cost, and asks a minimal, highly answerable question.

### Research Hypothesis
> *An information-gain-driven active clarification protocol significantly reduces **confidently-wrong** communication errors compared with greedy Top-1 commitment and fixed-confidence fallbacks, while maintaining low conversational overhead and interaction cost.*

---

## 3. Mathematical Foundations

### 3.1 Uncertainty Quantification
For an extracted sequence $X$ and gesture class distribution $P(Y=y|X)$ over $K$ candidate classes:

1. **Top-1 Confidence**:
   $$p_{(1)} = \max_y P(Y=y|X)$$

2. **Top-2 Competitive Margin**:
   $$\Delta = p_{(1)} - p_{(2)}$$
   A narrow margin indicates direct perceptual conflict between two dominant interpretations (e.g. Blue vs Black).

3. **Shannon Entropy (in bits)**:
   $$H(Y|X) = -\sum_{y \in \mathcal{Y}} P(Y=y|X) \log_2 P(Y=y|X)$$

4. **Normalized Entropy**:
   $$H_{\text{norm}}(Y|X) = \frac{H(Y|X)}{\log_2(K)}$$
   Bounded in $[0, 1]$, measuring distribution dispersion relative to the uniform distribution.

### 3.2 Multi-Criteria Ambiguity Detection
A gesture interpretation is flagged as **ambiguous** if any of the following triggers fire:
$$\text{Ambiguous}(X) \iff (p_{(1)} < \tau_{\text{conf}}) \lor (\Delta < \tau_{\text{margin}}) \lor (H(Y|X) > \tau_H)$$
Default calibrated parameters:
- $\tau_{\text{conf}} = 0.78$
- $\tau_{\text{margin}} = 0.20$
- $\tau_H = 1.25 \text{ bits}$

### 3.3 Expected Information Gain (EIG)
For a candidate clarification question $q \in \mathcal{Q}$ with answer options $A = \{a_1, a_2, \dots, a_m\}$:
$$IG(q) = H(Y|X) - \sum_{a \in A} P(a|q, X) \cdot H(Y|X, q, a)$$

**Assumption for Candidate-Choice Questions:**
- When the customer confirms a specific candidate sign $a_i = c_i$, ambiguity collapses to 0 ($H(Y|X, q, a_i) = 0$).
- When the customer indicates "Other / None of these", the remaining probability mass $p_{\text{rem}} = 1 - \sum_{j \in q} p_j$ is distributed across the remaining unqueried classes, yielding residual conditional entropy $H_{\text{rem}}$.
- Therefore:
  $$\mathbb{E}_a[H(Y|X, q, a)] = p_{\text{rem}} \cdot H_{\text{rem}}$$
  $$IG(q) = H(Y|X) - p_{\text{rem}} \cdot H_{\text{rem}}$$

### 3.4 Question Utility Optimization
We select the optimal question $q^*$ by maximizing a multi-attribute utility objective:
$$U(q) = IG(q) + \alpha \cdot \text{Context}(q) + \beta \cdot \text{Answerability}(q) - \lambda \cdot \text{Cost}(q)$$
$$q^* = \arg\max_{q \in \mathcal{Q}} U(q)$$

Where:
- $\text{Context}(q) \in [0, 1]$: Relevance of question candidates to active dialogue history (domain alignment, e.g. clothing colors vs payment).
- $\text{Answerability}(q) \in [0.1, 1.0]$: Cognitive ease of answering. A 2-choice question has higher answerability than a 5-choice question:
  $$\text{Answerability}(q) = \max(0.10, 1.0 - 0.12 \times (|A| - 2))$$
- $\text{Cost}(q)$: Interaction cost penalty for interrupting the customer:
  $$\text{Cost}(q) = \lambda \cdot (1.0 + 0.10 \times (|A| - 2))$$
- $\alpha = 0.30$, $\beta = 0.15$, $\lambda = 0.18$.

### 3.5 Clarification Decision Policy
- If $\text{Ambiguous}(X) = \text{False}$:
  $\rightarrow$ **COMMIT** top-1 candidate directly.
- If $\text{Ambiguous}(X) = \text{True}$:
  - If $U(q^*) \ge \tau_{\text{utility}}$ ($0.05$):
    $\rightarrow$ **CLARIFY** with question $q^*$.
  - Else:
    $\rightarrow$ **COMMIT** with low-confidence warning.

---

## 4. Subsystem Components

### 4.1 Feature Extraction (`core/landmarks.py`)
- Framework: MediaPipe Holistic Tasks (CPU Delegate, GPU disabled via `MEDIAPIPE_DISABLE_GPU=1` to guarantee macOS stability).
- Landmarks per frame:
  - Left Hand: 21 landmarks $\times$ 3 coords $(x, y, z) = 63$ features
  - Right Hand: 21 landmarks $\times$ 3 coords $(x, y, z) = 63$ features
  - Pose: 33 landmarks $\times$ 3 coords $(x, y, z) = 99$ features
  - Total frame feature vector: $63 + 63 + 99 = 225$ dimensions.
- Normalization: Zero-centered relative to root wrist and shoulder landmarks.
- Resampling: Temporal sequence linear index interpolation to exactly 48 frames.

### 4.2 Recognition Model (`core/recognizer.py`)
- Architecture: Deep Bidirectional LSTM (`ISLBiLSTM`) in PyTorch.
- Layers:
  1. Input: $(B, 48, 225)$
  2. Layer Normalization ($D=225$)
  3. Bidirectional LSTM (Hidden size 128, 2 layers, Dropout 0.30)
  4. Dual Temporal Pooling: Average Pooling + Max Pooling across time steps ($D=256$)
  5. Dense Projection Head: Linear(256, 128) + GELU activation + Dropout(0.30)
  6. Output Layer: Linear(128, $K$ classes)
  7. Softmax with Temperature Scaling $T=1.0$.
- Hardware Execution: Automatic device selection (Apple Silicon MPS / NVIDIA CUDA / CPU).

### 4.3 Dialogue State Machine (`core/dialogue.py`)
- Structure:
  - Turn memory: `Turn(speaker, raw, semantic, language, was_clarified, timestamp)`
  - Context vector: Dictionary `{concept: weight}` with exponential decay factor $\gamma = 0.85$ per turn.
  - Active domain tracker: Inferred domain (`clothing`, `color_selection`, `sizing`, `transaction`).
  - Resolution engine: Disambiguates customer input (numeric choice '1', text click, or gesture confirmation).

### 4.4 Multilingual Translation Engine (`core/translation.py`)
- Translates resolved semantic concepts into 10 official Indian languages:
  1. **Hindi** (`hin_Deva`)
  2. **Marathi** (`mar_Deva`)
  3. **Bengali** (`ben_Beng`)
  4. **Gujarati** (`guj_Gujr`)
  5. **Tamil** (`tam_Taml`)
  6. **Telugu** (`tel_Telu`)
  7. **Kannada** (`kan_Knda`)
  8. **Malayalam** (`mal_Mlym`)
  9. **Punjabi** (`pan_Guru`)
  10. **Odia** (`ory_Orya`)
- Dual-engine architecture:
  - **Neural Seq2Seq**: HuggingFace NLLB-200 (`facebook/nllb-200-distilled-600M`) or IndicTrans2.
  - **Verified Semantic Fallback**: Deterministic offline native translation engine covering all shopkeeper utterances, customer requests, and sign concepts.
  - Transparent labeling: Output explicitly indicates whether neural or semantic fallback was used.

### 4.5 Speech Output (`core/speech.py`)
- Client-side browser speech synthesis via Web Speech API (`window.speechSynthesis`) with BCP-47 language tags.
- Server-side audio synthesis using `gTTS` generating playable MP3 audio.
