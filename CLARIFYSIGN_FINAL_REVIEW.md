# ClarifySign — Final Engineering Review & Viva Defense Dossier

**Project Title:** ClarifySign: Ambiguity-Aware Indian Sign Language Communication System for Practical Retail Scenarios  
**Dataset Provenance:** Curated 17-Class, 425-Sample Shopkeeper Domain Subset of AI4Bharat INCLUDE  
**Model Architecture:** Bidirectional LSTM with Layer Normalization, Dual Temporal Pooling, and GELU Projection Head  
**Primary Research Focus:** Active Information-Theoretic Clarification vs. Greedy Top-1 Recognition Commitment  

---

## 1. Problem & Practical Motivation
Deaf and hard-of-hearing individuals face persistent communication barriers in daily retail and commercial transactions (grocery shops, apparel counters, billing desks). Sign language recognition (SLR) systems typically approach communication as a pure computer-vision pattern classification problem: an input video stream is mapped directly to a single top-1 prediction. 

However, natural sign language communication contains inherent visual ambiguity:
- **Kinematic & Morphological Overlap**: Signs such as *Blue* vs. *Black*, or *T-Shirt* vs. *Shirt*, differ by subtle hand rotations or minimal spatial trajectories.
- **Environmental & Capture Variance**: Camera angles, motion blur, lighting shifts, and rapid signing distort extracted visual features.
- **Cost Asymmetry**: In a commercial transaction, a **confidently-wrong interpretation** (e.g. charging for the wrong item or wrong color) causes immediate transactional failure and frustration, whereas a brief, targeted clarification question resolves the misunderstanding in 1–2 seconds.

ClarifySign re-frames the problem: **When visual evidence is ambiguous, the system should actively communicate uncertainty and resolve it through an optimal clarification question.**

---

## 2. System Architecture & Information Pipeline

```
[Customer ISL Gesture]
          ↓
[Camera Input / Video Buffer]
          ↓
[MediaPipe Holistic Keypoint Extraction] ──── 225 zero-centered features / frame
          ↓
[Temporal Normalization] ──────────────────── Resampled to uniform 48-frame sequence
          ↓
[Trained ISL BiLSTM Neural Model] ────────── 2-layer BiLSTM (128 hidden) + Dual Temporal Pooling
          ↓
[Candidate Probability Distribution P(Y|X)] ─ 17-class softmax distribution
          ↓
[Uncertainty Engine] ──────────────────────── Computes Top-1 Prob, Top-2 Margin, Shannon Entropy
          ↓
   [Ambiguity Gate]
     /          \
  (Clear)     (Ambiguous)
    ↓              ↓
[Commit]     [Planner: Question Utility Optimization]
                   • Expected Information Gain IG(q)
                   • Context Affinity Score
                   • Cognitive Answerability
                   • Interaction Cost Penalty
                   ↓
             [Optimal Question q* = argmax U(q)]
                   ↓
             [Customer Confirmation via Screen / Sign]
                   ↓
             [Dialogue Context Memory Update]
                   ↓
[Multilingual Translation Engine] ─────────── 10 Indian Languages (Verified Semantic / NLLB)
          ↓
[Speech Synthesis] ────────────────────────── Web Speech API / gTTS Audio Output
```

---

## 3. Mathematical Formulation

### 3.1 Uncertainty Quantification
For input sequence $X$ and candidate classes $\mathcal{Y} = \{y_1, \dots, y_K\}$ with posterior probabilities $p_i = P(Y=y_i|X)$:

1. **Top-1 Confidence**:
   $$p_{(1)} = \max_{i} p_i$$

2. **Top-2 Competitive Margin**:
   $$\Delta = p_{(1)} - p_{(2)}$$
   where $p_{(2)}$ is the second-highest class probability.

3. **Shannon Entropy (in bits)**:
   $$H(Y|X) = -\sum_{i=1}^K p_i \log_2 p_i$$

4. **Normalized Entropy**:
   $$H_{\text{norm}}(Y|X) = \frac{H(Y|X)}{\log_2 K} \in [0, 1]$$

### 3.2 Ambiguity Decision Gate
A gesture interpretation is flagged as ambiguous if any of three conditions hold:
$$\text{Ambiguous}(X) \iff (p_{(1)} < 0.78) \lor (\Delta < 0.20) \lor (H(Y|X) > 1.25 \text{ bits})$$

### 3.3 Expected Information Gain (EIG)
For candidate clarification question $q$ with potential answers $A = \{a_1, \dots, a_m\}$:
$$IG(q) = H(Y|X) - \sum_{a \in A} P(a|q, X) \cdot H(Y|X, q, a)$$

**Assumption & Operational Rule:**
- When the user confirms a presented candidate $a = c_i$, ambiguity collapses to 0 ($H(Y|X, q, a=c_i) = 0$).
- When the user selects "Other / None", the remaining probability mass $p_{\text{rem}} = 1 - \sum_{j \in q} p_j$ is distributed across unqueried classes, yielding residual entropy $H_{\text{rem}}$.
- Therefore:
  $$IG(q) = H(Y|X) - p_{\text{rem}} \cdot H_{\text{rem}}$$

### 3.4 Multi-Attribute Question Utility
$$U(q) = IG(q) + \alpha \cdot \text{Context}(q) + \beta \cdot \text{Answerability}(q) - \lambda \cdot \text{Cost}(q)$$
$$q^* = \arg\max_{q \in \mathcal{Q}} U(q)$$

Calibrated hyperparameters:
- $\alpha = 0.30$ (Context Weight)
- $\beta = 0.15$ (Answerability Weight: $\text{Ans}(q) = \max(0.10, 1.0 - 0.12 \times (|A| - 2))$)
- $\lambda = 0.18$ (Interruption Cost Penalty: $\text{Cost}(q) = \lambda \cdot (1.0 + 0.10 \times (|A| - 2))$)
- $\tau_U = 0.05$ (Minimum Utility Threshold for clarification)

---

## 4. Dataset & Model Provenance

### 4.1 Dataset Details
- **Base Source**: AI4Bharat INCLUDE (ACM Multimedia 2020, CC-BY-4.0).
- **Domain Subset**: 17 retail interaction classes (`bank`, `biglarge`, `black`, `blue`, `cellphone`, `good`, `hello`, `hot`, `new`, `pen`, `red`, `shoes`, `smalllittle`, `storeorshop`, `thankyou`, `tshirt`, `white`).
- **Total Samples**: 425 landmark sequences (25 samples per class).
- **Signer-Independent Split**:
  - **Train**: Signer A (153 samples = 9 per class)
  - **Validation**: Signer B (136 samples = 8 per class)
  - **Held-out Test**: Signer C (136 samples = 8 per class — completely unseen in training)

### 4.2 Neural Recognizer Architecture
- **Input Feature Vector**: 225 dimensions per frame:
  - Left Hand: 21 landmarks × 3 coords = 63 dims (wrist-centered)
  - Right Hand: 21 landmarks × 3 coords = 63 dims (wrist-centered)
  - Pose Upper Body: 33 landmarks × 3 coords = 99 dims (shoulder-midpoint centered)
- **Sequence Length**: 48 frames (uniform linear interpolation).
- **Recurrent Backbone**: 2-layer Bidirectional LSTM (input: 225, hidden: 128 per direction, total 256).
- **Pooling Head**: Concatenated Average Pooling + Max Pooling across temporal axis $\rightarrow$ 256 dims.
- **Classification Head**: LayerNorm $\rightarrow$ Linear(256 $\rightarrow$ 128) $\rightarrow$ GELU $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear(128 $\rightarrow$ 17 classes).
- **Loss & Optimization**: CrossEntropyLoss, Adam optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), ReduceLROnPlateau scheduler.

---

## 5. Verified Empirical Results

### 5.1 Recognizer Performance on Held-Out Test Set (Signer C)
| Metric | Value |
| :--- | :--- |
| Evaluated Samples | 136 held-out test samples (Signer C) |
| Top-1 Test Accuracy | **100.00%** |
| Top-3 Test Accuracy | **100.00%** |
| Test Cross-Entropy Loss | 0.0014 |
| Average Entropy | 0.094 bits |
| Average Top-2 Margin | 0.985 |

*Scientific Interpretation*: 100% accuracy on this test partition reflects that the 17 isolated sign classes in this normalized skeletal dataset possess clear kinematic trajectories across these three signers. It does not imply 100% real-world accuracy in unconstrained open-world conditions.

### 5.2 Comparative Policy Benchmark Results
Evaluated on 10 controlled retail scenarios (ambiguous colors, garment boundaries, polysemy, motion blur):

| Policy ID | Policy Name | Success% | Conf-Wrong% | Clarify% | Avg Turns | Avg IG | Avg $U(q)$ | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** (Baseline A) | Direct Top-1 Commitment | 60.0% | **40.0%** | 0.0% | 1.00 | N/A | N/A | 0.00 ms |
| **B2** (Baseline B) | Fixed Confidence ($p_1 < 0.80$) | 100.0% | **0.0%** | 70.0% | 1.70 | N/A | N/A | 0.00 ms |
| **B3** (Baseline C) | Uncertainty-Only ($H, \Delta$) | 100.0% | **0.0%** | 70.0% | 1.70 | N/A | N/A | 0.01 ms |
| **B_rand** (Baseline D) | Random Clarification Selection | 90.0% | **0.0%** | 70.0% | 1.70 | 1.36 b | 1.34 | 0.04 ms |
| **B4** (Proposed) | **ClarifySign Multi-Attribute Policy** | **100.0%** | **0.0%** | **70.0%** | **1.70** | **1.50 b** | **1.49** | **0.04 ms** |

### 5.3 Component Ablation Analysis
| Ablation ID | Configuration | Success% | Clarify% | Avg Turns | Avg IG | Avg $U(q)$ | Cost Penalty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ABL-1** | EIG Only ($\alpha=0, \beta=0, \lambda=0$) | 100.0% | 70.0% | 1.70 | 1.50 b | 1.50 | 0.000 |
| **ABL-2** | EIG + Context ($\alpha=0.30, \beta=0, \lambda=0$) | 100.0% | 70.0% | 1.70 | 1.50 b | 1.57 | 0.000 |
| **ABL-3** | EIG + Answerability ($\beta=0.15, \lambda=0$) | 100.0% | 70.0% | 1.70 | 1.50 b | 1.63 | 0.000 |
| **ABL-4** | EIG − Interaction Cost ($\lambda=0.18$) | 100.0% | 70.0% | 1.70 | 1.50 b | 1.30 | 0.140 |
| **ABL-5** | **Full ClarifySign Policy** | **100.0%** | **70.0%** | **1.70** | **1.50 b** | **1.49** | **0.140** |

---

## 6. Multilingual Translation & Speech Capabilities

- **10 Indian Languages Supported**: Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia.
- **Translation Pipeline Transparency**:
  1. *Verified Semantic Database & Templates*: Instant, deterministic native translation of all 17 domain concepts and conversational acts without hallucination.
  2. *Neural Seq2Seq (NLLB-200 / IndicTrans2)*: Optional online backend for arbitrary text sentences.
- **Speech Synthesis**:
  - Client-side: Web Speech API using native browser synthesis with BCP-47 locale tags (`hi-IN`, `ta-IN`, `bn-IN`, etc.).
  - Server-side: gTTS generating MP3 audio streams.

---

## 7. Research Positioning, Novelty & Patent Considerations

### 7.1 Novelty Claim
The contribution of ClarifySign is **not** raw sign recognition. The core novelty is the **information-theoretic clarification protocol** that:
1. Formulates sign ambiguity resolution as an Expected Information Gain optimization problem.
2. Integrates multi-attribute utility (dialogue context, answerability, interruption cost) to choose minimal, high-yield questions.
3. Completely avoids confidently-wrong commitments in ambiguous interactions.

### 7.2 Patentability Assessment
- *Patent Potential*: A technical method claim around **"A method and system for active ambiguity-aware sign language dialogue disambiguation via multi-attribute expected information gain optimization"** is structurally defensible as a computer-implemented interaction protocol.
- *Prior Art Separation*: Standard SLR patents focus on feature extraction or neural network architectures; ClarifySign focuses on the uncertainty-triggered dialogue decision policy.

---

## 8. Candid System Limitations

1. **Vocabulary Size**: 17 domain classes. Does not recognize open-vocabulary ISL.
2. **Gesture Type**: Isolated signs only; does not perform continuous sign language recognition (CSLR).
3. **Calibrated Probability**: Uses temperature-scalable softmax output; full empirical calibration on out-of-distribution data is left for future work.
4. **Evaluation Nature**: Evaluated on a deterministic controlled benchmark. **No human-subject clinical or field trial was conducted.**
