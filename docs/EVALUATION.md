# ClarifySign — Policy Benchmark & Recognizer Evaluation Report

This document records the empirical evaluation of the ClarifySign Indian Sign Language (ISL) recognition engine and the information-theoretic clarification policy.

---

## 1. Important Scope & Evaluation Methodology Notice

> **IMPORTANT SCIENTIFIC DISCLAIMER**:
> All policy benchmark results presented here were obtained from a **deterministic controlled benchmark** across 10 hand-crafted shopkeeper scenarios with synthetic candidate distributions. 
> **No human-subject evaluation was conducted.** 
> These experiments serve as an engineered proof-of-concept demonstrating that ambiguity-aware clarification mathematically prevents confidently-wrong recognitions and that Expected Information Gain (EIG) optimizes question selection.

---

## 2. Recognizer Quantitative Metrics

### 2.1 Dataset & Partitioning
- **Vocabulary**: 17 shopkeeper domain classes (`bank`, `biglarge`, `black`, `blue`, `cellphone`, `good`, `hello`, `hot`, `new`, `pen`, `red`, `shoes`, `smalllittle`, `storeorshop`, `thankyou`, `tshirt`, `white`).
- **Total Samples**: 425 skeletal landmark sequences (48 frames × 225 features).
- **Signer Split**:
  - **Train**: Signer A (153 samples — 9 per class)
  - **Validation**: Signer B (136 samples — 8 per class)
  - **Test (Held-out)**: Signer C (136 samples — 8 per class)
- **Signer Leakage Safeguard**: Active and verified. Signer C was completely unseen during model training.

### 2.2 Empirical Test Set Performance
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Test Top-1 Accuracy** | **100.00%** | All 136 test samples correctly classified at rank 1 |
| **Test Top-3 Accuracy** | **100.00%** | Target class present in top 3 predictions |
| **Average Test Loss** | **0.0014** | Cross-entropy loss on held-out Signer C |
| **Average Sample Entropy** | **0.094 bits** | Low entropy on clean test samples |
| **Average Top-2 Margin** | **0.985** | High separation between rank-1 and rank-2 |

### 2.3 Critical Analysis of 100% Accuracy
While 100.00% accuracy was achieved on the held-out test split, this result must be interpreted in proper context:
1. **Closed Vocabulary**: The model evaluates only 17 domain-specific signs, which possess distinct hand movement patterns.
2. **Normalized Skeletal Features**: Pre-extracted 225-dimensional MediaPipe landmark coordinates isolate hand kinematics and eliminate background/lighting noise.
3. **Open-World Reality**: In real-world deployment (casual signing, camera motion blur, extreme lighting, unmodeled gestures), recognition accuracy is expected to degrade. This is the exact motivation for ClarifySign's uncertainty and clarification mechanism.

---

## 3. Communication Policy Benchmark Evaluation

Evaluated across 10 controlled shopkeeper scenarios representing subtle color confusions, boundary garment signs, polysemous words, and high motion blur.

### 3.1 Comparative Policy Results

| Policy | Policy Name & Description | Success Rate (%) | Confidently-Wrong Rate (%) | Clarification Rate (%) | Avg Turns | Avg Information Gain | Avg Question Utility $U(q)$ | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** (Baseline A) | Direct Top-1 Commitment (Greedy) | **60.0%** | **40.0%** | 0.0% | 1.00 | N/A | N/A | 0.00 ms |
| **B2** (Baseline B) | Fixed Confidence Threshold ($p_1 < 0.80$) | **100.0%** | **0.0%** | 70.0% | 1.70 | N/A | N/A | 0.00 ms |
| **B3** (Baseline C) | Uncertainty-Only ($H > 1.25$ or $\Delta < 0.20$) | **100.0%** | **0.0%** | 70.0% | 1.70 | N/A | N/A | 0.01 ms |
| **B_rand** (Baseline D) | Random Clarification Selection | **90.0%** | **0.0%** | 70.0% | 1.70 | 1.36 bits | 1.34 | 0.04 ms |
| **B4** (Proposed) | **ClarifySign Multi-Attribute Policy** | **100.0%** | **0.0%** | 70.0% | 1.70 | **1.50 bits** | **1.49** | **0.04 ms** |

---

## 4. Component Ablation Analysis

To investigate the individual mathematical contributions of each term in the multi-attribute utility function:
$$U(q) = IG(q) + \alpha \cdot \text{Context}(q) + \beta \cdot \text{Answerability}(q) - \lambda \cdot \text{Cost}(q)$$

| Ablation ID | Configuration | Success Rate (%) | Clarification Rate (%) | Avg Turns | Avg Information Gain | Avg Question Utility $U(q)$ | Interaction Cost Penalty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ABL-1** | EIG Only ($\alpha=0, \beta=0, \lambda=0$) | 100.0% | 70.0% | 1.70 | 1.50 bits | 1.50 | 0.000 |
| **ABL-2** | EIG + Context ($\alpha=0.30, \beta=0, \lambda=0$) | 100.0% | 70.0% | 1.70 | 1.50 bits | 1.57 | 0.000 |
| **ABL-3** | EIG + Answerability ($\alpha=0, \beta=0.15, \lambda=0$) | 100.0% | 70.0% | 1.70 | 1.50 bits | 1.63 | 0.000 |
| **ABL-4** | EIG − Cost ($\alpha=0, \beta=0, \lambda=0.18$) | 100.0% | 70.0% | 1.70 | 1.50 bits | 1.30 | 0.140 |
| **ABL-5** | **Full ClarifySign Policy** | **100.0%** | **70.0%** | **1.70** | **1.50 bits** | **1.49** | **0.140** |

---

## 5. Latency Breakdown

To prevent misrepresenting sub-millisecond policy optimization as end-to-end system latency, the execution latency is explicitly partitioned:

| Pipeline Stage | Typical Latency | Description |
| :--- | :--- | :--- |
| **MediaPipe Holistic Extraction** | ~15–30 ms / frame | CPU/GPU landmark detection per video frame |
| **Temporal Resampling & Buffer** | < 1 ms | Resampling buffered frames to 48 timesteps |
| **ISLBiLSTM Neural Inference** | ~2–5 ms | PyTorch forward pass on Apple Silicon MPS / GPU |
| **Uncertainty & Ambiguity Gate** | ~0.01 ms | Vectorized entropy, margin, and trigger checks |
| **Clarification Utility Optimization** | ~0.04 ms | EIG, context, and multi-attribute utility calculation |
| **Semantic Translation Lookup** | < 0.1 ms | In-memory verified dictionary mapping |
| **Neural Translation (NLLB-200)** | ~250–600 ms | Seq2seq transformer inference (when enabled) |
| **Browser Web Speech (Client-side)** | 0 ms server latency | Native browser `window.speechSynthesis` |
| **gTTS Server-Side Audio Synthesis** | ~300–800 ms | Network round-trip audio generation |

---

## 6. Key Research Takeaways

1. **Safety Over Speed**: Baseline A (Direct Top-1) commits immediately in 1 turn, but fails 40% of ambiguous scenarios with confident errors. ClarifySign eliminates 100% of these errors by spending an average of only 0.70 additional turns.
2. **Information Optimization**: While naive baseline B2 and B3 trigger clarification, ClarifySign's EIG formulation selects questions that maximize uncertainty reduction (1.50 bits/question vs 1.36 bits in random question selection).
3. **Sub-millisecond Policy Overhead**: The complete utility ranking and question selection executes in 0.04 ms, making it computationally invisible in interactive dialogue.
