# ClarifySign — Comprehensive Viva Defense & Examination Guide

---

### 1. One-Line Project Definition
**ClarifySign is an ambiguity-aware Indian Sign Language communication protocol that estimates recognition uncertainty and initiates information-gain-optimized clarifications before translating retail interactions into 10 Indian languages.**

---

### 2. The Practical Problem
Deaf and hard-of-hearing customers face significant communication barriers in retail and shopkeeper environments (grocery stores, garment shops, counters). 
Signs often share visually overlapping hand configurations or motion paths (e.g. subtle rotational differences between colors like Blue and Black, or similar chest-level sweeps for T-shirt vs Shirt). Environmental noise, rapid signing, and lighting variations further distort recognition.

---

### 3. Existing Limitation of Traditional SLR Systems
Traditional Sign Language Recognition (SLR) systems operate in a **greedy, single-shot mode**:
- They classify an input video and directly output the **Top-1** class label.
- When an input is ambiguous, greedy commitment results in **confidently-wrong communication** (e.g. the system confidently tells the shopkeeper "Customer wants black shoes" when the customer actually signed "blue shirt").
- In high-stakes transactions, a confidently-wrong prediction is far more damaging than refusing to guess and asking a 2-second clarifying confirmation.

---

### 4. The Proposed System
ClarifySign introduces an **interactive, clarification-driven communication protocol**:
1. It computes a calibrated posterior probability distribution over candidate signs.
2. It estimates information-theoretic uncertainty using Shannon Entropy, Top-2 Margin, and Top-1 Confidence.
3. If uncertainty exceeds safety thresholds, the system **refuses to guess blindly**.
4. It evaluates candidate clarification questions and selects the question that maximizes **Expected Information Gain (EIG)** while minimizing customer cognitive burden and interaction cost.
5. Once the customer taps or gestures their confirmation, the dialogue state updates, and the verified semantic meaning is translated into the shopkeeper's preferred Indian language.

---

### 5. Architecture Summary
- **Input:** Video stream captured via local camera or browser webcam.
- **Landmark Extractor:** MediaPipe Holistic extracting 225 zero-centered hand and pose spatial coordinates per frame.
- **Temporal Normalizer:** Uniform resampling to exactly 48 temporal frames.
- **Neural Model:** Deep Bidirectional LSTM (`ISLBiLSTM`) trained on our curated AI4Bharat INCLUDE dataset.
- **Uncertainty Engine:** Measures Shannon Entropy $H(Y|X)$, Normalized Entropy, and Competitive Margin $\Delta = p_{(1)} - p_{(2)}$.
- **Clarification Policy:** Computes Question Utility $U(q) = IG(q) + \alpha \cdot \text{Context}(q) + \beta \cdot \text{Ans}(q) - \lambda \cdot \text{Cost}(q)$ and chooses $q^* = \arg\max U(q)$.
- **Dialogue Manager:** Multi-turn dialogue state with exponential context decay.
- **Multilingual Module:** Native translation into 10 official Indian languages with Web Speech API and gTTS audio synthesis.

---

### 6. Why ISL Recognition Alone is Insufficient
Even a model with 95% benchmark accuracy will make mistakes on 1 out of 20 customer gestures. In conversational settings, errors compound rapidly. Isolated recognition models have no awareness of dialogue history, retail context, or whether the user is choosing a color, asking about sizes, or making a payment. Active communication requires dialogic collaboration.

---

### 7. Why Uncertainty Matters
Confidence scores from raw neural networks are frequently miscalibrated. By evaluating the **gap between top candidates (Margin)** and the **dispersion of probability mass across all classes (Entropy)**, the system can distinguish between:
- *Confident & Clear:* $p_1 = 0.94, \Delta = 0.90, H = 0.25$ bits $\rightarrow$ Immediate commitment.
- *Ambiguous Conflict:* $p_1 = 0.49, p_2 = 0.42, \Delta = 0.07, H = 1.35$ bits $\rightarrow$ Dangerous to guess; must clarify.
- *Total Noise/Blur:* Flat distribution across 10 classes $\rightarrow$ Refusal to guess; prompt to re-sign.

---

### 8. How Entropy Works
Shannon Entropy measures the expected surprise or uncertainty in a probability distribution:
$$H(Y|X) = -\sum_{i=1}^K p_i \log_2(p_i) \quad \text{(measured in bits)}$$
- If the model is 100% certain of one sign ($p_1 = 1.0$), $H(Y|X) = 0$ bits.
- If the model is split equally between two signs ($p_1 = 0.5, p_2 = 0.5$), $H(Y|X) = 1.0$ bit.
- If the model is completely confused across 4 signs equally ($p = 0.25$), $H(Y|X) = \log_2(4) = 2.0$ bits.
Higher entropy signifies high model confusion across multiple competing interpretations.

---

### 9. How Information Gain Works
Expected Information Gain (EIG) measures the expected reduction in entropy achieved by asking a clarification question $q$:
$$IG(q) = H(Y|X) - \sum_{a \in A} P(a|q, X) \cdot H(Y|X, q, a)$$
- $H(Y|X)$ is the current uncertainty.
- $\sum_a P(a|q, X) \cdot H(Y|X, q, a)$ is the expected remaining uncertainty after the customer answers $q$.
- For example, if a binary question ("Did you mean Blue or Black?") definitively resolves the two dominant hypotheses, the expected residual entropy drops to near 0, giving an Information Gain of over 1.4 bits!

---

### 10. How Clarification Selection Works
The planner evaluates multiple question formats (binary disambiguation, top-1 confirmation, multi-choice) using a multi-attribute utility function:
$$U(q) = IG(q) + \alpha \cdot \text{Context}(q) + \beta \cdot \text{Answerability}(q) - \lambda \cdot \text{Cost}(q)$$
1. **$IG(q)$:** How many bits of confusion does asking this question eliminate?
2. **$\text{Context}(q)$:** Are the options in this question relevant to what we were just talking about (e.g. clothing colors)?
3. **$\text{Answerability}(q)$:** How easy is it for the customer to answer? Binary (2-choice) questions have higher answerability than 5-choice questions.
4. **$\text{Cost}(q)$:** Interaction penalty for interrupting the customer.
The system selects $q^* = \arg\max U(q)$, ensuring we ask the single most informative, least disruptive question.

---

### 11. How Conversational State & Context Work
`DialogueState` tracks turn history and maintains a **context activation memory** with an exponential decay factor ($\gamma = 0.85$ per turn):
- When the customer commits to "shirt", `context['shirt'] = 0.85`.
- In the next turn, if the gesture is ambiguous between "blue" (color) and "cellphone" (electronics), the clothing context elevates the utility of clarifying "blue", correctly guiding the dialogue toward color selection for the shirt.
- As turns progress without mentioning "shirt", its context weight decays naturally towards zero.

---

### 12. How 10-Language Multilingual Output Works
ClarifySign translates the final resolved concept into 10 official Indian languages:
Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Punjabi, and Odia.
- It uses Flores-200 / IndicTrans2 standard language codes (`hin_Deva`, `tam_Taml`, etc.).
- It features a dual-engine architecture:
  1. Neural Seq2Seq model (HuggingFace NLLB-200 / IndicTrans2).
  2. Verified Semantic Multilingual Engine: Genuine native translations for all shopkeeper dialogue acts and retail concepts, providing zero-latency CPU execution without requiring gigabytes of model downloads.
- Outputs speech in the target Indian language using both the browser Web Speech API and gTTS audio playback.

---

### 13. Why This is Different From a Normal Translator
A standard translator is a **feedforward pipe**: input video $\rightarrow$ output text.
If the input is distorted, the translator outputs an incorrect translation with no recourse.
ClarifySign is an **interactive communicative agent**:
- It models its own uncertainty.
- It can refuse to guess.
- It generates targeted clarification queries.
- It resolves customer feedback into semantic state before speaking to the shopkeeper.

---

### 14. What is Actually Novel
1. **Clarification Decision Policy:** Formulating ambiguity-driven clarification as an Expected Information Gain vs Interaction Cost optimization problem in ISL communication.
2. **Refusal-to-Guess Protocol:** Shifting from raw Top-1 accuracy to minimizing **confidently-wrong communication events** in retail settings.
3. **Context-Aware Utility Optimization:** Demonstrating that conversational context memory dynamically improves question selection efficiency.

---

### 15. Known Limitations (Technical Honesty)
1. **Isolated vs Continuous Signing:** The core recognition model operates on isolated word signs rather than continuous sentence-level sign language with non-manual grammar (facial expressions, mouthings).
2. **Curated Vocabulary:** The model is trained on a 17-class retail subset of AI4Bharat INCLUDE. Expanding to thousands of signs requires larger compute clusters.
3. **Webcam Lighting & Occlusion:** Extreme low light or hand occlusions can degrade MediaPipe landmark tracking.

---

### 16. Likely Professor Questions & Technically Honest Answers

#### Q1: "Did you build your own model or just call an existing API like Sanket or Google Translate?"
> **Answer:** "We built and trained our own neural recognition model from scratch using PyTorch. The architecture is a Deep Bidirectional LSTM with Layer Normalization and dual temporal pooling, trained on isolated gesture landmark sequences from the AI4Bharat INCLUDE corpus. The saved model weights (`models/isl_bilstm.pt`) and training metadata are included in the repository. We do not use Sanket or any external ISL recognition API."

#### Q2: "Why did you use Information Gain instead of just a confidence threshold?"
> **Answer:** "A fixed confidence threshold only tells you *whether* you are uncertain, but not *how to resolve it*. If a gesture has probabilities $[0.48, 0.42, 0.10]$, a confidence threshold can trigger a fallback, but it cannot differentiate between asking a binary question about the top two vs a confirmation question vs a 4-way question. Expected Information Gain mathematically quantifies the entropy reduction of each specific question format, allowing us to select the question that eliminates the most uncertainty with the least customer effort."

#### Q3: "What is your split protocol? Is there signer leakage?"
> **Answer:** "Our dataset pipeline inspects signer metadata. When signer IDs are available, we enforce a strict signer-independent split where Signer A and B are used for training and Signer C is held out for testing. If signer metadata is absent in raw data, the training script explicitly issues a warning that video-level stratified splitting is being used and that cross-signer generalization may be lower. We never fabricate signer independence."

#### Q4: "How does this compare to just asking the user to sign again?"
> **Answer:** "Asking the user to re-sign has a high interaction cost and often produces the exact same ambiguous gesture if lighting or hand orientation was the issue. Targeted clarification (e.g. asking 'Did you mean Blue or Black?') requires only a single tap or quick nod, resolving the ambiguity in under 2 seconds without user frustration."

#### Q5: "How are the 10 languages supported without a massive model?"
> **Answer:** "We support 10 languages through a transparent dual-engine architecture. For high-end systems, we support neural seq2seq models like NLLB-200. For lightweight local execution, we have a verified semantic multilingual engine that provides authentic native translations across all shopkeeper concepts in Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Punjabi, and Odia. The UI transparently labels which engine is active."
