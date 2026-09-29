# ClarifySign — final submission package

## What is actually ours
This project is **not an application wrapper around an existing ISL translator**. The recognition component in `core/model.py` is trained from random initialization by `scripts/train.py`. The project-specific contribution is the uncertainty-aware clarification layer in `core/uncertainty.py` + `core/clarification.py` + `core/dialogue.py`.

The external resource is the **INCLUDE dataset**, not an external recognition checkpoint. The official dataset card says its videos are distributed through Zenodo. INCLUDE contains isolated ISL signs; the first run uses the available dataset structure and trains our model locally.

## Exact architecture
Video → MediaPipe landmarks → 225-D sequence → BiLSTM temporal encoder → class posterior → entropy/margin → clarification policy → conversational state → multilingual rendering.

## Research contribution
The claim is not “first ISL translator”. The claim is:

> A clarification-driven communication protocol that uses uncertainty, information gain, contextual relevance and interaction cost to decide when a sign recognizer should ask the user instead of committing to a potentially wrong interpretation.

## Important submission honesty
The ZIP cannot contain the full public video dataset or a fabricated trained checkpoint. Running option 1 downloads the official dataset and trains the model. **Do not invent accuracy numbers**; copy the actual values printed by your run into the report.

## Dataset / limitations
INCLUDE is an isolated-sign dataset. It is not evidence of complete continuous conversation. The official dataset documentation also notes that its videos were collected in Chennai and are not a complete representation of regional ISL variation.

For continuous sentence-level work, cite iSign/ISLTranslate as future/extension work rather than claiming that this package already performs continuous sentence translation.

## Demo sequence for evaluation
1. Start the app.
2. Show a clear sign where top-1 confidence passes the policy threshold.
3. Show an ambiguous case where the system asks for clarification.
4. Select the intended candidate.
5. Change output language among Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Punjabi and Odia.
6. Open research metrics and show entropy, margin and candidate probabilities.
7. Run the policy stress test and preserve the generated CSV/JSON.

## Files
- `RUN_ME.py` — single entry point.
- `RUN_WINDOWS.bat` — Windows launcher.
- `app.py` — Streamlit application.
- `core/model.py` — project recognition model.
- `core/uncertainty.py` — uncertainty metrics.
- `core/clarification.py` — clarification decision engine.
- `core/dialogue.py` — conversational state.
- `core/translation.py` — multilingual translation adapter.
- `scripts/download_include.py` — official dataset downloader.
- `scripts/prepare_dataset.py` — landmark preparation.
- `scripts/train.py` — model training.
- `scripts/policy_stress_test.py` — policy experiment.
- `research_spec.md` — research protocol.
