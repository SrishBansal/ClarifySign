# Sign-to-Text API

The typed contracts live in `src/clarifysign/sign_to_text/`. They are model-provider interfaces, not an LLM interface and not a claim that any pretrained model is available.

## Pipeline contracts

```text
capture frames
  -> PoseExtractor.extract() -> PoseExtraction + InputQualityMetrics
  -> TemporalSignEncoder.encode() -> EncodedSignSequence
  -> SignToTextModel.predict() -> Prediction
```

| Interface | Method | Responsibility |
|---|---|---|
| `PoseExtractor` | `extract(frames)` | Converts capture-adapter frames into normalized numeric pose frames and quality metrics. |
| `TemporalSignEncoder` | `encode(extraction)` | Converts a variable pose sequence into model input. |
| `SignToTextModel` | `predict(encoded, input_quality)` | Returns a typed sign-to-text `Prediction`. |
| `SignToTextProvider` | `predict(SignToTextRequest)` | Existing application-level candidate-distribution contract; retained for current orchestration compatibility. |

## Prediction fields

`Prediction` always contains:

- `text`: top decoded text label/string;
- `top_k_candidates`: ranked `PredictionCandidate(text, probability)` values;
- `token_probabilities`: model-provided token/label probability mapping;
- `confidence`: confidence for the selected text;
- `latency_ms`: measured adapter inference latency;
- `model_version`: immutable adapter/model identifier;
- `input_quality`: frame count, valid-pose count/ratio, and temporal coverage.

Confidence is an interface value, not evidence of calibration. A fitted calibration artifact must be registered separately before a real provider makes calibrated-confidence claims.

## Included implementations

| Implementation | Use | Network/model behavior |
|---|---|---|
| `MockPoseExtractor`, `IdentityTemporalSignEncoder`, `MockSignToTextModel` | Unit tests and local development | Deterministic, CPU-only; no files or network. |
| `LegacyIsolatedSignClassifierAdapter` | Explicit local compatibility adapter for the prior isolated-sign classifier | Requires supplied local model and labels; never downloads weights. It can also accept an injected fake classifier in tests. |
| `TransformerCheckpointAdapter` | Future extension point | Deliberately raises until a reviewed fine-tuned local-checkpoint implementation is supplied. Automatic pretrained-model downloads are disabled. |

## Example

```python
from clarifysign.sign_to_text.providers import (
    IdentityTemporalSignEncoder,
    MockPoseExtractor,
    MockSignToTextModel,
)

extraction = MockPoseExtractor().extract(((0.0, 1.0),))
encoded = IdentityTemporalSignEncoder().encode(extraction)
prediction = MockSignToTextModel().predict(encoded, extraction)
assert prediction.text == "mock_sign"
```

Static dictionary/video playback is not part of this module. If retained at all, it must be implemented separately as a clearly labelled baseline or fallback provider under `baselines/`.
