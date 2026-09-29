# Calibrated Clarification Engine

`CalibratedClarificationEngine` accepts Pydantic `SignToTextPrediction` and `SpeechToTextPrediction` values from `src/clarifysign/clarification/schemas.py`. It uses the provider's calibrated candidate probabilities; it does not generate or treat an LLM confidence score as model confidence.

## Decision policy

Clarification is triggered when one or more of these deterministic conditions holds:

- calibrated top-1 confidence is below the configured threshold;
- Shannon entropy over the supplied top-k candidates exceeds the configured bits threshold;
- signal validity or temporal coverage is too low;
- the top two candidates expose different supplied semantic groups.

Options are constructed directly from the top-k candidate text/probability values. Expected utility is:

```text
expected utility = expected entropy reduction / interaction cost
```

For the initial deterministic response model, a confirmed structured option resolves the candidate entropy, so expected reduction is the entropy of the candidate distribution. Costs are configured per modality. The selected modality is deterministic: poor sign-video quality prefers speech, ambiguous sign with two candidates prefers a yes/no gesture, high-entropy speech prefers sign animation, and other cases use multiple-choice UI or text.

## Persistent confirmation state

`SQLiteDialogueStore` records requests and confirmations at a caller-supplied SQLite path. A valid confirmation produces the only resolved final text after an ambiguity request. Consecutive unanswered/failed requests reach `maximum_failures`, after which the engine returns an explicit safe fallback with no inferred final text. A confirmation resets the consecutive failure count.

## Controlled paraphrasing

`DeterministicTemplateRenderer` is the default. An optional `ControlledParaphraser` may paraphrase only the deterministic prompt and must retain every candidate term; otherwise its output is discarded. It is not used for probability, candidate, modality, or safety decisions.
