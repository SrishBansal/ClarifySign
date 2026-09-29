# Research Specification

## RQ
Can information-gain-driven active clarification reduce confidently-wrong communication events in ambiguous ISL interactions compared with direct top-1 translation, fixed-confidence fallback, and uncertainty-only clarification?

## Core equation
q* = argmax [ IG(q) + alpha*Context(q) + beta*Answerability(q) - lambda*InteractionCost(q) ]

## Baselines
B1 top-1; B2 fixed confidence; B3 entropy/margin threshold; B4 proposed policy.

## Primary metrics
Communication success, confidently-wrong rate, clarification rate, average turns, information gain per question, resolution time.

## Ablations
Remove context; remove information gain; remove interaction cost; vary lambda.

## Important limitation
The included code is the complete pipeline, but a trained ISL checkpoint cannot honestly be bundled without training on a licensed dataset. The repository therefore never fabricates recognition accuracy.
