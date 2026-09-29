# ClarifySign — Removal Report

**Scope:** apply only `REMOVE` items approved in [REMOVAL_AND_MIGRATION_PLAN.md](REMOVAL_AND_MIGRATION_PLAN.md). No runtime behavior, dataset, attribution, baseline fixture, or Git history was changed.

## Completed removal

| Path | Action | Reason | Pre-removal verification |
|---|---|---|---|
| `models/hand_landmarker.task` | Deleted | Approved `REMOVE`: the ~7.5 MB MediaPipe Hand bundle had no Python/runtime import, test reference, deployment reference, or functional documentation dependency. Current extraction uses only `models/holistic_landmarker.task` through `config.py:27` and `core/landmarks.py:26–70`. | Repository reference search found only a stale inventory mention in `AUDIT.md:97` and audit/plan references. `app.py`, `config.py`, `core/`, `training/`, `scripts/`, `evaluation/`, `tests/`, launchers, and `requirements.txt` contained no hand-landmarker reference. |

## Deferred approved removal

| Path | Status | Reason for deferral |
|---|---|---|
| `docs/PROJECT_STATUS.md` | Not deleted | The plan's approved row requires deletion **after** a replacement status summary exists in README or architecture. Neither currently contains that replacement. Adding one is a `REFACTOR` action, outside this `REMOVE`-only request. The document remains to avoid deleting its status information prematurely. |

## Moved files

None. No hard-coded translation, dictionary, scenario, dataset, or static mapping was moved because all such items are marked `MOVE_TO_BASELINE` or `REFACTOR`, not `REMOVE`.

## Modified files

| Path | Change | Reason |
|---|---|---|
| `docs/REMOVAL_REPORT.md` | Added | Required audit trail for this removal operation. |

## Imports and dependencies

No imports or dependencies required an update: `models/hand_landmarker.task` was not referenced by Python and is not represented by a separate requirement. `requirements.txt` is unchanged. The potentially removable `pandas`, `Pillow`, `matplotlib`, `gTTS`, `transformers`, and `sentencepiece` entries are `REFACTOR`-stage work and were deliberately left untouched.

## Validation performed

- Searched all runtime modules, tests, launchers, deployment/configuration files, and documentation for `hand_landmarker` and `PROJECT_STATUS` references before removal.
- Confirmed the only active MediaPipe model configuration is `HOLISTIC_TASK_PATH` in `config.py:27`.
- Collected the test suite successfully with `python3 -m pytest --collect-only -q` (27 tests) and executed it with `python3 -m pytest -q` (27 passed).

## Follow-up required

When a separate approved documentation-refactor PR adds the replacement status summary, remove `docs/PROJECT_STATUS.md` and update the stale file inventories in `AUDIT.md` and the historic audit documents as appropriate. Do not treat the missing hand bundle as a reason to remove or alter `models/holistic_landmarker.task`.
