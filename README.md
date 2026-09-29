# ClarifySign

ClarifySign is being refactored into a provider-based Indian Sign Language communication system. The modular package deliberately ships with mock providers only: it does not download data, model weights, or call external APIs during local development or tests.

## Architecture

```text
                              production direction
ISL video / pose  ->  preprocessing  ->  sign-to-text provider
                                             |
                                      confidence calibration
                                             |
                                      clarification engine
                                             |
                                      dialogue state
                                             |
                                 semantic intent / translation / TTS

                              future reverse direction
speech / text  ->  ASR provider  ->  semantic representation
                                             |
                                  sign-generation provider
                                             |
                                      avatar provider
```

Providers are interfaces, not fixed vendors. The application composition root selects implementations; the default local configuration uses deterministic mocks. A future sign-generation or avatar provider must be licensed and evaluated before it is exposed as a product capability. This repository makes no novelty claim about avatars or bidirectional translation.

## Repository layout

```text
src/clarifysign/
  config/             typed settings
  data/               dataclass request/response schemas
  preprocessing/      pose preprocessing interface
  sign_to_text/       recognition provider interface and mock
  speech/              ASR/TTS interfaces and mocks
  translation/         semantic translation interface and mock
  sign_generation/     motion-plan interface and mock
  avatar/              renderer interface and mock
  clarification/       confidence/margin decision service
  dialogue/            dialogue state
  evaluation/          typed evaluation case schema
  api/                 application service orchestration
app/
  backend/             provider composition root
  frontend/            dependency-free local wiring demo
baselines/             isolated comparison/fallback artifacts only
config/                configuration boundary documentation
configs/               non-secret defaults
data/, docs/, tests/   data, records, and test suite
```

The former `core/`, `training/`, root `app.py`, and `evaluation/` files remain legacy compatibility material while their approved migrations are performed in small pull requests. Training is intentionally not implemented or extended in the new architecture.

## Local development

Requires Python 3.10+; no GPU, dataset download, model download, or API key is required for the modular test path.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
python3 -m pytest -q
python3 -m app.frontend.local_demo
```

`configs/default.toml` selects only mock providers. Keep secrets in a local `.env` file; `.env` must never be committed. Production adapters belong behind the interfaces in `src/clarifysign/` and must declare their data, model, license, calibration, and network requirements.

## Baselines and safeguards

There are no word-to-video or phrase-to-video maps in the new production package. If static dictionary-video playback is later retained, it belongs in `baselines/` as an explicitly labelled fallback provider, never as the default sign-generation path.

Migration evidence and scope controls are recorded in:

- [Current repository audit](docs/CURRENT_REPO_AUDIT.md)
- [Removal and migration plan](docs/REMOVAL_AND_MIGRATION_PLAN.md)
- [Removal report](docs/REMOVAL_REPORT.md)
