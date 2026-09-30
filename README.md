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

## Production Full-Stack Application (FastAPI + React TypeScript)

ClarifySign provides an enterprise-ready production deployment consisting of a FastAPI REST backend and a modern React + TypeScript frontend with dark glassmorphism design.

### Architecture Overview

- **Frontend (`frontend/`)**: React 19 + TypeScript + Vite + Zustand.
  - Real-time webcam capture (10 FPS) via `getUserMedia` and Canvas API.
  - Real-time ISL gesture prediction displays with color-coded certainty thresholds.
  - Interactive Shannon entropy and margin uncertainty indicators.
  - Dynamic Clarification Cards driven by Expected Information Gain (EIG).
  - Multilingual translation grid supporting 10 Indian languages.
  - Integrated speech synthesis using Web Speech API and backend gTTS fallback.
- **Backend (`backend/app/`)**: FastAPI async REST service.
  - `POST /api/predict`: Extracts 225-dim skeletal landmarks (hands + pose) and evaluates `ISLBiLSTM`.
  - `POST /api/dialogue/start`: Evaluates ambiguity criteria and initiates an EIG dialogue turn.
  - `POST /api/dialogue/clarify`: Resolves clarification options and commits customer intent.
  - `POST /api/translate`: Provides verified multilingual translations.
  - `POST /api/speak`: Generates speech audio in Indian languages.
  - `GET /health`: Real-time health check and model loading status.

### Running with Docker Compose

Both backend and frontend can be started with a single command:

```bash
docker compose up --build
```

- Frontend is accessible at: `http://localhost:3000`
- Backend REST API and Swagger Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Running Locally for Development

**1. Backend (FastAPI)**
```bash
cd backend
pip install -r requirements.txt
export PYTHONPATH="..:."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Frontend (React + Vite)**
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to interact with the application.

### Running Backend Unit & API Tests

```bash
pytest backend/tests/ -v
```

All 36 unit and integration test cases validate endpoint responses, uncertainty computation, dialogue session lifecycle, and error handling.

