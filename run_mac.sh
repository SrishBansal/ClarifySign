#!/usr/bin/env bash
set -e

# ==============================================================================
# ClarifySign - macOS Unified Launcher
# Ambiguity-Aware Indian Sign Language Communication System
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "   ___ _            _  __        ____  _             "
echo "  / __| |__ _ _ _ (_)/ _|_  _  / ___|(_)__ _ _ _    "
echo " | (__| / _` | '_|| |  _| || | \___ \| / _` | ' \   "
echo "  \___|_\__,_|_|  |_|_|  \_, | |____/|_\__, |_||_|  "
echo "                         |__/          |___/        "
echo " Ambiguity-Aware Indian Sign Language Communication System (macOS)"
echo "========================================================================"

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "✓ Python: $PY_VERSION"

# Check Apple Silicon MPS
ACCEL=$(python3 -c '
import torch
if torch.backends.mps.is_available():
    print("Apple Silicon GPU Acceleration (MPS active)")
elif torch.cuda.is_available():
    print(f"CUDA GPU ({torch.cuda.get_device_name(0)})")
else:
    print("Host CPU")
' 2>/dev/null || echo "Standard Python environment")
echo "✓ Hardware: $ACCEL"

MODE="${1:-all}"

case "$MODE" in
    backend)
        echo "🚀 Starting ClarifySign FastAPI backend on http://127.0.0.1:8000 ..."
        cd "$PROJECT_ROOT/backend"
        exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
        ;;
    frontend)
        echo "🚀 Starting ClarifySign Vite frontend on http://127.0.0.1:5173 ..."
        cd "$PROJECT_ROOT/frontend"
        exec npm run dev
        ;;
    test)
        echo "🧪 Running ML & Backend Test Suites..."
        python3 -m pytest tests backend/tests -v
        ;;
    all|*)
        echo "🚀 Starting both Backend and Frontend..."
        (cd "$PROJECT_ROOT/backend" && uvicorn app.main:app --host 127.0.0.1 --port 8000) &
        BACKEND_PID=$!
        
        trap "echo 'Stopping services...'; kill $BACKEND_PID 2>/dev/null" EXIT INT TERM

        if [ -d "$PROJECT_ROOT/frontend" ]; then
            cd "$PROJECT_ROOT/frontend"
            npm run dev
        else
            echo "Frontend directory not found. Running backend only."
            wait $BACKEND_PID
        fi
        ;;
esac
