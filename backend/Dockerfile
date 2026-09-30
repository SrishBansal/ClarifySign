FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./backend_requirements.txt
RUN pip install --no-cache-dir -r ./backend_requirements.txt

# Copy backend application, core recognition & translation modules, and models
COPY config.py ./config.py
COPY core/ ./core/
COPY models/ ./models/
COPY backend/ ./backend/

# Set Python path to find root and backend modules
ENV PYTHONPATH="/workspace:/workspace/backend"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"

WORKDIR /workspace/backend

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
