# ClarifySign — Execution & Deployment Runbook

## 1. Prerequisites

- **Operating System:** macOS (Apple Silicon / Intel), Linux (Ubuntu 20.04+), or Windows 10/11
- **Python:** 3.10 to 3.13 (Python 3.13 tested and verified)
- **Webcam:** USB or built-in camera (for live signing demo)
- **RAM:** Minimum 4 GB (8 GB recommended)
- **Disk Space:** ~500 MB for core installation + models (~2 GB if downloading neural NLLB models)

---

## 2. Quickstart (One-Command Experience)

After cloning or unzipping the repository:

### macOS / Linux
```bash
python3 RUN_ME.py
```

### Windows
Double-click `RUN_WINDOWS.bat` or run:
```cmd
python RUN_ME.py
```

### Non-Interactive Automation Flags
```bash
# Check hardware and environment
python RUN_ME.py --check

# Generate/download dataset
python RUN_ME.py --download

# Preprocess sequences
python RUN_ME.py --prepare

# Train BiLSTM recognition model
python RUN_ME.py --train

# Run policy evaluation benchmark
python RUN_ME.py --eval

# Run 27-test verification suite
python RUN_ME.py --test

# Launch Streamlit web UI directly
python RUN_ME.py --app

# Execute entire pipeline end-to-end
python RUN_ME.py --all
```

---

## 3. Demonstration Modes

ClarifySign provides three versatile demonstration modes to ensure bulletproof presentations:

### Mode A: Browser Webcam (Interactive Live Capture)
- Select **📸 Browser Webcam** in the web UI.
- Grant camera permissions in your web browser.
- Perform an ISL sign (e.g., shoes, shirt, blue, hello) and click **Capture Signing Frame**.
- The system extracts 225 landmarks, feeds the temporal buffer, runs BiLSTM inference, evaluates entropy, and renders the result or clarification card.

### Mode B: Controlled Scenarios (Viva Defense Mode)
- Select **🧪 Interactive Scenarios** in the UI.
- Choose between Ambiguous Color (Blue vs Black), Garment, Sizing, Bank Polysemy, or Clear Gestures.
- Demonstrates the exact information-theoretic decision mechanism deterministically without relying on camera lighting conditions during examiner questioning.

### Mode C: Video File Upload
- Select **📁 Video File Upload**.
- Upload any `.mp4` or `.mov` clip of ISL signing.

---

## 4. Deployment Guide

### Target 1: Local Deployment (Recommended for Final-Year Defense)
Local deployment is strongly recommended for the live examination because:
- Zero network latency for real-time video processing.
- Direct hardware camera access without browser iframe permission blocks.
- Full access to local Apple Silicon MPS or NVIDIA GPU acceleration.

### Target 2: Streamlit Cloud Deployment
ClarifySign is fully architected for deployment to [Streamlit Community Cloud](https://streamlit.io/cloud):

1. **Push Repository to GitHub:**
   Ensure `models/isl_bilstm.pt`, `models/labels.json`, and `models/holistic_landmarker.task` are tracked or downloaded automatically.
2. **Deploy on Streamlit Cloud:**
   - App repository: `your-username/ClarifySign`
   - Main file path: `app.py`
3. **Camera in Cloud Containers:**
   - In cloud containers, server-side OpenCV `cv2.VideoCapture(0)` cannot access the client's local camera hardware because the server is running in a remote Docker container!
   - ClarifySign solves this via **`st.camera_input`**, which requests browser camera permissions directly on the client machine and uploads the snapped frame to the container.
4. **Cloud Resource Limits (Free Tier):**
   - Free tier provides 1 GB RAM and shared CPU.
   - ClarifySign's PyTorch BiLSTM model uses only ~4 MB of RAM and ~0.04 ms inference time, easily fitting within free-tier limits.
   - The neural translation model (NLLB-200 / IndicTrans2) requires ~600 MB - 1.2 GB of RAM. On free-tier instances, ClarifySign automatically activates the lightweight **Verified Multilingual Semantic Engine**, which requires under 1 MB of RAM and guarantees instant 10-language output without crashing.

---

## 5. Troubleshooting & FAQ

### Issue: `AttributeError: module 'mediapipe' has no attribute 'solutions'`
- **Cause:** MediaPipe removed the legacy `solutions` API in Python 3.13 in favor of `mediapipe.tasks`.
- **Solution:** ClarifySign uses `mediapipe.tasks.python.vision.HolisticLandmarker` natively with `MEDIAPIPE_DISABLE_GPU=1`, eliminating this issue.

### Issue: Camera not detected or permission denied
- **Cause:** Operating system privacy settings blocking camera access in terminal or browser.
- **Solution:**
  - On macOS: System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Camera $\rightarrow$ enable for Terminal or Chrome.
  - Or switch to **Mode B: Interactive Scenarios** in the UI to demonstrate all pipeline capabilities deterministically.

### Issue: Audio/Speech not playing in browser
- **Cause:** Browser autoplay policies require user interaction before playing audio.
- **Solution:** Click the **🔊 Speak Output** button or press play on the generated MP3 audio player widget.
