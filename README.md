# SignSpeak AI 🤟

**Real-time Sign Language → English Translation** powered by T5 Transformer + MediaPipe keypoint detection.

> Upload a sign language video or use your live camera — the AI extracts skeleton keypoints, normalizes them, and translates the sequence to English text in real-time.

---

## ✨ Features

- 🎥 **Video Upload Mode** — drag & drop a sign language video for translation
- 📷 **Live Camera Mode** — real-time webcam translation with LIVE indicator
- 📊 **Confidence Metrics** — real-time model confidence display for both video and live modes
- 🔊 **Text-to-Speech** — hear translations spoken aloud (Web Speech API)
- 📋 **Copy to Clipboard** — one-click copy of translated text
- 🎯 **Demo Mode** — works instantly without a server (browser-only demo fallback)
- 🌓 **Theme Support** — toggle between light and dark modes
- 🖥️ **Premium UI** — glassmorphic interface with micro-animations
- 📱 **Responsive** — works on desktop and mobile

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
│   Frontend   │────▶│  Flask Server (API)  │────▶│   T5 Model   │
│  HTML/JS/CSS │     │  + MediaPipe (GPU)   │     │  Inference   │
└──────────────┘     └─────────────────────┘     └──────────────┘
     │                        │
     │ video/frame            │ keypoints [T, 67, 4]
     │ (base64)               │ ↓ normalize + interpolate
     └────────────────────────┘ ↓ T5 encoder-decoder → text
```

**Pipeline:** Video Frames → MediaPipe Pose+Hand → 67-joint keypoints (x, y, conf, missing) → Shoulder-anchored normalization → T5-small encoder → English text

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/SignSpeak-AI.git
cd SignSpeak-AI
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python server.py
```

Then open **http://localhost:5000** in your browser.

> **No server?** Just open `frontend/index.html` directly — Demo Mode activates automatically with simulated translations.

### 3. (Optional) Train the Model

See [HOW2SIGN_TRAINING_RUNBOOK_V2.md](HOW2SIGN_TRAINING_RUNBOOK_V2.md) for full training instructions.

```bash
bash launch_local.sh
```

## 📁 Project Structure

```
SignSpeak-AI/
├── frontend/
│   ├── index.html          # Main UI
│   ├── app.js              # Client-side logic (tabs, camera, API calls)
│   └── style.css           # Dark futuristic theme
├── server.py               # Flask API server (inference + MediaPipe)
├── train_how2sign_t5.py    # T5 training pipeline (SignCL, CTC, temporal conv)
├── config.py               # Dataset path configuration
├── requirements.txt        # Python dependencies
├── models/                 # MediaPipe .task files (auto-downloaded)
├── runs/                   # Training checkpoints & logs
└── DATASET/                # How2Sign dataset (not included)
```

## 🧠 Model Details

| Component | Detail |
|-----------|--------|
| **Architecture** | T5-small encoder-decoder with temporal Conv1D compression |
| **Input** | 67 joints × 4 channels (x, y, confidence, missing flag) |
| **Normalization** | Shoulder-anchored signing-space normalization |
| **Auxiliary Losses** | SignCL contrastive loss + CTC loss |
| **Keypoints** | 25 body (OpenPose format) + 21 left hand + 21 right hand |
| **Dataset** | How2Sign (English ASL translations) |

## 📊 Training Metrics

Tracked per epoch: BLEU, BLEU-1/2/3/4, METEOR, ROUGE-L

## 🔧 Configuration

All dataset paths are in [`config.py`](config.py). Run `python config.py` to verify paths.

## 📝 Documentation

- [Training Runbook V2](HOW2SIGN_TRAINING_RUNBOOK_V2.md)
- [WSL CUDA Setup](WSL_CUDA_TRAINING.md)
- [Work Log](WORKLOG.md)

## 📄 License

MIT

---

**Built with ❤️ for accessibility**
