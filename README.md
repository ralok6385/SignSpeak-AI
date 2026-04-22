<p align="center">
  <img src="preview.png" alt="SignSpeak AI Banner" width="600" />
</p>

# SignSpeak AI 

**Real-time Sign Language → English Translation** powered by T5 Transformer + MediaPipe keypoint detection.

> Upload a sign language video or use your live camera — the AI extracts skeleton keypoints, normalizes them, and translates the sequence to English text in real-time.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00A67E.svg)](https://developers.google.com/mediapipe)

---

## 🖥️ Demo

> **No server required!** Just open `frontend/index.html` in your browser — Demo Mode activates automatically with simulated translations so you can explore the full UI instantly.

For full AI-powered translation, start the server and open http://localhost:5000.

## ✨ Features

- 🎥 **Video Upload Mode** — drag & drop a sign language video for translation
- 📷 **Live Camera Mode** — real-time webcam translation with LIVE indicator
- 📊 **Confidence Metrics** — real-time model confidence display for both video and live modes
- 🔊 **Text-to-Speech** — hear translations spoken aloud (Web Speech API)
- 📋 **Copy to Clipboard** — one-click copy of translated text
- 🎯 **Demo Mode** — works instantly without a server (browser-only demo fallback)
- 🌓 **Theme Support** — toggle between light and dark modes
- 🖥️ **Premium UI** — glassmorphic interface with micro-animations
-  **Keyboard Shortcuts** — Space to start/stop live translation, Enter to translate video
-  **Responsive** — works on desktop and mobile

##  Architecture

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

##  Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ralok6385/SignSpeak-AI.git
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
│   ├── index.html          # Main UI (glassmorphic dark/light theme)
│   ├── app.js              # Client-side logic (tabs, camera, API calls)
│   └── style.css           # Full theme system with light/dark modes
├── server.py               # Flask API server (inference + MediaPipe)
├── train_how2sign_t5.py    # T5 training pipeline (SignCL, CTC, temporal conv)
├── config.py               # Dataset path configuration
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
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

- [REST API Reference](API.md)
- [Training Runbook V2](HOW2SIGN_TRAINING_RUNBOOK_V2.md)
- [WSL CUDA Setup](WSL_CUDA_TRAINING.md)
- [Work Log](WORKLOG.md)

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for accessibility**
