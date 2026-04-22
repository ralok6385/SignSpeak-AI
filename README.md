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

## 🖥️ Live Demo

> **Project is Live!** Experience the real AI translation without any local setup:
> - **Frontend (Vercel):** [sign-speak-ai-seven.vercel.app](https://sign-speak-ai-seven.vercel.app)
> - **Backend (Hugging Face):** Hosted on HF Spaces for scalable AI inference.

*Note: The frontend automatically connects to the remote AI backend. If the backend is loading, the UI will gracefully switch to a high-fidelity Demo Mode.*

## ✨ Features

- 🎥 **Video Upload Mode** — drag & drop sign language videos for instant processing
- 📷 **Real-time Live Camera** — browser-based webcam translation with sub-second latency
- 🧠 **T5 Transformer Backend** — heavy AI lifting handled by a Python-based T5 model
- 📊 **Confidence Metrics** — visual confidence scores for every prediction
- 🎨 **Premium Glassmorphic UI** — stunning modern design with light/dark mode support
- ❤️ **Interactive Reactions** — WhatsApp-style floating emojis triggered by detected signs
- 🔊 **Text-to-Speech** — integrated voice feedback for accessibility
- 📋 **Production Architecture** — hybrid deployment (Vercel + Hugging Face Spaces)

## 📄 Research & Methodology

This project is backed by formal research. The methodology, architectural decisions (like using T5 for sequential coordinate translation), and training metrics are detailed in our accompanying research paper.

> **[Read the Full Research Paper (PDF) ↗](./SignSpeak_AI_Research_Paper.pdf)** *(Make sure to upload your PDF to the repo and name it this, or update the link!)*

## 🏗 Architecture

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
