# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copy the backend application files
COPY server.py .
COPY config.py .
COPY train_how2sign_t5.py .

# Move model weights to the expected directory (SAVE_DIR in config.py)
RUN mkdir -p runs/how2sign_t5_full
COPY best.pt ./runs/how2sign_t5_full/

# Copy MediaPipe model files into a models/ directory
RUN mkdir -p models
COPY hand_landmarker.task ./models/
COPY pose_landmarker_full.task ./models/

# Create frontend directory with a minimal landing page
RUN mkdir -p frontend && echo '<!DOCTYPE html><html><head><title>SignSpeak AI Backend</title><style>body{background:#0a0a1a;color:#e0e0e0;font-family:Inter,sans-serif;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}div{text-align:center;padding:40px;border:1px solid rgba(0,212,255,0.3);border-radius:16px;background:rgba(255,255,255,0.03)}h1{color:#00d4ff}p{color:#888}code{background:rgba(0,212,255,0.1);padding:2px 8px;border-radius:4px;color:#00d4ff}</style></head><body><div><h1>SignSpeak AI Backend</h1><p>API server is running successfully.</p><p>Endpoints: <code>/status</code> <code>/health</code> <code>/translate/video</code> <code>/translate/frame</code></p></div></body></html>' > frontend/index.html

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Run server.py on host 0.0.0.0 and port 7860
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]
