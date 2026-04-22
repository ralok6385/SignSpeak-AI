@echo off
title SignSpeak AI — Sign Language Translation Demo
cd /d "%~dp0"
color 0B

echo.
echo  ██████████████████████████████████████████████████████████
echo  █                                                        █
echo  █          S I G N S P E A K   A I                      █
echo  █       Sign Language Translation System                 █
echo  █                                                        █
echo  ██████████████████████████████████████████████████████████
echo.
echo  [1/3] Checking Python dependencies...
pip install flask flask-cors mediapipe opencv-python torch transformers --quiet 2>nul
echo        Done.
echo.
echo  [2/3] Loading AI model (this may take 10-20 seconds)...
echo        Model: T5 Encoder-Decoder  ^|  Trained on How2Sign dataset
echo        Device: CUDA GPU (if available) or CPU
echo.
echo  [3/3] Starting local inference server...
echo.
echo  ████████████████████████████████████████████████████████
echo  █  Open your browser at:  http://localhost:5000        █
echo  █  (Browser will open automatically in 5 seconds)      █
echo  ████████████████████████████████████████████████████████
echo.

:: Open browser after a short delay (runs in parallel)
start "" cmd /c "timeout /t 8 /nobreak >nul & start \"\" http://localhost:5000"

echo  Server is running. Press Ctrl+C to stop.
echo.

:: Run server in foreground (keeps window open, shows logs)
python server.py
pause
