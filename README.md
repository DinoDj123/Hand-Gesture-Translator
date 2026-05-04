
# Hand Gesture Translator

A simple hand gesture recognition project that uses **MediaPipe**, **OpenCV**, and a **KNN classifier** to predict hand gestures from video input.

## Features

- Live gesture prediction using a webcam
- Backend API for video-based predictions
- Frontend page for recording and sending gesture videos
- Dataset processing for training samples

## Usage

### 1. Live Prediction

Run:
```python -m backend.live_predict``` 

This opens your webcam, detects hand landmarks, and predicts gestures in real time.

### 2. Frontend + Backend Prediction

Start the API:
```python -m uvicorn backend.api:app --reload``` 

Then open:
`frontend/index.html` 

Record a gesture video and send it to the backend for prediction.

## Requirements

Install dependencies with:
``` pip install -r requirements.txt```