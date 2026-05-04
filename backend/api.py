from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile

import backend.video_processor as video_processor
from backend.KNeighClassifier import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

x, y = video_processor.get_videos_and_labels()


@app.get("/")
def root():
    return {"message": "Hand Gesture API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "samples_loaded": len(y)
    }


@app.post("/predict/")
def predict_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    temp_path = None

    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        data, detection_ratio = video_processor.process_video(temp_path)

        if detection_ratio < video_processor.MIN_DETECTION_RATIO:
            raise HTTPException(status_code=400,
                                detail="No hand detected clearly enough. Try again with your hand clearly visible.")
        features = video_processor.create_features(data)

        prediction, distance = predict(features, x, y, k=1)

        return {
            "prediction": prediction,
            "distance": float(distance),
            "detection_ratio": float(detection_ratio)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
