import csv
import os

import cv2
import mediapipe as mp
import numpy as np

TARGET_FRAMES = 60
LANDMARK_COUNT = 21
VALUES_PER_LANDMARK = 3
FEATURE_COUNT = LANDMARK_COUNT * VALUES_PER_LANDMARK

WRIST_LANDMARK_INDEX = 0
MIDDLE_MCP_LANDMARK_INDEX = 9

MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

EMPTY_FRAME = [0.0] * FEATURE_COUNT

MIN_DETECTION_RATIO = 0.3


# NORMALIZATION


def _resample_frames(video_landmarks, target_frames=TARGET_FRAMES):
    video_landmarks = np.array(video_landmarks, dtype=np.float32)

    if len(video_landmarks) == 0:
        return np.zeros((target_frames, FEATURE_COUNT), dtype=np.float32)

    if len(video_landmarks) > target_frames:
        indices = np.linspace(0, len(video_landmarks) - 1, target_frames).astype(int)
        return video_landmarks[indices]

    padding = np.zeros((target_frames - len(video_landmarks), FEATURE_COUNT), dtype=np.float32)
    return np.concatenate([video_landmarks, padding], axis=0)


def _center_hand(video_landmarks):
    video = video_landmarks.reshape(-1, LANDMARK_COUNT, VALUES_PER_LANDMARK)

    wrist = video[:, WRIST_LANDMARK_INDEX:WRIST_LANDMARK_INDEX + 1, :]
    video = video - wrist

    return video.reshape(-1, FEATURE_COUNT)


def _scale_hand(video_landmarks):
    video = video_landmarks.reshape(-1, LANDMARK_COUNT, VALUES_PER_LANDMARK)

    wrist = video[:, WRIST_LANDMARK_INDEX, :]
    middle_mcp = video[:, MIDDLE_MCP_LANDMARK_INDEX, :]

    scale = np.linalg.norm(middle_mcp - wrist, axis=1)
    scale[scale == 0] = 1.0

    video = video / scale[:, None, None]

    return video.reshape(-1, FEATURE_COUNT)


def normalize_video(video_landmarks, target_frames=TARGET_FRAMES):
    video_landmarks = _resample_frames(video_landmarks, target_frames)
    video_landmarks = _center_hand(video_landmarks)
    video_landmarks = _scale_hand(video_landmarks)

    return video_landmarks


# ====================================================


mp_hands = mp.solutions.hands


def extract_landmarks_from_frame(frame, hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if not result.multi_hand_landmarks:
        return EMPTY_FRAME.copy(), None

    hand_landmarks = result.multi_hand_landmarks[0]
    frame_landmarks = []

    for landmark in hand_landmarks.landmark:
        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

    return frame_landmarks, hand_landmarks


def process_video(video_path):
    video_landmarks = []
    detected_frames = 0
    total_frames = 0

    cap = cv2.VideoCapture(video_path)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
                        ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            total_frames += 1

            frame_landmarks, hand_landmarks = extract_landmarks_from_frame(frame, hands)

            if hand_landmarks:
                detected_frames += 1

            video_landmarks.append(frame_landmarks)

    cap.release()

    normalized = normalize_video(video_landmarks, target_frames=TARGET_FRAMES)

    if total_frames == 0:
        detection_ratio = 0.0
    else:
        detection_ratio = detected_frames / total_frames

    return normalized, detection_ratio


# DATASET GENERATOR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_directory = os.path.join(BASE_DIR, "dataset", "raw_videos")
target_directory = os.path.join(BASE_DIR, "dataset", "processed")

if __name__ == "__main__":
    os.makedirs(target_directory, exist_ok=True)
    labels = []

    for gesture in os.listdir(raw_directory):
        gesture_path = os.path.join(raw_directory, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for file in os.listdir(gesture_path):
            if not file.endswith(".mp4"):
                continue
            video_path = os.path.join(gesture_path, file)

            data, detection_ratio = process_video(video_path)
            filename = f"{gesture}_{file.split('.')[0]}.npy"
            save_path = os.path.join(target_directory, filename)

            np.save(save_path, data)
            labels.append([filename, gesture])

    # SAVE LABELS

    with open(os.path.join(target_directory, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        writer.writerows(labels)


def create_features(data):
    motion = np.diff(data, axis=0)

    # shape/static
    first_position = data[0]
    middle_position = data[len(data) // 2]
    last_position = data[-1]
    mean_position = np.mean(data, axis=0)

    # movement
    motion_energy = np.sum(np.abs(motion), axis=0)
    motion_magnitude = np.linalg.norm(motion, axis=1)
    max_motion = np.max(np.abs(motion), axis=0)

    return np.concatenate([
        first_position,
        middle_position,
        last_position,
        mean_position,
        motion_energy,
        motion_magnitude,
        max_motion
    ])


def get_videos_and_labels():
    x = []
    y = []

    labels_path = os.path.join(target_directory, "labels.csv")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Could not find labels.csv at: {labels_path}\n"
            "You need to process your dataset first by running:\n"
            "python -m backend.video_processor"
        )

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            file_name = row["file"]
            label = row["label"]

            file_path = os.path.join(target_directory, file_name)

            data = np.load(file_path)  # shape (TARGET_FRAMES, FEATURE_COUNT)
            features = create_features(data)
            x.append(features)
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    return x, y
