import cv2
import mediapipe as mp
import backend.video_processor as video_processor
from backend.KNeighClassifier import predict

WINDOW_SIZE = 40
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

x, y = video_processor.get_videos_and_labels()
print(x.shape)
current_prediction = ""
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    live_landmarks = []
    live_detections = []

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        frame_landmarks, hand_landmarks = video_processor.extract_landmarks_from_frame(frame, hands)

        hand_detected = hand_landmarks is not None

        if hand_detected:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            current_prediction = "No hand detected"

        live_landmarks.append(frame_landmarks)
        live_detections.append(hand_detected)

        if len(live_landmarks) > WINDOW_SIZE:
            live_landmarks.pop(0)
            live_detections.pop(0)

        detection_ratio = sum(live_detections) / len(live_detections)

        if len(live_landmarks) == WINDOW_SIZE:
            if detection_ratio < video_processor.MIN_DETECTION_RATIO:
                current_prediction = "No hand detected clearly enough."
            else:
                normalized = video_processor.normalize_video(live_landmarks,
                                                             target_frames=video_processor.TARGET_FRAMES)
                features = video_processor.create_features(normalized)
                current_prediction = predict(features, x, y, k=1)[0]

        cv2.putText(frame, current_prediction, (30, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0),
                    thickness=2)
        cv2.imshow("Hand scan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
