import cv2
import mediapipe as mp
import time
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "models",
    "hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    timestamp_ms = int(time.time() * 1000)

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()