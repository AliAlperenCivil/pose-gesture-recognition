import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# skeleton lines between landmark indices (shoulders, arms, hips, legs)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_index = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_index)
    frame_index += 1

    if result.pose_landmarks:
        points = result.pose_landmarks[0]
        h, w = frame.shape[:2]
        pixels = [(int(p.x * w), int(p.y * h)) for p in points]

        for a, b in CONNECTIONS:
            cv2.line(frame, pixels[a], pixels[b], (0, 255, 0), 2)
        for x, y in pixels:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
