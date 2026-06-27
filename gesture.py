import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

GESTURES = {
    1: "hands down",
    2: "hands up",
    3: "right hand up",
    4: "left hand up",
    5: "hands crossed",
    6: "leg raised",
}

# landmarks we use: shoulders, elbows, wrists, hips, knees, ankles
USED = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
TEMPLATE_FILE = "templates.json"
K = 3


def make_landmarker():
    base = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(opts)


# Build a feature vector that does not depend on body size or position.
# We move the hip center to (0,0) and divide by the torso length.
def get_feature(lm):
    shoulder_x = (lm[11].x + lm[12].x) / 2
    shoulder_y = (lm[11].y + lm[12].y) / 2
    hip_x = (lm[23].x + lm[24].x) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    torso = ((shoulder_x - hip_x) ** 2 + (shoulder_y - hip_y) ** 2) ** 0.5
    if torso < 1e-6:
        return None

    feature = []
    for i in USED:
        feature.append((lm[i].x - hip_x) / torso)
        feature.append((lm[i].y - hip_y) / torso)
    return feature


def all_visible(lm, threshold=0.5):
    return all(lm[i].visibility > threshold for i in USED)


def draw_skeleton(frame, lm):
    h, w = frame.shape[:2]
    pixels = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pixels[a], pixels[b], (0, 255, 0), 2)
    for i in USED:
        cv2.circle(frame, pixels[i], 5, (0, 0, 255), -1)


# k nearest neighbours: find closest templates, then majority vote
def classify(feature, templates):
    distances = []
    for t in templates:
        d = np.linalg.norm(np.array(feature) - np.array(t["feature"]))
        distances.append((d, t["gesture"]))
    distances.sort(key=lambda item: item[0])

    votes = {}
    for _, gesture in distances[:K]:
        votes[gesture] = votes.get(gesture, 0) + 1
    return max(votes, key=votes.get)


def teach_mode():
    landmarker = make_landmarker()
    cap = cv2.VideoCapture(0)
    frame_index = 0

    templates = []
    try:
        with open(TEMPLATE_FILE) as f:
            templates = json.load(f)
    except FileNotFoundError:
        pass

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, frame_index)
        frame_index += 1

        lm = result.pose_landmarks[0] if result.pose_landmarks else None
        if lm:
            draw_skeleton(frame, lm)

        cv2.putText(frame, "1-6 = save pose, s = save file, ESC = quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"templates: {len(templates)}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Teach mode", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s"):
            with open(TEMPLATE_FILE, "w") as f:
                json.dump(templates, f)
            print("Saved", len(templates), "templates to", TEMPLATE_FILE)
        elif ord("1") <= key <= ord("6") and lm:
            gesture = key - ord("0")
            if all_visible(lm):
                feature = get_feature(lm)
                if feature:
                    templates.append({"gesture": gesture, "feature": feature})
                    print(f"Captured gesture {gesture} ({GESTURES[gesture]}), total {len(templates)}")
            else:
                print("Not all body points visible, step back.")

    cap.release()
    cv2.destroyAllWindows()


def run_mode():
    try:
        with open(TEMPLATE_FILE) as f:
            templates = json.load(f)
    except FileNotFoundError:
        print("No templates yet. Run: python gesture.py teach")
        return
    if not templates:
        print("Template file is empty. Teach some poses first.")
        return

    landmarker = make_landmarker()
    cap = cv2.VideoCapture(0)
    frame_index = 0
    shot_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, frame_index)
        frame_index += 1

        label = "..."
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            draw_skeleton(frame, lm)
            if all_visible(lm):
                feature = get_feature(lm)
                if feature:
                    gesture = classify(feature, templates)
                    label = f"{gesture}: {GESTURES[gesture]}"
            else:
                label = "step back (full body needed)"

        cv2.putText(frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "s = save screenshot, ESC = quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s"):
            safe_name = label.replace(": ", "_").replace(" ", "_")
            filename = f"shot_{shot_count}_{safe_name}.png"
            cv2.imwrite(filename, frame)
            shot_count += 1
            print("Saved screenshot:", filename)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "teach":
        teach_mode()
    else:
        run_mode()
