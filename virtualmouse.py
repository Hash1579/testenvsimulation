import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------
# Screen size
# ----------------------------
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# ----------------------------
# Cursor & smoothing state
# ----------------------------
cursor_x, cursor_y = screen_width // 2, screen_height // 2
smooth_x, smooth_y = cursor_x, cursor_y

alpha = 0.45

prev_ix, prev_iy = None, None

# ----------------------------
# Load MediaPipe hand model
# ----------------------------
model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.4
)

detector = vision.HandLandmarker.create_from_options(options)

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# ----------------------------
# Gesture states
# ----------------------------
clicked = False
alt_active = False
alt_start_time = 0
ALT_HOLD_DURATION = 4.0  # seconds

# ----------------------------
# Main loop
# ----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # ---- Draw ALL hand landmarks ----
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Required landmarks
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            ring_tip = hand_landmarks[16]

            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            rx, ry = int(ring_tip.x * w), int(ring_tip.y * h)

            # ------------------------------------------------
            # Mouse movement (velocity based)
            # ------------------------------------------------
            if prev_ix is None:
                prev_ix, prev_iy = ix, iy

            dx = ix - prev_ix
            dy = iy - prev_iy
            prev_ix, prev_iy = ix, iy

            speed = np.hypot(dx, dy)

            gain = 0.12 + 0.006 * (speed ** 1.4)
            prediction_factor = 1.3

            cursor_x += dx * gain * prediction_factor
            cursor_y += dy * gain * prediction_factor

            cursor_x = max(0, min(screen_width, cursor_x))
            cursor_y = max(0, min(screen_height, cursor_y))

            smooth_x = alpha * cursor_x + (1 - alpha) * smooth_x
            smooth_y = alpha * cursor_y + (1 - alpha) * smooth_y

            pyautogui.moveTo(smooth_x, smooth_y)

            # ----------------------------
            # Left click (index + thumb)
            # ----------------------------
            click_dist = np.hypot(ix - tx, iy - ty)

            if click_dist < 50 and not clicked:
                pyautogui.click()
                clicked = True

            if click_dist > 40:
                clicked = False

            # ----------------------------
            # ALT + TAB (4-second HOLD)
            # ----------------------------
            alt_tab_dist = np.hypot(rx - tx, ry - ty)
            current_time = time.time()

            # Trigger ALT+TAB once
            if alt_tab_dist < 45 and not alt_active:
                pyautogui.keyDown("alt")
                pyautogui.press("tab")
                alt_active = True
                alt_start_time = current_time

            # Auto-release ALT after 4 seconds
            if alt_active and (current_time - alt_start_time) > ALT_HOLD_DURATION:
                pyautogui.keyUp("alt")
                alt_active = False

            # Debug info
            cv2.putText(
                frame,
                f"Click:{int(click_dist)}  AltTab:{int(alt_tab_dist)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()