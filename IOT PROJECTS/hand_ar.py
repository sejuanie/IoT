import collections
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

CAM_SOURCE = 0  # laptop webcam (Path A).
# CAM_SOURCE = "http://localhost:8080/video"  # phone over USB (your path: B)
FRAME_W, FRAME_H = 1280, 720

TRAIL_LENGTH = 18 #points remembered per fingertip trail
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb tip, index tip, middle tip, ring tip, pinky tip
INDEX_TIP_ID = 8
THUMB_TIP_ID = 4

SKELETON_COLOR = (255, 0, 180)
SKELETON_GLOW_COLOR = (255, 120, 255)
JOINT_COLOR = (0, 255, 255)

BRUSH_RADIUS = 6
DRAW_COLOR = (80, 255, 255)         # neon yellow paint
DRAW_GLOW_COLOR = (255, 255, 80)
DRAW_STROKE_THICKNESS = 3

TRAIL_COLORS = [
    (255, 80, 80), #thumb - blue-ish
    (80, 255, 255), #index - yello
    (80, 255, 80), #middle - green
    (255, 80, 255), #ring - magenta
    (80, 160, 255) #pinky - orange
]

ORB_RADIUS = 40
ORB_COLOR = (255, 220, 0)
ORB_GLOW_COLOR = (255, 255, 150)
PINCH_THRESHOLD = 40

HAND_CONNECTIONS = [
    (c.start, c.end)
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]


def glow_line(img, p1, p2, color, glow_color, thickness=2):
    cv2.line(img, p1, p2, glow_color, thickness + 6, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def glow_circle(img, center, radius, color, glow_color):
    cv2.circle(img, center, radius + 8, glow_color, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)


def count_fingers(pts):
    
    wrist = pts[0]
    extended = 0

    # Four fingers: tip farther from the wrist than its pip joint == extended.
    finger_tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]

    for tip_id, pip_id in finger_tips_pips:
        if (
            math.hypot(
                pts[tip_id][0] - wrist[0],
                pts[tip_id][1] - wrist[1]
            )
            >
            math.hypot(
                pts[pip_id][0] - wrist[0],
                pts[pip_id][1] - wrist[1]
            )
        ):
            extended += 1

    pinky_mcp = pts[17]

    thumb_tip_dist = math.hypot(
        pts[4][0] - pinky_mcp[0],
        pts[4][1] - pinky_mcp[1]
    )

    thumb_ip_dist = math.hypot(
        pts[3][0] - pinky_mcp[0],
        pts[3][1] - pinky_mcp[1]
    )

    if thumb_tip_dist > thumb_ip_dist:
        extended += 1

    return extended


def draw_text_with_bg(
    img,
    text,
    org,
    scale=0.9,
    color=(255, 255, 255),
    bg=(0, 0, 0),
    thickness=2,
    alpha=0.55
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x, y = org
    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x - 10, y - th - 10),
        (x + tw + 10, y + baseline + 6),
        bg,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        img,
        1 - alpha,
        0,
        img
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return

    print("Downloading hand landmark model (one-time, ~8 MB)...")

    import urllib.request

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model downloaded.")


def main():
    ensure_model()

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    if isinstance(CAM_SOURCE, int):
        cap = cv2.VideoCapture(
            CAM_SOURCE,
            cv2.CAP_DSHOW
        )
    else:
        cap = cv2.VideoCapture(CAM_SOURCE)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_W
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_H
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check CAM_SOURCE / camera permissions."
        )

    #trails[hand_index][finger_index] = deque of (x,y) points
    trails = [
        [
            collections.deque(maxlen=TRAIL_LENGTH)
            for _ in FINGERTIP_IDS
        ]
        for _ in range(2)
    ]

    orb_pos = None #set once we know frame size
    orb_held = False
    drawing = False      # 'd' toggles the paintbrush on/off
    prev_brush_pt = None
    brush_canvas = None
    prev_time = time.time()

    print("Hand & Finger AR Tracker running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        if orb_pos is None:
            orb_pos = [w // 2, h // 2]

        if brush_canvas is None:
            brush_canvas = np.zeros(
                (h, w, 3),
                np.uint8
            )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        overlay_layer = np.zeros_like(frame)

        pinch_active_any = False
        index_tip_px = None

        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(
                result.hand_landmarks[:2]
            ):
                handedness_label = "Right"

                if (
                    result.handedness
                    and hand_idx < len(result.handedness)
                ):
                    handedness_label = (
                        result.handedness[hand_idx][0].category_name
                    )

                pts = [
                    (
                        int(p.x * w),
                        int(p.y * h)
                    )
                    for p in hand_landmarks
                ]

                for a, b in HAND_CONNECTIONS:
                    glow_line(
                        overlay_layer,
                        pts[a],
                        pts[b],
                        SKELETON_COLOR,
                        SKELETON_GLOW_COLOR,
                        2
                    )

                for p in pts:
                    glow_circle(
                        overlay_layer,
                        p,
                        4,
                        JOINT_COLOR,
                        (0, 255, 255)
                    )

                # --- fingertip particle trails ---
                for f_idx, tip_id in enumerate(FINGERTIP_IDS):
                    trails[hand_idx][f_idx].append(
                        pts[tip_id]
                    )

                    trail = trails[hand_idx][f_idx]
                    n = len(trail)

                    for i in range(1, n):
                        fade = i / n
                        radius = max(
                            1,
                            int(6 * fade)
                        )

                        color = tuple(
                            int(c * fade)
                            for c in TRAIL_COLORS[f_idx]
                        )

                        cv2.circle(
                            overlay_layer,
                            trail[i],
                            radius,
                            color,
                            -1,
                            cv2.LINE_AA
                        )

                    glow_circle(
                        overlay_layer,
                        pts[tip_id],
                        8,
                        TRAIL_COLORS[f_idx],
                        TRAIL_COLORS[f_idx]
                    )

                # --- gesture readout ---
                finger_count = count_fingers(pts)

                thumb_tip = pts[THUMB_TIP_ID]
                index_tip = pts[INDEX_TIP_ID]

                pinch_dist = math.hypot(
                    thumb_tip[0] - index_tip[0],
                    thumb_tip[1] - index_tip[1]
                )

                is_pinching = (
                    pinch_dist < PINCH_THRESHOLD
                )

                label_x = min(
                    pts[0][0],
                    w - 260
                )

                label_y = max(
                    pts[0][1] + 60,
                    40
                )

                status = (
                    f"{handedness_label} hand: "
                    f"{finger_count} fingers"
                )

                if is_pinching:
                    status += " | PINCH"
                    pinch_active_any = True

                draw_text_with_bg(
                    overlay_layer,
                    status,
                    (label_x, label_y),
                    scale=0.7,
                    color=(255, 255, 255),
                    bg=(40, 0, 40)
                )

                if (
                    handedness_label == "Right"
                    or not result.handedness
                ):
                    index_tip_px = index_tip

                elif index_tip_px is None:
                    index_tip_px = index_tip

        # --- virtual object interaction ---
        if drawing and index_tip_px is not None:
            orb_pos[0], orb_pos[1] = index_tip_px

            orb_color = (
                ORB_GLOW_COLOR
                if orb_held
                else ORB_COLOR
            )

            pulse = int(
                6 * math.sin(time.time() * 4)
            )

            glow_circle(
                overlay_layer,
                tuple(orb_pos),
                ORB_RADIUS + pulse,
                ORB_COLOR,
                orb_color
            )

        if drawing and index_tip_px is not None:
            if prev_brush_pt is not None:
                glow_line(
                    brush_canvas,
                    prev_brush_pt,
                    index_tip_px,
                    DRAW_COLOR,
                    DRAW_GLOW_COLOR,
                    DRAW_STROKE_THICKNESS
                )

            prev_brush_pt = index_tip_px

        else:
            prev_brush_pt = None

        overlay_layer = np.maximum(
            overlay_layer,
            brush_canvas
        )

        frame = cv2.addWeighted(
            frame,
            0.65,
            overlay_layer,
            1.0,
            0
        )

        # FPS + title HUD
        now = time.time()

        fps = 1.0 / max(
            now - prev_time,
            1e-6
        )

        prev_time = now

        draw_text_with_bg(
            frame,
            "Hand & Finger AR Tracker",
            (20, 40),
            scale=1.0,
            color=(0, 255, 255),
            bg=(0, 0, 0)
        )
        
        draw_text_with_bg(frame, f"FPS: {fps:.0f}", (20, 80), scale=0.7, 
            color=(200, 200, 200), bg=(0, 0, 0), alpha=0.4)

        draw_text_with_bg(
            frame,
            "D: draw " + ("ON" if drawing else "OFF"),
            (w - 160, h - 20),
            scale=0.6,
            color=(
                (80, 255, 255)
                if drawing
                else (200, 200, 200)
            ),
            bg=(0, 0, 0),
            alpha=0.4
        )

        cv2.imshow(
            "Hand & Finger AR Tracker",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('d'):
            drawing = not drawing
            prev_brush_pt = None

        if key in (ord('q'), 27):
            break

        if key == ord('r'):
            orb_pos = [
                w // 2,
                h // 2
            ]

            orb_held = False

            brush_canvas.fill(0)
            prev_brush_pt = None

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()