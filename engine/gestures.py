"""Gesture control using MediaPipe's HandLandmarker (Tasks API).

Replaces the legacy HSV colour-blob tracking entirely. We track a
single hand and use the average of the palm-base landmarks (wrist +
the four MCP knuckles) as the hand "center" -- this point is far more
stable than a fingertip because it barely moves when the fingers curl
or spread, which keeps the paddle from jittering.

The model bundle (~8MB) is downloaded on first run via
`engine.utils.ensure_hand_model`. If that download fails (e.g. no
internet available yet), gesture tracking simply reports "unavailable"
and `game_state.Game` falls back to mouse control instead of crashing.
"""

import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from engine.utils import Smoother, clamp, ensure_hand_model, HAND_MODEL_PATH


# Landmarks that make up a stable palm-center estimate.
_PALM_LANDMARKS = (0, 5, 9, 13, 17)


class GestureController:
    def __init__(self, smoothing_alpha=0.55, detection_confidence=0.6, tracking_confidence=0.5):
        self.available = ensure_hand_model()
        self._landmarker = None

        if self.available:
            try:
                options = vision.HandLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=detection_confidence,
                    min_hand_presence_confidence=tracking_confidence,
                    min_tracking_confidence=tracking_confidence,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
            except Exception:
                self.available = False
                self._landmarker = None

        self.smoother = Smoother(alpha=smoothing_alpha)
        self.last_point = None       # smoothed (x, y) in pixel coords of the given frame
        self.last_norm_point = None  # smoothed (nx, ny) normalized 0..1
        self.detected = False

    def process(self, frame_bgr):
        """Run hand detection on a BGR frame. Returns smoothed (x, y) pixel
        coordinates of the hand center, or None if no hand is visible."""
        if not self.available or self._landmarker is None:
            self.detected = False
            return None

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception:
            self.detected = False
            return None

        if not result.hand_landmarks:
            self.detected = False
            return None

        landmarks = result.hand_landmarks[0]
        nx = sum(landmarks[i].x for i in _PALM_LANDMARKS) / len(_PALM_LANDMARKS)
        ny = sum(landmarks[i].y for i in _PALM_LANDMARKS) / len(_PALM_LANDMARKS)
        nx = clamp(nx, 0.0, 1.0)
        ny = clamp(ny, 0.0, 1.0)

        smoothed_norm = self.smoother.update((nx, ny))
        self.last_norm_point = smoothed_norm
        self.last_point = (smoothed_norm[0] * w, smoothed_norm[1] * h)
        self.detected = True
        return self.last_point

    def reset(self):
        self.smoother.reset()
        self.last_point = None
        self.last_norm_point = None
        self.detected = False

    def draw_debug(self, frame_bgr, radius=10, color=(0, 255, 140)):
        if self.last_point is None:
            return
        p = (int(self.last_point[0]), int(self.last_point[1]))
        cv2.circle(frame_bgr, p, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, p, radius + 4, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()


class CameraWorker(threading.Thread):
    """Grabs camera frames and runs hand detection on a background thread.

    MediaPipe inference (tens of milliseconds per frame on CPU) used to
    run inline in the render loop, which capped the *entire* game --
    menus included, since a frame was grabbed every tick regardless of
    state -- to whatever FPS the model could sustain. Decoupling capture
    from rendering lets the UI redraw at a smooth, uncapped rate while
    tracking updates land in the background whenever they're ready.
    """

    def __init__(self, camera, gesture: GestureController):
        super().__init__(daemon=True)
        self.camera = camera
        self.gesture = gesture
        self.enabled = False  # only run the (expensive) detector while actually needed
        self._lock = threading.Lock()
        self._frame = None
        self._running = True

    def run(self):
        while self._running:
            if self.camera is None or not self.camera.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            if self.enabled:
                self.gesture.process(frame)
                self.gesture.draw_debug(frame)
            else:
                self.gesture.detected = False

            with self._lock:
                self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
