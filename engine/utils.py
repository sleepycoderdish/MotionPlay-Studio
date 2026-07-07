"""Shared constants, math helpers, drawing helpers and sound utilities.

Everything in this module is stateless or self-contained so any other
engine module (or main.py) can import from here without circular imports.
"""

import os
import sys
import wave
import struct
import math
from collections import deque

import numpy as np
import cv2

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def project_root():
    """Root folder of the project, works whether run from source or frozen."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(*parts):
    return os.path.join(project_root(), *parts)


ASSETS_DIR = resource_path("assets")
SOUNDS_DIR = resource_path("assets", "sounds")
MODELS_DIR = resource_path("assets", "models")
DATA_DIR = resource_path("data")
HIGHSCORE_FILE = os.path.join(DATA_DIR, "highscores.json")

HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# --------------------------------------------------------------------------- #
# Window / arena constants
# --------------------------------------------------------------------------- #

WINDOW_NAME = "MotionPlay Studio"

WIDTH, HEIGHT = 1080, 680

HUD_TOP = 64            # top HUD bar height
ARENA_MARGIN = 18        # gap between HUD/window edge and playable arena

ARENA_LEFT = ARENA_MARGIN
ARENA_RIGHT = WIDTH - ARENA_MARGIN
ARENA_TOP = HUD_TOP + ARENA_MARGIN
ARENA_BOTTOM = HEIGHT - ARENA_MARGIN

ARENA_WIDTH = ARENA_RIGHT - ARENA_LEFT
ARENA_HEIGHT = ARENA_BOTTOM - ARENA_TOP

FPS_TARGET = 60

# Ball
BALL_RADIUS = 14
BALL_BASE_SPEED = 360.0      # px / second
BALL_MAX_SPEED = 950.0
BALL_SPEED_INCREMENT = 1.06   # multiplier applied on every paddle hit

# Paddles (circular mallets)
PADDLE_RADIUS = 38
PADDLE_LANE_DEPTH = 110       # how far a paddle may travel inward from its edge
PADDLE_MAX_SPEED = 1400.0     # px / second, used for AI easing
PLAYER_PADDLE_SPEED = 2600.0  # px / second -- fast enough to feel instant, but still
                              # glides rather than teleporting, which smooths out any
                              # single noisy hand-tracking frame

MAX_SCORE = 7

# Player identifiers
PLAYER = "player"
AI = "ai"


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_cubic(t):
    t = clamp(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


def ease_out_back(t):
    t = clamp(t, 0.0, 1.0)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def vec_length(dx, dy):
    return math.hypot(dx, dy)


class Smoother:
    """Exponential-moving-average smoother for a 2D point.

    Removes jitter from noisy landmark detections while still feeling
    responsive (unlike a plain windowed average which lags behind fast
    motion).
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, point):
        if self.value is None:
            self.value = point
        else:
            self.value = (
                lerp(self.value[0], point[0], self.alpha),
                lerp(self.value[1], point[1], self.alpha),
            )
        return self.value

    def reset(self):
        self.value = None


class RollingAverage:
    """Simple rolling average, handy for FPS counters."""

    def __init__(self, size=30):
        self.size = size
        self.samples = deque(maxlen=size)

    def push(self, value):
        self.samples.append(value)

    @property
    def average(self):
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #

def draw_text(canvas, text, pos, scale=0.7, color=(255, 255, 255), thickness=2,
              font=cv2.FONT_HERSHEY_DUPLEX, align="left", shadow=True):
    """Draw text with an optional drop shadow and simple alignment."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if align == "center":
        x -= tw // 2
    elif align == "right":
        x -= tw

    if shadow:
        cv2.putText(canvas, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def draw_glow_circle(canvas, center, radius, color, intensity=0.55, layers=3):
    """Fake a neon glow by stacking progressively larger, softer circles.

    Only blends a small region-of-interest around the shape rather than
    the whole canvas -- doing a full-frame addWeighted for every glowing
    object (paddles, ball, bumpers) added up to a real per-frame cost
    that showed up as stutter in the Neon Arena.
    """
    cx, cy = center
    pad = int(radius + layers * radius * 0.55) + 2
    h, w = canvas.shape[:2]
    x0, y0 = max(0, cx - pad), max(0, cy - pad)
    x1, y1 = min(w, cx + pad), min(h, cy + pad)
    if x1 <= x0 or y1 <= y0:
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
        return

    roi = canvas[y0:y1, x0:x1]
    local_center = (cx - x0, cy - y0)
    overlay = np.zeros_like(roi)
    for i in range(layers, 0, -1):
        r = int(radius + i * radius * 0.55)
        alpha = intensity * (1 - i / (layers + 1))
        cv2.circle(overlay, local_center, r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(roi, 1.0, overlay, alpha, 0, dst=roi)
        overlay[:] = 0
    cv2.circle(roi, local_center, radius, color, -1, cv2.LINE_AA)


def draw_glow_rect(canvas, pt1, pt2, color, thickness=3, glow_size=9):
    """Draw a rectangle outline with an additive neon glow halo, blending
    only the border strip (ROI) instead of the entire canvas."""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = canvas.shape[:2]
    pad = glow_size + thickness + 2
    rx0, ry0 = max(0, x1 - pad), max(0, y1 - pad)
    rx1, ry1 = min(w, x2 + pad), min(h, y2 + pad)

    roi = canvas[ry0:ry1, rx0:rx1]
    local_pt1 = (x1 - rx0, y1 - ry0)
    local_pt2 = (x2 - rx0, y2 - ry0)
    overlay = roi.copy()
    for i in range(glow_size, 0, -2):
        cv2.rectangle(overlay, local_pt1, local_pt2, color, thickness + i, cv2.LINE_AA)
        cv2.addWeighted(roi, 0.92, overlay, 0.08, 0, dst=roi)
        overlay[:] = roi
    cv2.rectangle(roi, local_pt1, local_pt2, color, thickness, cv2.LINE_AA)


def draw_gradient_rect(canvas, pt1, pt2, color_top, color_bottom):
    x1, y1 = pt1
    x2, y2 = pt2
    h = max(1, y2 - y1)
    for i in range(h):
        t = i / h
        color = tuple(int(lerp(color_top[c], color_bottom[c], t)) for c in range(3))
        cv2.line(canvas, (x1, y1 + i), (x2, y1 + i), color, 1)


def draw_rounded_rect(canvas, pt1, pt2, color, radius=14, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    r = max(r, 1)

    if thickness < 0:
        cv2.rectangle(canvas, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def blend_overlay(canvas, alpha=0.45, color=(0, 0, 0)):
    """Darken the whole canvas -- used behind menu panels."""
    overlay = np.full_like(canvas, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)


class Button:
    """A clickable / hoverable rectangular UI button."""

    def __init__(self, rect, label, action=None, sub_label=None, enabled=True):
        self.rect = rect  # (x, y, w, h)
        self.label = label
        self.sub_label = sub_label
        self.action = action
        self.enabled = enabled
        self.hover_t = 0.0  # animated hover progress 0..1
        self.was_hovering = False

    def contains(self, px, py):
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h

    def update_hover(self, hovered, dt, speed=8.0):
        target = 1.0 if hovered and self.enabled else 0.0
        self.hover_t += (target - self.hover_t) * min(1.0, dt * speed)

    def draw(self, canvas, accent_color=(0, 200, 255), text_color=(235, 235, 235),
              base_color=(35, 38, 48), border_color=(80, 84, 100)):
        x, y, w, h = self.rect
        t = self.hover_t
        fill = tuple(int(lerp(base_color[c], accent_color[c] * 0.35, t)) for c in range(3))
        border = tuple(int(lerp(border_color[c], accent_color[c], t)) for c in range(3))

        if not self.enabled:
            fill = (28, 28, 32)
            border = (55, 55, 60)
            text_color = (110, 110, 115)

        draw_rounded_rect(canvas, (x, y), (x + w, y + h), fill, radius=12, thickness=-1)
        draw_rounded_rect(canvas, (x, y), (x + w, y + h), border, radius=12, thickness=2)

        cy = y + h // 2 + (7 if not self.sub_label else -3)
        draw_text(canvas, self.label, (x + w // 2, cy), scale=0.62, thickness=2,
                   color=text_color, align="center", shadow=False)
        if self.sub_label:
            draw_text(canvas, self.sub_label, (x + w // 2, y + h - 10), scale=0.42,
                       thickness=1, color=(170, 170, 175), align="center", shadow=False)


# --------------------------------------------------------------------------- #
# Sound generation + playback
# --------------------------------------------------------------------------- #

_SAMPLE_RATE = 44100


def _synth_tone(freq, duration, volume=0.5, wave_type="sine", fade=True, sample_rate=_SAMPLE_RATE):
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    if wave_type == "sine":
        wave_data = np.sin(2 * np.pi * freq * t)
    elif wave_type == "square":
        wave_data = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "triangle":
        wave_data = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    else:
        wave_data = np.sin(2 * np.pi * freq * t)

    if fade:
        fade_len = max(1, int(n * 0.08))
        envelope = np.ones(n)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        wave_data *= envelope

    return wave_data * volume


def _synth_sequence(notes, sample_rate=_SAMPLE_RATE):
    chunks = [_synth_tone(freq, dur, volume=vol, wave_type=wt, sample_rate=sample_rate)
              for freq, dur, vol, wt in notes]
    return np.concatenate(chunks)


def _write_wav(path, samples, sample_rate=_SAMPLE_RATE):
    samples = np.clip(samples, -1.0, 1.0)
    ints = (samples * 32767).astype(np.int16)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack("<%dh" % len(ints), *ints))


_SOUND_DEFS = {
    "click": lambda: _synth_tone(880, 0.06, 0.35, "sine"),
    "hover": lambda: _synth_tone(600, 0.04, 0.2, "sine"),
    "paddle_hit": lambda: _synth_tone(520, 0.08, 0.55, "triangle"),
    "wall_hit": lambda: _synth_tone(320, 0.06, 0.4, "sine"),
    "goal": lambda: _synth_sequence([(392, 0.10, 0.5, "square"), (523, 0.10, 0.5, "square"),
                                      (659, 0.16, 0.55, "square")]),
    "powerup": lambda: _synth_sequence([(660, 0.06, 0.4, "sine"), (880, 0.06, 0.4, "sine"),
                                         (1046, 0.10, 0.45, "sine")]),
    "countdown_tick": lambda: _synth_tone(700, 0.12, 0.45, "sine"),
    "countdown_go": lambda: _synth_sequence([(700, 0.08, 0.5, "square"), (1100, 0.22, 0.55, "square")]),
    "win": lambda: _synth_sequence([(523, 0.12, 0.5, "triangle"), (659, 0.12, 0.5, "triangle"),
                                     (784, 0.12, 0.5, "triangle"), (1046, 0.30, 0.6, "triangle")]),
    "lose": lambda: _synth_sequence([(400, 0.18, 0.5, "sine"), (300, 0.18, 0.5, "sine"),
                                      (220, 0.30, 0.5, "sine")]),
}


def ensure_hand_model():
    """Download the MediaPipe HandLandmarker model bundle if it isn't
    cached locally yet. Returns True if the model is available on disk,
    False if it could not be fetched (e.g. no internet on first run) --
    the caller is expected to degrade gracefully in that case."""
    if os.path.exists(HAND_MODEL_PATH) and os.path.getsize(HAND_MODEL_PATH) > 0:
        return True
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        import urllib.request
        tmp_path = HAND_MODEL_PATH + ".part"
        urllib.request.urlretrieve(HAND_MODEL_URL, tmp_path)
        os.replace(tmp_path, HAND_MODEL_PATH)
        return True
    except Exception:
        return False


def ensure_sound_assets():
    """Generate the small procedural WAV library if it isn't on disk yet."""
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    for name, generator in _SOUND_DEFS.items():
        path = os.path.join(SOUNDS_DIR, f"{name}.wav")
        if not os.path.exists(path):
            try:
                _write_wav(path, generator())
            except Exception:
                pass  # never let asset generation crash the game


class SoundManager:
    """Best-effort sound playback that never raises.

    Uses the stdlib `winsound` module (Windows only). On any other
    platform, or if anything goes wrong, playback silently becomes a
    no-op so the game keeps running without audio.
    """

    def __init__(self):
        self.enabled = True
        try:
            import winsound
            self._winsound = winsound
        except ImportError:
            self._winsound = None
            self.enabled = False

        ensure_sound_assets()

    def play(self, name):
        if not self.enabled or self._winsound is None:
            return
        path = os.path.join(SOUNDS_DIR, f"{name}.wav")
        if not os.path.exists(path):
            return
        try:
            self._winsound.PlaySound(path, self._winsound.SND_FILENAME | self._winsound.SND_ASYNC)
        except Exception:
            pass  # never let a bad codec / device kill the game

    def toggle(self):
        if self._winsound is not None:
            self.enabled = not self.enabled
