"""Map / theme definitions.

Each theme only changes the visual dressing of the arena (colors,
background, glow style, optional decorative bumper obstacles) -- the
gameplay mechanics are identical across every map, as required.
"""

import math
import time

import cv2
import numpy as np

from engine.utils import (
    ARENA_LEFT, ARENA_RIGHT, ARENA_TOP, ARENA_BOTTOM, ARENA_WIDTH, ARENA_HEIGHT,
    WIDTH, HEIGHT, draw_gradient_rect, draw_glow_rect, draw_glow_circle, clamp,
)


class Bumper:
    """A small static obstacle the puck bounces off elastically."""

    def __init__(self, x, y, radius=26):
        self.x = x
        self.y = y
        self.radius = radius


class MapTheme:
    def __init__(self, key, name, bg_top, bg_bottom, table_color, border_color,
                 center_color, ball_color, ball_glow, paddle_left_color,
                 paddle_right_color, accent, neon=False, ice=False, bumpers=None):
        self.key = key
        self.name = name
        self.bg_top = bg_top
        self.bg_bottom = bg_bottom
        self.table_color = table_color
        self.border_color = border_color
        self.center_color = center_color
        self.ball_color = ball_color
        self.ball_glow = ball_glow
        self.paddle_left_color = paddle_left_color
        self.paddle_right_color = paddle_right_color
        self.accent = accent
        self.neon = neon
        self.ice = ice
        self.bumpers = bumpers or []

    def render_background(self, canvas):
        draw_gradient_rect(canvas, (0, 0), (WIDTH, HEIGHT), self.bg_top, self.bg_bottom)

    def render_arena(self, canvas):
        pt1 = (ARENA_LEFT, ARENA_TOP)
        pt2 = (ARENA_RIGHT, ARENA_BOTTOM)

        # table surface -- blend only within the arena rect, not the whole canvas
        table_roi = canvas[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        table_overlay = np.full_like(table_roi, self.table_color, dtype=np.uint8)
        cv2.addWeighted(table_overlay, 0.92, table_roi, 0.08, 0, dst=table_roi)

        if self.ice:
            self._render_ice_shine(canvas, pt1, pt2)

        # center line + circle
        cx = (ARENA_LEFT + ARENA_RIGHT) // 2
        cy = (ARENA_TOP + ARENA_BOTTOM) // 2
        cv2.line(canvas, (cx, ARENA_TOP), (cx, ARENA_BOTTOM), self.center_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 70, self.center_color, 2, cv2.LINE_AA)

        # goal mouths (visual only, highlight the scoring edges)
        goal_h = ARENA_HEIGHT * 0.38
        goal_y1 = int(cy - goal_h / 2)
        goal_y2 = int(cy + goal_h / 2)
        cv2.line(canvas, (ARENA_LEFT, goal_y1), (ARENA_LEFT, goal_y2), self.accent, 5, cv2.LINE_AA)
        cv2.line(canvas, (ARENA_RIGHT, goal_y1), (ARENA_RIGHT, goal_y2), self.accent, 5, cv2.LINE_AA)

        # border
        if self.neon:
            draw_glow_rect(canvas, pt1, pt2, self.border_color, thickness=3, glow_size=8)
        else:
            cv2.rectangle(canvas, pt1, pt2, self.border_color, 4, cv2.LINE_AA)

        for bumper in self.bumpers:
            center = (int(bumper.x), int(bumper.y))
            if self.neon:
                draw_glow_circle(canvas, center, bumper.radius, self.accent, intensity=0.4, layers=2)
            else:
                cv2.circle(canvas, center, bumper.radius, self.table_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, center, bumper.radius, self.accent, 3, cv2.LINE_AA)

    def _render_ice_shine(self, canvas, pt1, pt2):
        # blend only the arena ROI, not the full canvas, to keep this cheap
        roi = canvas[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        t = time.time() * 0.15
        overlay = roi.copy()
        w = pt2[0] - pt1[0]
        h = pt2[1] - pt1[1]
        for i in range(4):
            offset = (i * 140 + int((t * 90) % 140))
            x = offset
            if x > w:
                continue
            pts = np.array([
                [x, h], [x + 40, h],
                [x - 60, 0], [x - 100, 0],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 255, 255))
        cv2.addWeighted(overlay, 0.05, roi, 0.95, 0, dst=roi)


def _make_maps():
    cx = (ARENA_LEFT + ARENA_RIGHT) // 2
    cy = (ARENA_TOP + ARENA_BOTTOM) // 2

    classic = MapTheme(
        key="classic", name="Classic",
        bg_top=(28, 30, 36), bg_bottom=(14, 15, 18),
        table_color=(46, 92, 48), border_color=(235, 235, 235),
        center_color=(230, 230, 230), ball_color=(240, 240, 240),
        ball_glow=(240, 240, 240),
        paddle_left_color=(60, 130, 235), paddle_right_color=(50, 200, 120),
        accent=(210, 180, 60), neon=False, ice=False, bumpers=[],
    )

    ice = MapTheme(
        key="ice", name="Ice Arena",
        bg_top=(60, 35, 20), bg_bottom=(30, 18, 10),
        table_color=(150, 90, 40), border_color=(255, 255, 255),
        center_color=(255, 255, 255), ball_color=(255, 250, 235),
        ball_glow=(255, 255, 255),
        paddle_left_color=(60, 160, 255), paddle_right_color=(255, 140, 60),
        accent=(255, 255, 255), neon=False, ice=True, bumpers=[],
    )
    # NOTE: OpenCV uses BGR -- "blue table" reads as (B,G,R) below instead.
    ice.table_color = (210, 140, 60)      # icy blue
    ice.bg_top = (70, 40, 15)
    ice.bg_bottom = (35, 20, 8)

    neon = MapTheme(
        key="neon", name="Neon Arena",
        bg_top=(25, 5, 25), bg_bottom=(5, 0, 8),
        table_color=(18, 8, 22), border_color=(255, 60, 220),
        center_color=(255, 60, 220), ball_color=(255, 255, 255),
        ball_glow=(255, 240, 60),
        paddle_left_color=(255, 180, 20), paddle_right_color=(60, 230, 255),
        accent=(60, 230, 255), neon=True, ice=False,
        bumpers=[Bumper(cx, cy - 130, 24), Bumper(cx, cy + 130, 24)],
    )

    return {m.key: m for m in (classic, ice, neon)}


MAPS = _make_maps()
MAP_ORDER = ["classic", "ice", "neon"]


def get_map(key):
    return MAPS.get(key, MAPS["classic"])
