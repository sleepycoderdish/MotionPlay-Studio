"""Core physics bodies: the puck (Ball) and the circular mallets (Paddle).

Movement is expressed in pixels/second and integrated with a delta-time
each frame, so gameplay speed stays consistent regardless of the actual
frame rate the camera/CV pipeline manages to sustain.
"""

import random
from collections import deque

import cv2

from engine.utils import (
    clamp, BALL_RADIUS, BALL_BASE_SPEED, BALL_MAX_SPEED,
    BALL_SPEED_INCREMENT, PADDLE_RADIUS, PADDLE_MAX_SPEED,
    ARENA_LEFT, ARENA_RIGHT, ARENA_TOP, ARENA_BOTTOM, draw_glow_circle,
)


class Ball:
    """The puck. Tracks its own trail for a motion-blur style render."""

    def __init__(self, x, y):
        self.origin = (x, y)
        self.x = x
        self.y = y
        self.radius = BALL_RADIUS
        self.vx = 0.0
        self.vy = 0.0
        self.trail = deque(maxlen=10)
        self.last_hit_by = None  # 'player' | 'ai' | None
        self.speed_multiplier = 1.0  # temporary boost from power-ups
        self.reset(direction_bias=random.choice([-1, 1]))

    def reset(self, direction_bias=None):
        self.x, self.y = self.origin
        self.trail.clear()
        self.last_hit_by = None
        self.speed_multiplier = 1.0

        angle_spread = 0.55  # radians either side of horizontal
        angle = random.uniform(-angle_spread, angle_spread)
        direction = direction_bias if direction_bias is not None else random.choice([-1, 1])

        import math
        speed = BALL_BASE_SPEED
        self.vx = direction * speed * (1 - abs(angle) * 0.3)
        self.vy = speed * math.sin(angle) * 2.2

    def update(self, dt):
        self.trail.append((int(self.x), int(self.y)))
        eff_vx = self.vx * self.speed_multiplier
        eff_vy = self.vy * self.speed_multiplier
        self.x += eff_vx * dt
        self.y += eff_vy * dt

    def bump_speed(self):
        speed = (self.vx ** 2 + self.vy ** 2) ** 0.5
        if speed <= 0:
            return
        new_speed = min(speed * BALL_SPEED_INCREMENT, BALL_MAX_SPEED)
        scale = new_speed / speed
        self.vx *= scale
        self.vy *= scale

    def current_speed(self):
        return ((self.vx ** 2 + self.vy ** 2) ** 0.5) * self.speed_multiplier

    def draw(self, canvas, theme):
        # motion trail
        n = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            alpha_r = int(self.radius * (0.35 + 0.5 * (i / max(1, n))))
            fade_color = tuple(int(c * (0.15 + 0.55 * (i / max(1, n)))) for c in theme.ball_color)
            cv2.circle(canvas, (tx, ty), max(2, alpha_r), fade_color, -1, cv2.LINE_AA)

        if theme.neon:
            draw_glow_circle(canvas, (int(self.x), int(self.y)), self.radius, theme.ball_glow, intensity=0.5, layers=3)
        else:
            cv2.circle(canvas, (int(self.x), int(self.y)), self.radius, theme.ball_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2, cv2.LINE_AA)


class Paddle:
    """A circular air-hockey mallet constrained to a lane on one side."""

    def __init__(self, side, y, lane_min_x, lane_max_x, color):
        self.side = side  # 'left' or 'right'
        self.radius = PADDLE_RADIUS
        self.base_radius = PADDLE_RADIUS
        self.color = color
        self.lane_min_x = lane_min_x
        self.lane_max_x = lane_max_x

        self.x = lane_min_x if side == "left" else lane_max_x
        self.y = y
        self.prev_x, self.prev_y = self.x, self.y
        self.vx, self.vy = 0.0, 0.0

        self.frozen_until = 0.0
        self.bigger_until = 0.0

    def move_toward(self, target_x, target_y, dt, max_speed=PADDLE_MAX_SPEED):
        """Eased movement toward a target -- used by the AI paddle."""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist < 1e-3:
            return
        step = min(dist, max_speed * dt)
        self.x += dx / dist * step
        self.y += dy / dist * step
        self.x = clamp(self.x, self.lane_min_x, self.lane_max_x)
        self.y = clamp(self.y, ARENA_TOP + self.radius, ARENA_BOTTOM - self.radius)

    def update_velocity(self, dt):
        if dt <= 0:
            return
        self.vx = (self.x - self.prev_x) / dt
        self.vy = (self.y - self.prev_y) / dt
        self.prev_x, self.prev_y = self.x, self.y

    def apply_bigger(self, duration, now):
        self.bigger_until = now + duration

    def apply_freeze(self, duration, now):
        self.frozen_until = now + duration

    def reset_effects(self):
        self.frozen_until = 0.0
        self.bigger_until = 0.0
        self.radius = self.base_radius

    def reset_position(self):
        self.x = self.lane_min_x if self.side == "left" else self.lane_max_x
        self.y = (ARENA_TOP + ARENA_BOTTOM) / 2
        self.prev_x, self.prev_y = self.x, self.y
        self.vx, self.vy = 0.0, 0.0

    def is_frozen(self, now):
        return now < self.frozen_until

    def tick_size(self, now):
        self.radius = int(self.base_radius * 1.4) if now < self.bigger_until else self.base_radius

    def draw(self, canvas, theme, now=0.0):
        center = (int(self.x), int(self.y))
        frozen = self.is_frozen(now)
        color = (150, 200, 255) if frozen else self.color

        if theme.neon:
            draw_glow_circle(canvas, center, self.radius, color, intensity=0.45, layers=3)
        else:
            cv2.circle(canvas, center, self.radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, self.radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, center, max(4, self.radius // 3), (255, 255, 255), 2, cv2.LINE_AA)

        if frozen:
            cv2.circle(canvas, center, self.radius + 6, (255, 255, 255), 1, cv2.LINE_AA)
