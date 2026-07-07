"""AI opponent controlling the right-hand paddle.

Three difficulty presets change three things: how quickly the AI
reacts to a new ball state (reaction_delay), how fast it can move
(max_speed), and how accurate its aim is (error_margin + whether it
bothers predicting bounces at all). That's enough for each level to
feel distinctly different while keeping the logic simple.
"""

import random
from dataclasses import dataclass

from engine.utils import ARENA_TOP, ARENA_BOTTOM, clamp


@dataclass
class AIConfig:
    key: str
    label: str
    reaction_delay: float   # seconds between target re-evaluations
    max_speed: float        # px/second
    error_margin: float     # px of random aim noise
    predict_bounces: bool   # simulate wall reflections ahead of time
    idle_recenter: float    # how strongly it drifts back to center when idle


DIFFICULTIES = {
    "easy": AIConfig("easy", "Easy", reaction_delay=0.45, max_speed=430.0,
                      error_margin=60.0, predict_bounces=False, idle_recenter=0.15),
    "medium": AIConfig("medium", "Medium", reaction_delay=0.18, max_speed=780.0,
                        error_margin=24.0, predict_bounces=True, idle_recenter=0.35),
    "hard": AIConfig("hard", "Hard", reaction_delay=0.05, max_speed=1150.0,
                      error_margin=6.0, predict_bounces=True, idle_recenter=0.6),
}

DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def _predict_intercept_y(ball, target_x, margin):
    """Predict where the ball's y will be when it reaches target_x,
    accounting for wall bounces via a reflection (modulo) trick."""
    if ball.vx == 0:
        return ball.y

    t = (target_x - ball.x) / ball.vx
    if t < 0:
        return ball.y  # ball is moving away from this x-plane

    raw_y = ball.y + ball.vy * t

    top = ARENA_TOP + margin
    bottom = ARENA_BOTTOM - margin
    span = max(1.0, bottom - top)

    y_rel = (raw_y - top) % (2 * span)
    if y_rel > span:
        y_rel = 2 * span - y_rel
    return top + y_rel


class AIController:
    def __init__(self, config: AIConfig):
        self.config = config
        self._timer = 0.0
        self.target_x = None
        self.target_y = None

    def set_difficulty(self, config: AIConfig):
        self.config = config
        self._timer = 0.0

    def update(self, paddle, ball, dt, now):
        cfg = self.config

        if paddle.is_frozen(now):
            return  # frozen paddles simply cannot act

        self._timer += dt
        if self.target_x is None or self._timer >= cfg.reaction_delay:
            self._timer = 0.0
            self._recompute_target(paddle, ball, cfg)

        paddle.move_toward(self.target_x, self.target_y, dt, max_speed=cfg.max_speed)

    def _recompute_target(self, paddle, ball, cfg):
        approaching = ball.vx > 0  # ball moving from left toward the AI on the right

        rest_x = paddle.lane_max_x
        lunge_zone_x = paddle.lane_min_x

        if approaching and ball.x >= lunge_zone_x - 260:
            # ball is close enough to be worth lunging forward for
            self.target_x = clamp(ball.x + 10, paddle.lane_min_x, paddle.lane_max_x)
        else:
            self.target_x = rest_x

        if approaching:
            if cfg.predict_bounces:
                predicted = _predict_intercept_y(ball, self.target_x, paddle.radius)
            else:
                predicted = ball.y
            noise = random.uniform(-cfg.error_margin, cfg.error_margin)
            self.target_y = clamp(predicted + noise, ARENA_TOP + paddle.radius, ARENA_BOTTOM - paddle.radius)
        else:
            # idle: drift back toward vertical center, softly, based on difficulty
            center_y = (ARENA_TOP + ARENA_BOTTOM) / 2
            self.target_y = paddle.y + (center_y - paddle.y) * cfg.idle_recenter


def get_config(key):
    return DIFFICULTIES.get(key, DIFFICULTIES["medium"])
