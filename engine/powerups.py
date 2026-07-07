"""Power-up spawning, pickup and timed-effect bookkeeping.

Exactly one power-up token exists on the field at a time. It spawns
every 20-30s, despawns if untouched for 10s, and applies its effect to
whichever side last touched the puck when it is collected.
"""

import math
import random
from enum import Enum, auto

from engine.utils import ARENA_LEFT, ARENA_RIGHT, ARENA_TOP, ARENA_BOTTOM, PLAYER, AI

SPAWN_MIN_DELAY = 20.0
SPAWN_MAX_DELAY = 30.0
DESPAWN_AFTER = 10.0
TOKEN_RADIUS = 20

SPEED_BOOST_DURATION = 10.0
FREEZE_DURATION = 3.0
BIGGER_PADDLE_DURATION = 10.0
SPEED_BOOST_MULTIPLIER = 1.4


class PowerUpType(Enum):
    SPEED_BOOST = auto()
    FREEZE_OPPONENT = auto()
    DOUBLE_SCORE = auto()
    BIGGER_PADDLE = auto()


POWERUP_STYLE = {
    PowerUpType.SPEED_BOOST: {"label": "Speed Boost", "color": (60, 200, 255), "glyph": ">>"},
    PowerUpType.FREEZE_OPPONENT: {"label": "Freeze Opponent", "color": (255, 220, 90), "glyph": "*"},
    PowerUpType.DOUBLE_SCORE: {"label": "Double Score", "color": (120, 90, 255), "glyph": "x2"},
    PowerUpType.BIGGER_PADDLE: {"label": "Bigger Paddle", "color": (90, 255, 140), "glyph": "+"},
}


class PowerUpToken:
    def __init__(self, ptype, x, y):
        self.type = ptype
        self.x = x
        self.y = y
        self.radius = TOKEN_RADIUS
        self.spawned_at = 0.0


class ActiveEffect:
    def __init__(self, label, color, side, until):
        self.label = label
        self.color = color
        self.side = side
        self.until = until


class PowerUpManager:
    def __init__(self):
        self.active_token = None
        self.next_spawn_at = None
        self.effects = []  # list[ActiveEffect]

    def start_match(self, now):
        self.active_token = None
        self.next_spawn_at = now + random.uniform(SPAWN_MIN_DELAY, SPAWN_MAX_DELAY)
        self.effects.clear()

    def update(self, now, ball):
        # spawn / despawn bookkeeping
        if self.active_token is None:
            if self.next_spawn_at is not None and now >= self.next_spawn_at:
                self._spawn(now)
        else:
            if now - self.active_token.spawned_at >= DESPAWN_AFTER:
                self.active_token = None
                self.next_spawn_at = now + random.uniform(SPAWN_MIN_DELAY, SPAWN_MAX_DELAY)

        # revert / maintain ball speed boost
        boost_active = any(e.label == POWERUP_STYLE[PowerUpType.SPEED_BOOST]["label"] and e.until > now
                            for e in self.effects)
        ball.speed_multiplier = SPEED_BOOST_MULTIPLIER if boost_active else 1.0

        # drop expired effects
        self.effects = [e for e in self.effects if e.until > now]

    def _spawn(self, now):
        margin = TOKEN_RADIUS + 60
        x = random.uniform(ARENA_LEFT + margin, ARENA_RIGHT - margin)
        y = random.uniform(ARENA_TOP + margin, ARENA_BOTTOM - margin)
        ptype = random.choice(list(PowerUpType))
        token = PowerUpToken(ptype, x, y)
        token.spawned_at = now
        self.active_token = token

    def try_collect(self, ball_x, ball_y, ball_radius):
        if self.active_token is None:
            return None
        dx = ball_x - self.active_token.x
        dy = ball_y - self.active_token.y
        if math.hypot(dx, dy) <= ball_radius + self.active_token.radius:
            token = self.active_token
            self.active_token = None
            return token
        return None

    def apply_effect(self, token, benefitting_side, now, ball, left_paddle, right_paddle, score_mgr):
        style = POWERUP_STYLE[token.type]
        opponent_side = AI if benefitting_side == PLAYER else PLAYER
        opponent_paddle = right_paddle if opponent_side == AI else left_paddle
        own_paddle = left_paddle if benefitting_side == PLAYER else right_paddle

        if token.type == PowerUpType.SPEED_BOOST:
            self.effects.append(ActiveEffect(style["label"], style["color"], benefitting_side, now + SPEED_BOOST_DURATION))

        elif token.type == PowerUpType.FREEZE_OPPONENT:
            opponent_paddle.apply_freeze(FREEZE_DURATION, now)
            self.effects.append(ActiveEffect(style["label"], style["color"], opponent_side, now + FREEZE_DURATION))

        elif token.type == PowerUpType.BIGGER_PADDLE:
            own_paddle.apply_bigger(BIGGER_PADDLE_DURATION, now)
            self.effects.append(ActiveEffect(style["label"], style["color"], benefitting_side, now + BIGGER_PADDLE_DURATION))

        elif token.type == PowerUpType.DOUBLE_SCORE:
            score_mgr.arm_double_score(benefitting_side)
            self.effects.append(ActiveEffect(style["label"], style["color"], benefitting_side, now + 40.0))

        self.next_spawn_at = now + random.uniform(SPAWN_MIN_DELAY, SPAWN_MAX_DELAY)

    def get_active_effects(self, now):
        return [e for e in self.effects if e.until > now]
