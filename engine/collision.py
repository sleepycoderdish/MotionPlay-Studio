"""Collision detection & resolution for the puck.

All resolution is circle-vs-circle (mallets, bumpers, puck are all
discs) or circle-vs-wall, which keeps the math cheap enough to run
every frame without a broad/narrow phase split.
"""

import math

from engine.utils import ARENA_LEFT, ARENA_RIGHT, ARENA_TOP, ARENA_BOTTOM, PLAYER, AI


def _reflect_off_circle(ball, cx, cy, obstacle_radius, restitution=1.0, extra_speed=0.0,
                         paddle_vx=0.0, paddle_vy=0.0, velocity_transfer=0.0):
    dx = ball.x - cx
    dy = ball.y - cy
    dist = math.hypot(dx, dy) or 1e-6
    min_dist = ball.radius + obstacle_radius

    if dist >= min_dist:
        return False

    nx, ny = dx / dist, dy / dist

    # push the ball out so it doesn't stay stuck inside the obstacle
    overlap = min_dist - dist
    ball.x += nx * overlap
    ball.y += ny * overlap

    # reflect velocity about the collision normal
    dot = ball.vx * nx + ball.vy * ny
    ball.vx = (ball.vx - 2 * dot * nx) * restitution
    ball.vy = (ball.vy - 2 * dot * ny) * restitution

    # transfer a fraction of the paddle's own motion into the puck for feel
    ball.vx += paddle_vx * velocity_transfer
    ball.vy += paddle_vy * velocity_transfer

    if extra_speed:
        speed = math.hypot(ball.vx, ball.vy) or 1e-6
        scale = (speed + extra_speed) / speed
        ball.vx *= scale
        ball.vy *= scale

    return True


def handle_wall_collision(ball, sound=None):
    bounced = False
    if ball.y - ball.radius <= ARENA_TOP:
        ball.y = ARENA_TOP + ball.radius
        ball.vy = abs(ball.vy)
        bounced = True
    elif ball.y + ball.radius >= ARENA_BOTTOM:
        ball.y = ARENA_BOTTOM - ball.radius
        ball.vy = -abs(ball.vy)
        bounced = True

    if bounced and sound:
        sound.play("wall_hit")
    return bounced


def handle_paddle_collision(ball, paddle, side_name, sound=None):
    hit = _reflect_off_circle(
        ball, paddle.x, paddle.y, paddle.radius,
        restitution=1.0, extra_speed=18.0,
        paddle_vx=paddle.vx, paddle_vy=paddle.vy, velocity_transfer=0.18,
    )
    if hit:
        ball.bump_speed()
        ball.last_hit_by = side_name
        if sound:
            sound.play("paddle_hit")
    return hit


def handle_bumper_collisions(ball, bumpers, sound=None):
    for bumper in bumpers:
        hit = _reflect_off_circle(ball, bumper.x, bumper.y, bumper.radius, restitution=1.02, extra_speed=6.0)
        if hit and sound:
            sound.play("wall_hit")


def check_goal(ball):
    """Returns 'player' if AI scored (ball left through the left goal,
    i.e. the player failed to defend), 'ai' if the player scored, else None."""
    if ball.x - ball.radius <= ARENA_LEFT:
        return AI  # ball exited player's (left) side -> AI scores
    if ball.x + ball.radius >= ARENA_RIGHT:
        return PLAYER  # ball exited AI's (right) side -> player scores
    return None


def handle_powerup_pickup(ball, powerup_manager, sound=None):
    collected = powerup_manager.try_collect(ball.x, ball.y, ball.radius)
    if collected and sound:
        sound.play("powerup")
    return collected
