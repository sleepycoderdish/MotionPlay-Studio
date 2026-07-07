"""Top level game-state machine and orchestration.

The `Game` class owns every subsystem (camera, gesture tracking, sound,
physics, AI, power-ups, score/high-scores) and drives a small explicit
state machine:

    MAIN_MENU <-> DIFFICULTY_MENU
              <-> MAP_MENU
              <-> HIGH_SCORES
    MAIN_MENU -> COUNTDOWN -> PLAYING <-> PAUSED
                              PLAYING -> GAME_OVER -> (restart) COUNTDOWN
                                                    -> (menu) MAIN_MENU
"""

import os
import time

import cv2
import numpy as np

from engine import ai, maps, powerups
from engine.physics import Ball, Paddle
from engine.score import ScoreManager, HighScoreStore
from engine.gestures import GestureController, CameraWorker
from engine.collision import (
    handle_wall_collision, handle_paddle_collision, handle_bumper_collisions,
    check_goal, handle_powerup_pickup,
)
from engine.utils import (
    WIDTH, HEIGHT, WINDOW_NAME, HUD_TOP, ARENA_LEFT, ARENA_RIGHT, ARENA_TOP, ARENA_BOTTOM,
    PADDLE_RADIUS, PADDLE_LANE_DEPTH, PLAYER_PADDLE_SPEED, MAX_SCORE, PLAYER, AI as AI_SIDE,
    clamp, lerp, ease_out_back, RollingAverage, Button, SoundManager,
    draw_text, draw_rounded_rect, blend_overlay, draw_glow_circle,
)

# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #

MAIN_MENU = "main_menu"
DIFFICULTY_MENU = "difficulty_menu"
MAP_MENU = "map_menu"
HIGH_SCORES = "high_scores"
COUNTDOWN = "countdown"
PLAYING = "playing"
PAUSED = "paused"
GAME_OVER = "game_over"

COUNTDOWN_STEP = 0.8
COUNTDOWN_SEQUENCE = ["3", "2", "1", "GO!"]
TRANSITION_DURATION = 0.22

ACCENT = (0, 210, 255)


class Game:
    def __init__(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        self.canvas = None
        self.running = True

        self._show_loading_splash("Starting MotionPlay Studio...")

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        self.camera_ok = self.camera.isOpened()
        if self.camera_ok:
            # a modest, fixed capture size keeps both OpenCV and MediaPipe
            # inference fast regardless of the webcam's native resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._show_loading_splash("Loading hand-tracking model...")
        self.gesture = GestureController()

        # hand detection runs on its own thread so a slow inference frame
        # never stalls menu clicks or gameplay rendering
        self.camera_worker = CameraWorker(self.camera, self.gesture)
        self.camera_worker.start()

        self._show_loading_splash("Preparing sound effects...")
        self.sound = SoundManager()
        self.highscores = HighScoreStore()

        self.selected_difficulty = "medium"
        self.selected_map = "classic"

        self.mouse_pos = (0, 0)
        self.mouse_click_pos = None

        self.fps_avg = RollingAverage(30)
        self.last_time = time.time()

        self.hand_frame = None  # last camera frame (BGR, mirrored) for PiP + tracking
        self.hand_point_norm = None

        self.countdown_start = 0.0
        self.countdown_last_index = -1

        self.last_broken_records = {}
        self.last_winner = None

        self._init_entities()
        self.buttons = []
        self._enter_state(MAIN_MENU)

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    def _show_loading_splash(self, message):
        """Paint an immediate frame so the OS window shows up right away
        instead of looking frozen while the heavier subsystems (camera,
        MediaPipe model, sound synthesis) initialize."""
        splash = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")
        splash[:] = (22, 23, 28)
        draw_text(splash, "MotionPlay Studio", (WIDTH // 2, HEIGHT // 2 - 30), scale=1.4,
                  thickness=3, color=(255, 255, 255), align="center")
        draw_text(splash, message, (WIDTH // 2, HEIGHT // 2 + 20), scale=0.6, thickness=1,
                  color=(0, 210, 255), align="center", shadow=False)
        cv2.imshow(WINDOW_NAME, splash)
        cv2.waitKey(1)

    def _init_entities(self):
        self.theme = maps.get_map(self.selected_map)
        self.ball = Ball((ARENA_LEFT + ARENA_RIGHT) // 2, (ARENA_TOP + ARENA_BOTTOM) // 2)

        left_lane = (ARENA_LEFT + PADDLE_RADIUS, ARENA_LEFT + PADDLE_RADIUS + PADDLE_LANE_DEPTH)
        right_lane = (ARENA_RIGHT - PADDLE_RADIUS - PADDLE_LANE_DEPTH, ARENA_RIGHT - PADDLE_RADIUS)

        self.left_paddle = Paddle("left", (ARENA_TOP + ARENA_BOTTOM) // 2, *left_lane, self.theme.paddle_left_color)
        self.right_paddle = Paddle("right", (ARENA_TOP + ARENA_BOTTOM) // 2, *right_lane, self.theme.paddle_right_color)

        self.ai_controller = ai.AIController(ai.get_config(self.selected_difficulty))
        self.score = ScoreManager()
        self.powerup_mgr = powerups.PowerUpManager()

    def _apply_theme(self):
        self.theme = maps.get_map(self.selected_map)
        self.left_paddle.color = self.theme.paddle_left_color
        self.right_paddle.color = self.theme.paddle_right_color

    # ------------------------------------------------------------------ #
    # Input plumbing
    # ------------------------------------------------------------------ #

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click_pos = (x, y)

    def _update_buttons(self, dt):
        click = self.mouse_click_pos
        for btn in self.buttons:
            hovered = btn.contains(*self.mouse_pos) and btn.enabled
            if hovered and not btn.was_hovering:
                self.sound.play("hover")
            btn.was_hovering = hovered
            btn.update_hover(hovered, dt)
            if click and hovered and btn.action:
                self.sound.play("click")
                btn.action()
                break  # the action may have rebuilt self.buttons -- stop iterating the old list
        self.mouse_click_pos = None

    def _draw_buttons(self):
        for btn in self.buttons:
            btn.draw(self.canvas, accent_color=self.theme.accent if self.theme else ACCENT)

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #

    def _enter_state(self, state):
        self.state = state
        self.transition_start = time.time()
        builders = {
            MAIN_MENU: self._build_main_menu_buttons,
            DIFFICULTY_MENU: self._build_difficulty_buttons,
            MAP_MENU: self._build_map_buttons,
            HIGH_SCORES: self._build_highscore_buttons,
            PAUSED: self._build_pause_buttons,
            GAME_OVER: self._build_gameover_buttons,
        }
        self.buttons = builders.get(state, lambda: [])() or []

        if state == COUNTDOWN:
            self.countdown_start = time.time()
            self.countdown_last_index = -1

    def start_new_match(self):
        self._apply_theme()
        self.ball.reset()
        self.left_paddle.reset_effects()
        self.right_paddle.reset_effects()
        self.left_paddle.reset_position()
        self.right_paddle.reset_position()
        self.ai_controller.set_difficulty(ai.get_config(self.selected_difficulty))
        self.gesture.reset()
        self._enter_state(COUNTDOWN)

    def _begin_playing(self):
        self.score.start()
        self.powerup_mgr.start_match(time.time())
        self._enter_state(PLAYING)

    def go_to_main_menu(self):
        self._enter_state(MAIN_MENU)

    def quit_game(self):
        self.running = False

    def set_difficulty(self, key):
        self.selected_difficulty = key
        self._enter_state(MAIN_MENU)

    def set_map(self, key):
        self.selected_map = key
        self._apply_theme()
        self._enter_state(MAIN_MENU)

    def toggle_pause(self):
        if self.state == PLAYING:
            self._enter_state(PAUSED)
        elif self.state == PAUSED:
            self._enter_state(PLAYING)

    # ------------------------------------------------------------------ #
    # Menu button builders
    # ------------------------------------------------------------------ #

    def _centered_buttons(self, labels_actions, start_y, w=340, h=56, gap=18, sub_labels=None):
        x = WIDTH // 2 - w // 2
        buttons = []
        for i, (label, action) in enumerate(labels_actions):
            sub = sub_labels[i] if sub_labels else None
            buttons.append(Button((x, start_y + i * (h + gap), w, h), label, action, sub_label=sub))
        return buttons

    def _build_main_menu_buttons(self):
        diff_label = ai.get_config(self.selected_difficulty).label
        map_label = maps.get_map(self.selected_map).name
        return self._centered_buttons([
            ("Start Game", self.start_new_match),
            (f"Difficulty: {diff_label}", lambda: self._enter_state(DIFFICULTY_MENU)),
            (f"Map: {map_label}", lambda: self._enter_state(MAP_MENU)),
            ("High Scores", lambda: self._enter_state(HIGH_SCORES)),
            ("Exit", self.quit_game),
        ], start_y=250)

    def _build_difficulty_buttons(self):
        entries = []
        for key in ai.DIFFICULTY_ORDER:
            cfg = ai.get_config(key)
            label = cfg.label + ("  [selected]" if key == self.selected_difficulty else "")
            entries.append((label, (lambda k=key: self.set_difficulty(k))))
        entries.append(("Back", self.go_to_main_menu))
        return self._centered_buttons(entries, start_y=230)

    def _build_map_buttons(self):
        entries = []
        for key in maps.MAP_ORDER:
            theme = maps.get_map(key)
            label = theme.name + ("  [selected]" if key == self.selected_map else "")
            entries.append((label, (lambda k=key: self.set_map(k))))
        entries.append(("Back", self.go_to_main_menu))
        return self._centered_buttons(entries, start_y=230)

    def _build_highscore_buttons(self):
        return self._centered_buttons([("Back", self.go_to_main_menu)], start_y=560)

    def _build_pause_buttons(self):
        return self._centered_buttons([
            ("Resume", self.toggle_pause),
            ("Restart Match", self.start_new_match),
            ("Main Menu", self.go_to_main_menu),
            ("Exit", self.quit_game),
        ], start_y=260)

    def _build_gameover_buttons(self):
        return self._centered_buttons([
            ("Play Again", self.start_new_match),
            ("Main Menu", self.go_to_main_menu),
            ("Exit", self.quit_game),
        ], start_y=430)

    # ------------------------------------------------------------------ #
    # Camera / gesture plumbing
    # ------------------------------------------------------------------ #

    def _apply_player_control(self, dt):
        if self.left_paddle.is_frozen(time.time()):
            return

        if self.gesture.detected and self.gesture.last_norm_point:
            nx, ny = self.gesture.last_norm_point
        elif not self.camera_ok or not self.gesture.available:
            # graceful fallback when no webcam / hand-tracking model is available: use the mouse
            nx = clamp(self.mouse_pos[0] / WIDTH, 0.0, 1.0)
            ny = clamp(self.mouse_pos[1] / HEIGHT, 0.0, 1.0)
        else:
            return  # camera available but no hand currently visible -> hold position

        target_x = lerp(self.left_paddle.lane_min_x, self.left_paddle.lane_max_x, nx)
        target_y = lerp(ARENA_TOP + self.left_paddle.radius, ARENA_BOTTOM - self.left_paddle.radius, ny)
        # glide toward the target instead of teleporting there -- this masks the
        # ~15-20Hz update rate of hand tracking behind smooth, continuous motion
        # at the render loop's own (much higher) frame rate.
        self.left_paddle.move_toward(target_x, target_y, dt, max_speed=PLAYER_PADDLE_SPEED)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, dt):
        self._update_buttons(dt)

        if self.state == COUNTDOWN:
            self._update_countdown(dt)
        elif self.state == PLAYING:
            self._update_playing(dt)

    def _update_countdown(self, dt):
        self._apply_player_control(dt)
        elapsed = time.time() - self.countdown_start
        index = min(int(elapsed // COUNTDOWN_STEP), len(COUNTDOWN_SEQUENCE) - 1)
        if index != self.countdown_last_index:
            self.countdown_last_index = index
            self.sound.play("countdown_go" if index == len(COUNTDOWN_SEQUENCE) - 1 else "countdown_tick")
        if elapsed >= COUNTDOWN_STEP * len(COUNTDOWN_SEQUENCE):
            self._begin_playing()

    def _update_playing(self, dt):
        now = time.time()

        self._apply_player_control(dt)
        self.ai_controller.update(self.right_paddle, self.ball, dt, now)

        self.left_paddle.update_velocity(dt)
        self.right_paddle.update_velocity(dt)
        self.left_paddle.tick_size(now)
        self.right_paddle.tick_size(now)

        self.ball.update(dt)

        handle_wall_collision(self.ball, self.sound)
        handle_paddle_collision(self.ball, self.left_paddle, PLAYER, self.sound)
        handle_paddle_collision(self.ball, self.right_paddle, AI_SIDE, self.sound)
        handle_bumper_collisions(self.ball, self.theme.bumpers, self.sound)

        self.powerup_mgr.update(now, self.ball)
        collected = handle_powerup_pickup(self.ball, self.powerup_mgr, self.sound)
        if collected:
            side = self.ball.last_hit_by or PLAYER
            self.powerup_mgr.apply_effect(collected, side, now, self.ball,
                                           self.left_paddle, self.right_paddle, self.score)

        goal_side = check_goal(self.ball)
        if goal_side:
            self.sound.play("goal")
            self.score.record_goal(goal_side)
            self.ball.reset()
            winner = self.score.check_winner()
            if winner:
                self._finish_match(winner)

    def _finish_match(self, winner):
        self.score.finish()
        player_won = winner == PLAYER
        self.last_winner = winner
        self.last_broken_records = self.highscores.update_from_match(
            self.score.player_score, self.score.elapsed_time(),
            self.score.player_best_streak_this_match, player_won,
        )
        self.sound.play("win" if player_won else "lose")
        self._enter_state(GAME_OVER)

    # ------------------------------------------------------------------ #
    # Key handling
    # ------------------------------------------------------------------ #

    def handle_key(self, key):
        if key == -1:
            return
        if key == 27:  # ESC
            if self.state in (PLAYING,):
                self.toggle_pause()
            elif self.state in (DIFFICULTY_MENU, MAP_MENU, HIGH_SCORES, PAUSED, GAME_OVER):
                self.go_to_main_menu()
        elif key in (ord("p"), ord("P")):
            if self.state in (PLAYING, PAUSED):
                self.toggle_pause()
        elif key in (ord("r"), ord("R")):
            if self.state in (PLAYING, PAUSED, GAME_OVER):
                self.start_new_match()
        elif key in (ord("q"), ord("Q")):
            self.running = False

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def render(self):
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")

        if self.state == MAIN_MENU:
            self._render_main_menu()
        elif self.state == DIFFICULTY_MENU:
            self._render_simple_menu("Select Difficulty")
        elif self.state == MAP_MENU:
            self._render_simple_menu("Select Map")
        elif self.state == HIGH_SCORES:
            self._render_high_scores()
        elif self.state in (COUNTDOWN, PLAYING, PAUSED, GAME_OVER):
            self._render_arena_scene()
            if self.state == COUNTDOWN:
                self._render_countdown_overlay()
            elif self.state == PAUSED:
                self._render_pause_overlay()
            elif self.state == GAME_OVER:
                self._render_gameover_overlay()

        self._render_transition_fade()
        return self.canvas

    def _render_transition_fade(self):
        """Brief fade-in on every state change so screen switches read as
        an intentional transition instead of an abrupt, jarring cut."""
        elapsed = time.time() - self.transition_start
        if elapsed >= TRANSITION_DURATION:
            return
        fade_t = 1.0 - (elapsed / TRANSITION_DURATION)
        blend_overlay(self.canvas, alpha=fade_t * 0.85)

    def _render_main_menu(self):
        self.theme.render_background(self.canvas) if self.theme else self.canvas.fill(18)
        draw_text(self.canvas, "MotionPlay Studio", (WIDTH // 2, 130), scale=1.7, thickness=3,
                  color=(255, 255, 255), align="center")
        draw_text(self.canvas, "Gesture-Controlled Arcade Air Hockey", (WIDTH // 2, 170), scale=0.7,
                  thickness=1, color=(0, 210, 255), align="center")
        self._draw_buttons()
        draw_text(self.canvas, "Move your hand in front of the webcam to control your paddle",
                  (WIDTH // 2, HEIGHT - 30), scale=0.5, thickness=1, color=(150, 150, 155), align="center")

    def _render_simple_menu(self, title):
        self.theme.render_background(self.canvas)
        draw_text(self.canvas, title, (WIDTH // 2, 140), scale=1.2, thickness=3,
                  color=(255, 255, 255), align="center")
        self._draw_buttons()

    def _render_high_scores(self):
        self.theme.render_background(self.canvas)
        draw_text(self.canvas, "High Scores", (WIDTH // 2, 120), scale=1.2, thickness=3,
                  color=(255, 255, 255), align="center")

        data = self.highscores.data
        cards = [
            ("BEST SCORE", str(data.get("best_score", 0))),
            ("FASTEST WIN", HighScoreStore.format_time(data.get("fastest_win"))),
            ("HIGHEST STREAK", str(data.get("highest_streak", 0))),
        ]
        card_w, card_h, gap = 260, 160, 30
        total_w = card_w * 3 + gap * 2
        x0 = WIDTH // 2 - total_w // 2
        y0 = 220
        for i, (label, value) in enumerate(cards):
            x = x0 + i * (card_w + gap)
            draw_rounded_rect(self.canvas, (x, y0), (x + card_w, y0 + card_h), (32, 34, 44), radius=16)
            draw_rounded_rect(self.canvas, (x, y0), (x + card_w, y0 + card_h), (0, 210, 255), radius=16, thickness=2)
            draw_text(self.canvas, value, (x + card_w // 2, y0 + 85), scale=1.3, thickness=3,
                      color=(255, 255, 255), align="center", shadow=False)
            draw_text(self.canvas, label, (x + card_w // 2, y0 + 125), scale=0.5, thickness=1,
                      color=(160, 200, 220), align="center", shadow=False)

        self._draw_buttons()

    def _render_arena_scene(self):
        now = time.time()
        self.theme.render_background(self.canvas)
        self.theme.render_arena(self.canvas)

        token = self.powerup_mgr.active_token
        if token:
            style = powerups.POWERUP_STYLE[token.type]
            pulse = 1.0 + 0.12 * abs(((now * 3) % 2.0) - 1.0)
            r = int(token.radius * pulse)
            center = (int(token.x), int(token.y))
            draw_glow_circle(self.canvas, center, r, style["color"], intensity=0.5, layers=3)
            cv2.circle(self.canvas, center, r, (255, 255, 255), 2, cv2.LINE_AA)
            draw_text(self.canvas, style["glyph"], (center[0], center[1] + 6), scale=0.5,
                      thickness=2, color=(20, 20, 20), align="center", shadow=False)

        self.left_paddle.draw(self.canvas, self.theme, now)
        self.right_paddle.draw(self.canvas, self.theme, now)
        self.ball.draw(self.canvas, self.theme)

        self._render_hud(now)
        self._render_camera_pip()

    def _render_hud(self, now):
        draw_rounded_rect(self.canvas, (0, 0), (WIDTH, HUD_TOP), (18, 19, 24), radius=0)
        cv2.line(self.canvas, (0, HUD_TOP), (WIDTH, HUD_TOP), (60, 62, 70), 2)

        draw_text(self.canvas, f"YOU  {self.score.player_score}", (24, 40), scale=0.85, thickness=2,
                  color=self.theme.paddle_left_color, align="left", shadow=False)
        draw_text(self.canvas, f"AI  {self.score.ai_score}", (WIDTH - 24, 40), scale=0.85, thickness=2,
                  color=self.theme.paddle_right_color, align="right", shadow=False)

        draw_text(self.canvas, self.score.elapsed_str(), (WIDTH // 2, 32), scale=0.7, thickness=2,
                  color=(255, 255, 255), align="center", shadow=False)
        draw_text(self.canvas, f"{ai.get_config(self.selected_difficulty).label.upper()} · {self.theme.name.upper()}",
                  (WIDTH // 2, 54), scale=0.45, thickness=1, color=(160, 165, 175), align="center", shadow=False)

        effects = self.powerup_mgr.get_active_effects(now)
        if effects:
            e = effects[0]
            remaining = max(0, int(e.until - now))
            owner = "YOU" if e.side == PLAYER else "AI"
            draw_text(self.canvas, f"{e.label.upper()} ({owner}) {remaining}s", (WIDTH // 2 - 260, 40),
                      scale=0.5, thickness=1, color=e.color, align="left", shadow=False)

        fps = self.fps_avg.average
        draw_text(self.canvas, f"FPS {fps:0.0f}", (WIDTH // 2 + 210, 40), scale=0.5, thickness=1,
                  color=(140, 220, 140) if fps >= 24 else (140, 140, 220), align="left", shadow=False)

        if not self.camera_ok:
            draw_text(self.canvas, "No camera -- mouse controls your paddle", (24, HUD_TOP + 22), scale=0.45,
                      thickness=1, color=(255, 180, 90), align="left")
        elif not self.gesture.available:
            draw_text(self.canvas, "Hand-tracking model unavailable -- mouse controls your paddle",
                      (24, HUD_TOP + 22), scale=0.45, thickness=1, color=(255, 180, 90), align="left")

    def _render_camera_pip(self):
        if self.hand_frame is None:
            return
        pip_w, pip_h = 200, 150
        small = cv2.resize(self.hand_frame, (pip_w, pip_h))
        x0 = WIDTH - pip_w - 20
        y0 = HEIGHT - pip_h - 20
        cv2.rectangle(self.canvas, (x0 - 3, y0 - 3), (x0 + pip_w + 3, y0 + pip_h + 3), (0, 210, 255), 2)
        self.canvas[y0:y0 + pip_h, x0:x0 + pip_w] = small

    def _render_countdown_overlay(self):
        blend_overlay(self.canvas, alpha=0.25)
        elapsed = time.time() - self.countdown_start
        step_t = (elapsed % COUNTDOWN_STEP) / COUNTDOWN_STEP
        scale = 2.6 + 1.4 * (1 - ease_out_back(step_t))
        index = min(int(elapsed // COUNTDOWN_STEP), len(COUNTDOWN_SEQUENCE) - 1)
        text = COUNTDOWN_SEQUENCE[index]
        draw_text(self.canvas, text, (WIDTH // 2, HEIGHT // 2 + 30), scale=scale, thickness=6,
                  color=(0, 210, 255) if text == "GO!" else (255, 255, 255), align="center")

    def _render_pause_overlay(self):
        blend_overlay(self.canvas, alpha=0.55)
        draw_text(self.canvas, "GAME PAUSED", (WIDTH // 2, 170), scale=1.3, thickness=3,
                  color=(255, 255, 255), align="center")
        draw_text(self.canvas, "Press P to Resume", (WIDTH // 2, 210), scale=0.6, thickness=1,
                  color=(180, 185, 195), align="center")
        self._draw_buttons()

    def _render_gameover_overlay(self):
        blend_overlay(self.canvas, alpha=0.6)
        player_won = self.last_winner == PLAYER
        title = "YOU WIN!" if player_won else "AI WINS"
        color = (90, 230, 130) if player_won else (90, 120, 230)
        draw_text(self.canvas, title, (WIDTH // 2, 150), scale=1.6, thickness=4, color=color, align="center")
        draw_text(self.canvas, f"Final Score  {self.score.player_score} - {self.score.ai_score}",
                  (WIDTH // 2, 195), scale=0.7, thickness=2, color=(255, 255, 255), align="center")
        draw_text(self.canvas, f"Time Taken  {self.score.elapsed_str()}", (WIDTH // 2, 225), scale=0.55,
                  thickness=1, color=(190, 195, 205), align="center")

        y = 260
        for key, label in (("best_score", "NEW BEST SCORE!"), ("fastest_win", "NEW FASTEST WIN!"),
                            ("highest_streak", "NEW LONGEST STREAK!")):
            if self.last_broken_records.get(key):
                draw_text(self.canvas, label, (WIDTH // 2, y), scale=0.55, thickness=2,
                          color=(255, 215, 60), align="center")
                y += 28

        self._draw_buttons()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        try:
            while self.running:
                now = time.time()
                dt = min(now - self.last_time, 0.05)
                self.last_time = now
                if dt > 0:
                    self.fps_avg.push(1.0 / dt)

                self.camera_worker.enabled = self.state in (COUNTDOWN, PLAYING)
                self.hand_frame = self.camera_worker.get_frame()

                self.update(dt)
                self.render()

                cv2.imshow(WINDOW_NAME, self.canvas)
                key = cv2.waitKey(1) & 0xFF
                self.handle_key(key)

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
        finally:
            self.shutdown()

    def shutdown(self):
        self.camera_worker.stop()
        self.camera_worker.join(timeout=1.0)
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        self.gesture.close()
        cv2.destroyAllWindows()
