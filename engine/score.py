"""Match scoring/timing plus persistent local high-score tracking."""

import json
import os
import time

from engine.utils import MAX_SCORE, PLAYER, AI, HIGHSCORE_FILE, DATA_DIR


class ScoreManager:
    def __init__(self):
        self.player_score = 0
        self.ai_score = 0
        self.player_streak = 0
        self.ai_streak = 0
        self.player_best_streak_this_match = 0
        self.double_score_armed = {PLAYER: False, AI: False}
        self.match_start_time = None
        self.match_end_time = None
        self.last_goal_side = None
        self.last_goal_points = 0

    def start(self):
        self.player_score = 0
        self.ai_score = 0
        self.player_streak = 0
        self.ai_streak = 0
        self.player_best_streak_this_match = 0
        self.double_score_armed = {PLAYER: False, AI: False}
        self.match_start_time = time.time()
        self.match_end_time = None
        self.last_goal_side = None
        self.last_goal_points = 0

    def arm_double_score(self, side):
        self.double_score_armed[side] = True

    def record_goal(self, side):
        points = 2 if self.double_score_armed.get(side) else 1

        if side == PLAYER:
            self.player_score += points
            self.player_streak += 1
            self.ai_streak = 0
            self.player_best_streak_this_match = max(self.player_best_streak_this_match, self.player_streak)
        else:
            self.ai_score += points
            self.ai_streak += 1
            self.player_streak = 0

        self.double_score_armed = {PLAYER: False, AI: False}
        self.last_goal_side = side
        self.last_goal_points = points
        return points

    def check_winner(self):
        if self.player_score >= MAX_SCORE:
            return PLAYER
        if self.ai_score >= MAX_SCORE:
            return AI
        return None

    def finish(self):
        self.match_end_time = time.time()

    def elapsed_time(self):
        if self.match_start_time is None:
            return 0.0
        end = self.match_end_time if self.match_end_time else time.time()
        return end - self.match_start_time

    def elapsed_str(self):
        total = int(self.elapsed_time())
        return f"{total // 60:02d}:{total % 60:02d}"


class HighScoreStore:
    DEFAULTS = {"best_score": 0, "fastest_win": None, "highest_streak": 0}

    def __init__(self, path=HIGHSCORE_FILE):
        self.path = path
        self.data = dict(self.DEFAULTS)
        self.load()

    def load(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    loaded = json.load(f)
                self.data = {**self.DEFAULTS, **loaded}
            except (json.JSONDecodeError, OSError):
                self.data = dict(self.DEFAULTS)
                self.save()
        else:
            self.save()

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except OSError:
            pass

    def update_from_match(self, player_score, time_taken, best_streak, player_won):
        """Update records, returning which categories were newly broken."""
        broken = {"best_score": False, "fastest_win": False, "highest_streak": False}

        if player_score > self.data.get("best_score", 0):
            self.data["best_score"] = player_score
            broken["best_score"] = True

        if player_won:
            current_fastest = self.data.get("fastest_win")
            if current_fastest is None or time_taken < current_fastest:
                self.data["fastest_win"] = round(time_taken, 2)
                broken["fastest_win"] = True

        if best_streak > self.data.get("highest_streak", 0):
            self.data["highest_streak"] = best_streak
            broken["highest_streak"] = True

        if any(broken.values()):
            self.save()

        return broken

    @staticmethod
    def format_time(seconds):
        if seconds is None:
            return "--:--"
        total = int(seconds)
        return f"{total // 60:02d}:{total % 60:02d}"
