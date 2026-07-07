"""MotionPlay Studio -- gesture-controlled arcade air hockey.

Run with:  python main.py
Controls:  move your hand in front of the webcam to steer your paddle,
           P to pause/resume, R to restart, Esc to back out of a screen.
"""

from engine.game_state import Game


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
