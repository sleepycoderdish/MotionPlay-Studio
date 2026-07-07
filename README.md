# MotionPlay Studio

A polished, gesture-controlled arcade air hockey game. Move your hand in front of
your webcam to steer your paddle -- no mouse, no keyboard, no color-marker
gloves required. Built with Python, OpenCV and MediaPipe Hands.


## Features

- **Gesture control (MediaPipe Hands)** -- single-hand tracking with palm-center
  smoothing removes jitter while staying responsive; a small debug circle shows
  exactly where the game sees your hand.
- **AI opponent, 3 difficulties** -- Easy/Medium/Hard change reaction time, top
  speed and aim accuracy; Medium and Hard predict the puck's bounce trajectory.
- **Power-ups** -- Speed Boost, Freeze Opponent, Double Score and Bigger Paddle
  spawn every 20-30s, despawn after 10s if untouched, and show as a HUD chip
  with a countdown while active.
- **Three maps** -- Classic, Ice Arena (frozen-lake shine) and Neon Arena
  (glowing borders + a pair of bounce bumpers), purely cosmetic -- mechanics are
  identical across all three.
- **Full game-state flow** -- Main Menu, Difficulty/Map selection, animated
  "3-2-1-GO!" countdown, Playing, Pause, Game Over, all mouse *and*
  keyboard-driven.
- **Local high scores** -- best score, fastest win and longest scoring streak
  are persisted to `data/highscores.json` and shown on the main menu.
- **Procedural sound** -- every effect (paddle hit, wall bounce, goal, power-up,
  countdown, win/lose) is synthesized on first launch, no audio files to fetch.
- **Graceful degradation** -- no webcam, or the hand-tracking model failed to
  download? The game keeps running and lets you steer with the mouse instead
  of crashing.

## Requirements

- Python 3.11+ (tested on 3.13)
- A webcam (optional -- see "no camera" fallback above)
- Internet access on first run only, to download the ~8MB MediaPipe hand
  landmarker model bundle (cached afterwards in `assets/models/`)

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Controls

| Input | Action |
|---|---|
| Move your hand in front of the webcam | Steer your (left, blue) paddle -- both up/down and forward/back |
| Mouse | Click menu buttons; also drives your paddle if no camera/model is available |
| `P` | Pause / resume |
| `R` | Restart the current match |
| `Esc` | Pause during a match, or back out of a submenu |
| `Q` | Quit immediately |

## How a match works

Pick a difficulty and map from the main menu, hit **Start Game**, and steer
your paddle with your hand after the "3-2-1-GO!" countdown. First to
**7 goals** wins. Power-up tokens occasionally appear on the table -- knock the
puck into one to trigger its effect for whichever side last touched the puck.

## Project structure

```
MotionPlayStudio/
├── main.py                 # thin entry point
├── engine/
│   ├── physics.py           # Ball + Paddle (circular mallets)
│   ├── collision.py         # wall/paddle/bumper/goal/power-up collision
│   ├── ai.py                 # AI opponent difficulty presets + prediction
│   ├── powerups.py           # power-up spawn/despawn/effect system
│   ├── maps.py                # Classic / Ice Arena / Neon Arena themes
│   ├── score.py               # match scoring + JSON high-score persistence
│   ├── gestures.py             # MediaPipe HandLandmarker wrapper
│   ├── game_state.py            # state machine, input, rendering, main loop
│   └── utils.py                  # constants, math/drawing helpers, sound synth
├── assets/
│   ├── sounds/                   # auto-generated WAV effects
│   ├── models/                    # auto-downloaded hand-tracking model (gitignored)
│   ├── backgrounds/                # optional custom art slot (see its README)
│   └── icons/                       # optional custom art slot (see its README)
├── data/
│   └── highscores.json               # persisted local high scores
└── requirements.txt
```

## Notes on implementation

- **Physics is delta-time based**, so gameplay speed stays consistent
  regardless of how fast the camera/CV pipeline is running that moment.
- **Paddles and the puck are all discs**, so every collision (paddle, bumper,
  wall) resolves with the same cheap circle-reflection math, including a small
  velocity transfer from a fast-moving paddle for a satisfying hit.
- **Sound playback** uses the stdlib `winsound` module (Windows) and silently
  no-ops anywhere it isn't available, so a missing audio backend never crashes
  the game.

## Troubleshooting

- **"Hand-tracking model unavailable" banner**: the game couldn't download the
  MediaPipe model on first launch (no internet). It will retry next launch;
  meanwhile the mouse controls your paddle.
- **"No camera" banner**: no webcam was detected; the mouse controls your
  paddle instead.
- **Low FPS**: close other webcam-using apps, or lower your webcam's
  resolution in your OS camera settings -- the FPS counter in the HUD tells you
  exactly what you're getting.
