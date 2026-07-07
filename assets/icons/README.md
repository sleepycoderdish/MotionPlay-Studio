# icons/

Power-up glyphs, the FPS indicator, and every UI icon are drawn with
OpenCV primitives/text (see `engine/utils.py` and `engine/powerups.py`)
so the game needs zero external icon files to run.

Drop `.png` icons (ideally with alpha) in here and blend them onto the
canvas in `engine/game_state.py` if you want custom artwork instead of
the procedural glyphs.
