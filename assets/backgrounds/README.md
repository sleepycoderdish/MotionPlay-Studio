# backgrounds/

MotionPlay Studio renders every arena (Classic, Ice Arena, Neon Arena)
procedurally with OpenCV gradients/shapes, so no image assets are
required to run the game.

Drop a `.png`/`.jpg` in here and wire it up in `engine/maps.py`
(`MapTheme.render_background`) if you want to swap in custom hand-drawn
or photographic backgrounds for a theme.
