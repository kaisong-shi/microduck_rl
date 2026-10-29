"""Soccer (11 v 11 Microduck) building blocks.

Layout (see ``docs/soccer_tasks.md``):
  pitch.py  — dimensions, field/goal/wall geometry, ball rolling resistance
  scene.py  — the 22-duck world: entity configs, formation, index helpers
  obs.py    — per-duck observation builder (canonical frame)      [P3]
  rewards.py— soccer reward terms                                  [P3]
  vecenv.py — 22-lane multi-agent wrapper around mjlab's env       [P4]
"""

from mjlab_microduck.soccer import pitch, scene

__all__ = ["pitch", "scene"]
