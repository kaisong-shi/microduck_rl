"""Soccer pitch + ball for the 22-duck (11 v 11) task family.

WHY A SEPARATE MODULE
  The pitch is shared by every soccer env (S0 1v1 … S4 11v11), by the training
  world builder, by the headless eval and by the viewer. All of its numbers are
  measured/sized constants (see ``docs/soccer_tasks.md`` §0), so they live here
  and nowhere else — an env cfg must never hardcode ``3.5``.

WHERE THE FLOOR COMES FROM
  ``pitch_spec()`` can emit its own ground plane (``include_floor=True``, used by
  the standalone/tests/scripts path). Inside an mjlab env the floor is
  **mjlab's terrain plane** (``cfg.scene.terrain.terrain_type = "plane"``), which
  is the body/geom named ``terrain`` — the whole repo's contact patterns
  (``ContactMatch(mode="body", pattern="terrain")``) and the env-origin
  machinery key off that name, so the pitch entity attaches with
  ``include_floor=False`` and only adds lines / goals / walls. Grass look is a
  terrain *material* override, not extra geometry (see
  ``PITCH_TERRAIN_TEXTURE`` / ``PITCH_TERRAIN_MATERIAL``).

MIRROR / HANDEDNESS
  Team 0 defends ``x = -PITCH_L/2`` and attacks ``+x``; team 1 is the exact 180°
  rotation of team 0 about the centre spot: ``(x, y, yaw) -> (-x, -y, yaw + pi)``.
  Never mirror y alone — that swaps the kicking foot and produces two different
  skills wearing one policy.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils import spec_config as spec_cfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_XML

# ── Pitch ─────────────────────────────────────────────────────────────────────
PITCH_L = 7.0
"""Goal line to goal line (m). Long axis = x."""

PITCH_W = 4.5
"""Touchline to touchline (m)."""

CENTER_CIRCLE_R = 0.61
"""Centre circle radius — 0.17 x PITCH_L/2, human ratio 9.15/52.5 = 0.174."""

PENALTY_D = 1.10
"""Penalty area depth from the goal line (m)."""

PENALTY_W = 2.67
"""Penalty area width (m) — 0.59 x PITCH_W, human ratio 40.3/68 = 0.59."""

LINE_W = 0.02
"""Field-line width (m). Purely visual (collisionless)."""

LINE_Z = 0.002
"""Field lines sit 2 mm above the floor so they never z-fight the plane."""

# ── Goals ─────────────────────────────────────────────────────────────────────
GOAL_W = 0.70
"""Goal mouth width (m) = 15.6 % of PITCH_W (human 7.32/68 = 10.8 %)."""

GOAL_H = 0.35
"""Goal mouth height (m) = 1.4 duck heights (human 2.44/1.8 = 1.36)."""

GOAL_D = 0.25
"""Goal depth behind the goal line (m)."""

POST_HALF = 0.010
"""Half the post/crossbar cross-section (m) -> 20 mm square posts."""

NET_T = 0.010
"""Net/back-panel thickness (m)."""

# ── Boundary walls ────────────────────────────────────────────────────────────
WALL_H = 0.20
"""Invisible boundary wall height (m) — above the 0.25 m duck but below its head."""

WALL_MARGIN = 0.30
"""Distance from the touchline/goal line to the wall's inner face (m)."""

WALL_T = 0.02
"""Boundary wall thickness (m)."""

# ── Ball ──────────────────────────────────────────────────────────────────────
BALL_RADIUS = 0.035
"""70 mm ball (matches ``ball.xml``)."""

BALL_ROLL_LOSS = 2.15e-4
"""``frictionloss`` written on the ball freejoint's 6 DOFs (in-memory only).

MuJoCo's ``geom.friction[2]`` is a *torque-limit constraint*, not physical
rolling resistance: at 5e-4 a 1 m/s ball travels 6.19 m but creeps for 76 s.
``dof_frictionloss`` is a constant force/torque (true Coulomb rolling
resistance), so distance goes as v^2 and the ball stops in finite time.

Measured with the REAL 15 g / 70 mm ball (``ball.xml`` explicit
``diaginertia``) on a plane at dt=0.005, initial condition = pure rolling
(no skidding):

    d = 1.95 v^2,  t = 3.9 v
    1.00 m/s -> 1.95 m / 3.9 s     0.30 m/s -> 0.18 m / 1.2 s
    0.60 m/s -> 0.70 m / 2.3 s     2.00 m/s -> 7.80 m / 7.8 s

Post-stop drift < 0.02 mm over 10 s (no creep tail).

DO NOT re-derive this from ``frictionloss/mass``: the linear DOFs contribute
~3 % of the resistance, the *angular* DOFs (torque / r) supply the rest, and a
wrong-sign initial spin makes the ball skid instead of roll — a skid decays 10x
faster and silently fakes any calibration (that bug cost one calibration
round-trip). The ball's *contact* model is untouched (``condim=3``, ``friction``
from ``ball.xml``) so ball-ground / ball-foot contact stays bit-identical to the
``BallKick`` training asset; only rolling resistance differs.
"""

SIM_TIMESTEP = 0.005
"""Physics timestep (s) — matches the repo's training recipe."""

CONTROL_DECIMATION = 4
"""Physics steps per control step -> 50 Hz policy (matches the repo)."""

# ── Match / sim ───────────────────────────────────────────────────────────────
EPISODE_S = 30.0
"""Training episode length for S0–S3. One possession cycle is ~13 s (a 1.9 m
kick, ~4 s of rolling, then ~11 s for the kicker to walk 1.9 m at 0.17 m/s)."""

EPISODE_S_11V11 = 40.0
"""S4/S5 11v11 only — a full-size game moves the ball further per episode."""

NJMAX = 2600
NCONMAX = 1100
"""Contact budget per world (mjwarp allocates ``nconmax * nworld`` contact slots,
so nconmax — not njmax — is what eats VRAM).

Measured on the real 22-duck + ball + pitch world at 512 worlds, with random
actions and ducks collapsing into heaps (worst case for contacts):

    njmax=1850 nconmax=900   -> "nefc overflow, increase njmax to ~1950"
    njmax=2200 nconmax=1100  -> no overflow over 2000 physics steps,
                                max nefc/world = 2017, 12.6 GB
    njmax=2400 nconmax=1600  -> CUDA graph OOM at 512 worlds
    nconmax=800              -> "increase nconmax to 803 … 821" and still climbing

njmax keeps ~25 % headroom over the measured 2017 because a 30 s episode ends
with a 22-duck pile; njmax is cheap, nconmax is not.
"""

NUM_WORLDS = 512
"""Parallel matches per training run (512 x 22 = 11264 policy lanes).

Measured throughput of the real world (RTX 4090, dt=0.005, decimation 4):

    worlds   ms/control step   ms per world
     128          159.0            1.24
     512          331.0            0.65   <- 4x the envs for 2x the time

so 512 is the sweet spot: 24 control steps/iteration -> ~7.9 s/iteration of pure
physics (the reason P5's 11v11 stage is budgeted in hours, not minutes).
"""

NUM_DUCKS_PER_TEAM = 11
NUM_DUCKS = 2 * NUM_DUCKS_PER_TEAM


# ── Derived helpers (use these, never the raw numbers) ────────────────────────
def goal_line_x(side: int) -> float:
    """x of the team-``side`` goal line (+1 = team 0's target, -1 = its own)."""
    assert side in (-1, 1)
    return side * PITCH_L / 2


def goal_center(side: int) -> tuple[float, float, float]:
    """World position of the centre of the goal mouth (on the goal line)."""
    return (goal_line_x(side), 0.0, GOAL_H / 2)


def wall_inner_x() -> float:
    return PITCH_L / 2 + WALL_MARGIN


def wall_inner_y() -> float:
    return PITCH_W / 2 + WALL_MARGIN


# ── Field lines (visual only) ─────────────────────────────────────────────────
_LINE_RGBA = (0.95, 0.95, 0.95, 0.85)
_CIRCLE_SEGMENTS = 24


def _add_line(
    body,
    name: str,
    center: tuple[float, float],
    length: float,
    yaw: float = 0.0,
) -> None:
    """Add one collisionless field line (a thin flat box on the floor)."""
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, LINE_W / 2, LINE_Z / 2),
        pos=(center[0], center[1], LINE_Z / 2),
        euler=(0.0, 0.0, yaw),
        contype=0,
        conaffinity=0,
        rgba=_LINE_RGBA,
    )


def _add_field_lines(body) -> None:
    hl, hw = PITCH_L / 2, PITCH_W / 2
    # Perimeter.
    _add_line(body, "line_touch_pos_y", (0.0, hw), PITCH_L)
    _add_line(body, "line_touch_neg_y", (0.0, -hw), PITCH_L)
    _add_line(body, "line_goal_pos_x", (hl, 0.0), PITCH_W, yaw=math.pi / 2)
    _add_line(body, "line_goal_neg_x", (-hl, 0.0), PITCH_W, yaw=math.pi / 2)
    # Halfway line + centre spot.
    _add_line(body, "line_halfway", (0.0, 0.0), PITCH_W, yaw=math.pi / 2)
    body.add_geom(
        name="line_center_spot",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(LINE_W, LINE_Z / 2),  # radius, half-height (a flat disc)
        pos=(0.0, 0.0, LINE_Z / 2),
        contype=0,
        conaffinity=0,
        rgba=_LINE_RGBA,
    )
    # Centre circle, approximated by tangent segments (MuJoCo has no torus and
    # this is decoration only).
    seg_half = CENTER_CIRCLE_R * math.sin(math.pi / _CIRCLE_SEGMENTS)
    for i in range(_CIRCLE_SEGMENTS):
        a = 2 * math.pi * i / _CIRCLE_SEGMENTS
        _add_line(
            body,
            f"line_circle_{i:02d}",
            (CENTER_CIRCLE_R * math.cos(a), CENTER_CIRCLE_R * math.sin(a)),
            2 * seg_half,
            yaw=a + math.pi / 2,
        )
    # Penalty areas (3 lines each side; the goal line closes the box).
    for side in (-1, 1):
        px = side * (hl - PENALTY_D)
        s = f"p{'o' if side > 0 else 'n'}s"
        _add_line(body, f"line_box_{s}_front", (px, 0.0), PENALTY_W, yaw=math.pi / 2)
        for sgn, tag in ((1, "y+"), (-1, "y-")):
            _add_line(
                body,
                f"line_box_{s}_{tag}".replace("+", "p").replace("-", "n"),
                (side * (hl - PENALTY_D / 2), sgn * PENALTY_W / 2),
                PENALTY_D,
            )


# ── Goals ─────────────────────────────────────────────────────────────────────
_POST_RGBA = (0.92, 0.92, 0.92, 1.0)
_NET_RGBA = (0.85, 0.88, 0.92, 0.35)


def _add_goal(body, side: int) -> None:
    """One goal: 2 posts + crossbar (solid) + back/side panels (net)."""
    tag = "pos" if side > 0 else "neg"
    x = goal_line_x(side)
    gh, gw, gd = GOAL_H, GOAL_W, GOAL_D

    for sgn, ytag in ((1, "ypos"), (-1, "yneg")):
        body.add_geom(
            name=f"goal_{tag}_post_{ytag}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(POST_HALF, POST_HALF, gh / 2),
            pos=(x, sgn * gw / 2, gh / 2),
            rgba=_POST_RGBA,
        )
    body.add_geom(
        name=f"goal_{tag}_bar",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(POST_HALF, gw / 2 + POST_HALF, POST_HALF),
        pos=(x, 0.0, gh),
        rgba=_POST_RGBA,
    )
    # Net box: back panel + two side panels. Solid for EVERYTHING, ball
    # included, on purpose:
    #   * MuJoCo collides a pair when (contype_1 & conaffinity_2) |
    #     (contype_2 & conaffinity_1). The ball's contype is 1 and so is a
    #     duck's, so "a panel that stops ducks but not the ball" is impossible
    #     without re-classing the ball away from the floor — not worth it.
    #   * A solid net is the better model anyway: an arriving ball dead-stops
    #     (default solref gives no restitution) and therefore STAYS in the goal,
    #     so the goal event does not depend on catching a single frame while the
    #     ball rolls back out through the mouth.
    body.add_geom(
        name=f"goal_{tag}_net_back",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(NET_T / 2, gw / 2, gh / 2),
        pos=(side * (PITCH_L / 2 + gd), 0.0, gh / 2),
        rgba=_NET_RGBA,
    )
    for sgn, ytag in ((1, "ypos"), (-1, "yneg")):
        body.add_geom(
            name=f"goal_{tag}_net_{ytag}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(gd / 2, NET_T / 2, gh / 2),
            pos=(side * (PITCH_L / 2 + gd / 2), sgn * gw / 2, gh / 2),
            rgba=_NET_RGBA,
        )


# ── Boundary walls ────────────────────────────────────────────────────────────
def _add_walls(body) -> None:
    """Four invisible walls, so a stray ball/duck is not a lost episode."""
    t = WALL_T
    xi, yi = wall_inner_x(), wall_inner_y()
    # Inner face at ±(half + margin); thickness grows outward.
    cx, cy = xi + t / 2, yi + t / 2
    specs = (
        ("wall_pos_x", (cx, 0.0), (t / 2, cy, WALL_H / 2)),
        ("wall_neg_x", (-cx, 0.0), (t / 2, cy, WALL_H / 2)),
        ("wall_pos_y", (0.0, cy), (cx, t / 2, WALL_H / 2)),
        ("wall_neg_y", (0.0, -cy), (cx, t / 2, WALL_H / 2)),
    )
    for name, (px, py), size in specs:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=(px, py, WALL_H / 2),
            rgba=(1.0, 1.0, 1.0, 0.0),   # alpha 0 -> invisible, still solid
        )


# ── Spec builders ─────────────────────────────────────────────────────────────
_FLOOR_XML = """
<mujoco model="soccer_pitch">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-20"/>
  </visual>
  <worldbody>
    <light name="sun" pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0"
          rgba="0.17 0.40 0.15 1"/>
  </worldbody>
</mujoco>
"""


def pitch_spec(include_floor: bool = True) -> mujoco.MjSpec:
    """Build the pitch: (optional floor) + field lines + 2 goals + 4 walls.

    ``include_floor=False`` is what the mjlab env uses — mjlab's terrain plane is
    the floor there, and two coincident planes would be a z-fighting, double
    contact mess. Contains no ball and no ducks; those are separate entities.
    """
    if include_floor:
        spec = mujoco.MjSpec.from_string(_FLOOR_XML)
    else:
        spec = mujoco.MjSpec()
        spec.modelname = "soccer_pitch"

    body = spec.worldbody.add_body(name="pitch", pos=(0.0, 0.0, 0.0))
    _add_field_lines(body)
    for side in (-1, 1):
        _add_goal(body, side)
    _add_walls(body)
    return spec


def ball_spec() -> mujoco.MjSpec:
    """``ball.xml`` verbatim + a rolling-resistance ``frictionloss``.

    The file itself is NOT modified (it is the shared BallKick asset): the
    friction loss is written into the in-memory spec only, before compile. It
    lands on all six freejoint DOFs (3 linear + 3 angular). The angular DOFs
    carry the physics of rolling resistance (torque / r); the linear ones add
    ~3 %, and together they also damp the ball's spin, which ``condim=3`` has no
    torsional friction for — otherwise a struck ball would spin in place
    forever.
    """
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))
    joint = spec.joint("ball_free")
    assert joint is not None, "ball.xml must declare a freejoint named 'ball_free'"
    joint.frictionloss = BALL_ROLL_LOSS
    return spec


# ── mjlab entity configs ──────────────────────────────────────────────────────
PITCH_ENTITY_CFG = EntityCfg(
    spec_fn=lambda: pitch_spec(include_floor=False),
    # No `collisions` editor on purpose: with an empty tuple mjlab leaves the
    # spec's own contype/conaffinity alone, which is how the lines stay
    # collisionless and the net back panel stays ball-transparent.
)

BALL_ENTITY_CFG = EntityCfg(
    spec_fn=ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_RADIUS)),
)

# Grass look for mjlab's terrain plane (body/geom name "terrain"). Applied by
# the env cfg as ``cfg.scene.terrain.textures = (...)``. Checker with
# texrepeat=(1, 9) gives nine 0.5 m bands across the pitch — mowing stripes.
PITCH_TERRAIN_TEXTURE = spec_cfg.TextureCfg(
    name="grass",
    type="2d",
    builtin="checker",
    mark="none",
    rgb1=(0.17, 0.40, 0.15),
    rgb2=(0.14, 0.35, 0.13),
    width=300,
    height=300,
)

PITCH_TERRAIN_MATERIAL = spec_cfg.MaterialCfg(
    name="grass",
    texture="grass",
    texuniform=True,
    texrepeat=(1.0, 9.0),
    reflectance=0.05,
    geom_names_expr=("terrain$",),
)
