"""Pitch geometry invariants (P1-T5).

These lock the numbers the whole soccer task family is balanced around: if the
pitch, goal or wall dimensions drift, tempo / shot-range / scoring-rate
estimates in docs/soccer_tasks.md are silently wrong, and the bug surfaces only
as "the policy plays weird" hours into training.
"""

import math

import mujoco
import numpy as np
import pytest

from mjlab_microduck.soccer import pitch as P


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return P.pitch_spec().compile()


def gid(m: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert i >= 0, f"geom {name!r} missing"
    return i


# ── Constants ────────────────────────────────────────────────────────────────
def test_pitch_dimensions():
    assert (P.PITCH_L, P.PITCH_W) == (7.0, 4.5)
    # Density check: 4.8 duck-heights per duck across the short axis
    # (human football ~10 at 68 m / 1.8 m; SPL 5.8). Both were chosen on
    # purpose — see docs/soccer_tasks.md §0.
    assert 4.0 < P.PITCH_W / 2 / 0.25 < 10.0


def test_goal_is_goal_shaped():
    assert (P.GOAL_W, P.GOAL_H, P.GOAL_D) == (0.70, 0.35, 0.25)
    # Goal mouth must be wider than the ball, and taller than 4 ball diameters.
    assert P.GOAL_W > 8 * P.BALL_RADIUS
    assert P.GOAL_H > 4 * P.BALL_RADIUS
    # Goal mouth as a fraction of pitch width ~ human 10.8 %.
    assert 0.10 < P.GOAL_W / P.PITCH_W < 0.20


def test_ball_matches_asset():
    assert P.BALL_RADIUS == 0.035


# ── Spec structure ───────────────────────────────────────────────────────────
def test_spec_has_no_joints_and_a_single_pitch_body():
    sp = P.pitch_spec(include_floor=False)
    m = sp.compile()
    assert m.nq == 0
    assert m.nu == 0
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    assert bodies == ["world", "pitch"]


def test_include_floor_toggles_the_plane():
    with_floor = P.pitch_spec(include_floor=True).compile()
    without = P.pitch_spec(include_floor=False).compile()
    assert with_floor.geom_type[gid(with_floor, "floor")] == mujoco.mjtGeom.mjGEOM_PLANE
    assert with_floor.nlight == 1
    assert without.nlight == 0
    assert without.ngeom == with_floor.ngeom - 1


def test_field_lines_are_collisionless(model):
    line_geoms = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
    ]
    line_geoms = [n for n in line_geoms if n.startswith("line_")]
    assert len(line_geoms) >= 8
    for name in line_geoms:
        i = gid(model, name)
        assert model.geom_contype[i] == 0, name
        assert model.geom_conaffinity[i] == 0, name


def test_walls_are_invisible_but_solid(model):
    for name in ("wall_pos_x", "wall_neg_x", "wall_pos_y", "wall_neg_y"):
        i = gid(model, name)
        assert model.geom_rgba[i][3] == 0.0, f"{name} must be invisible"
        assert model.geom_contype[i] != 0, f"{name} must be solid"


def test_wall_positions():
    m = P.pitch_spec().compile()
    xi, yi = P.wall_inner_x(), P.wall_inner_y()
    assert xi == pytest.approx(P.PITCH_L / 2 + P.WALL_MARGIN)
    assert yi == pytest.approx(P.PITCH_W / 2 + P.WALL_MARGIN)
    for name, axis, inner in (
        ("wall_pos_x", 0, +xi),
        ("wall_neg_x", 0, -xi),
        ("wall_pos_y", 1, +yi),
        ("wall_neg_y", 1, -yi),
    ):
        i = gid(m, name)
        half = m.geom_size[i][axis]
        pos = m.geom_pos[i][axis]
        # Inner face sits exactly at +-inner.
        assert (pos - half) if inner > 0 else (pos + half) == pytest.approx(inner, abs=1e-9)
    # The wall must clear the goal box (goal back panel is the deepest object).
    deepest = P.PITCH_L / 2 + P.GOAL_D + P.NET_T
    assert deepest < xi, "goal sticks out through the boundary wall"
    # ...and be tall enough to stop the ball, low enough to see over.
    assert P.WALL_H >= 5 * P.BALL_RADIUS


# ── Goals ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("side,tag", [(-1, "neg"), (1, "pos")])
def test_goal_geometry(model, side, tag):
    x = P.goal_line_x(side)
    assert x == pytest.approx(side * 3.5)

    for ytag, sgn in (("ypos", 1), ("yneg", -1)):
        i = gid(model, f"goal_{tag}_post_{ytag}")
        assert model.geom_pos[i][0] == pytest.approx(x)
        assert model.geom_pos[i][1] == pytest.approx(sgn * P.GOAL_W / 2)
        assert model.geom_pos[i][2] == pytest.approx(P.GOAL_H / 2)
        # 20 mm square posts.
        assert model.geom_size[i][0] == pytest.approx(P.POST_HALF)
        assert model.geom_size[i][1] == pytest.approx(P.POST_HALF)

    bar = gid(model, f"goal_{tag}_bar")
    assert model.geom_pos[bar][2] == pytest.approx(P.GOAL_H)
    # Crossbar spans the full mouth including both post halves.
    assert model.geom_size[bar][1] == pytest.approx(P.GOAL_W / 2 + P.POST_HALF)

    net = gid(model, f"goal_{tag}_net_back")
    assert model.geom_pos[net][0] == pytest.approx(side * (P.PITCH_L / 2 + P.GOAL_D))
    # Solid for everything, including the ball: an arriving ball must stay in
    # the goal instead of rolling back out through the mouth.
    assert model.geom_contype[net] != 0
    assert model.geom_conaffinity[net] != 0


def test_goal_mouth_clears_the_ball():
    """A rolling ball must fit between the posts with margin on both sides."""
    clearance = (P.GOAL_W - 2 * P.POST_HALF) / 2 - P.BALL_RADIUS
    assert clearance > 0.05, f"ball barely fits in the goal ({clearance:.3f} m)"


def test_goal_center_helper():
    for side in (-1, 1):
        cx, cy, cz = P.goal_center(side)
        assert cx == pytest.approx(P.goal_line_x(side))
        assert (cy, cz) == pytest.approx((0.0, P.GOAL_H / 2))


# ── Field-line layout in world space ─────────────────────────────────────────
def _world_extent(model, name):
    """Full world-space extent of a box geom as (dx, dy), from its quat."""
    i = gid(model, name)
    s = model.geom_size[i]
    w, x, y, z = model.geom_quat[i]
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return (np.abs(rot) @ np.array(s) * 2)[:2]


def test_field_lines_are_oriented_right(model):
    """Touchlines run along x, goal/halfway/front lines along y.

    This is asserted numerically because a camera render is a bad judge here:
    at a low camera angle the thin vertical lines sit near the frame edge and
    look "missing", which costs an hour of chasing a non-bug.
    """
    half_l, half_w = P.PITCH_L / 2, P.PITCH_W / 2
    cases = {
        "line_touch_pos_y": ((0.0, half_w), (P.PITCH_L, P.LINE_W)),
        "line_touch_neg_y": ((0.0, -half_w), (P.PITCH_L, P.LINE_W)),
        "line_goal_pos_x": ((half_l, 0.0), (P.LINE_W, P.PITCH_W)),
        "line_goal_neg_x": ((-half_l, 0.0), (P.LINE_W, P.PITCH_W)),
        "line_halfway": ((0.0, 0.0), (P.LINE_W, P.PITCH_W)),
    }
    for name, (center, extent) in cases.items():
        i = gid(model, name)
        assert model.geom_pos[i][:2] == pytest.approx(center, abs=1e-9), name
        got = _world_extent(model, name)
        assert got == pytest.approx(np.array(extent), abs=1e-9), (
            f"{name}: world extent {got} != {extent}"
        )

    # Penalty boxes: front line along y at |x| = L/2 - PENALTY_D, sides along x.
    for side, tag in ((-1, "pns"), (1, "pos")):
        front = gid(model, f"line_box_{tag}_front")
        assert model.geom_pos[front][0] == pytest.approx(side * (half_l - P.PENALTY_D))
        assert _world_extent(model, f"line_box_{tag}_front") == pytest.approx(
            (P.LINE_W, P.PENALTY_W), abs=1e-9
        )
        for ytag, sgn in (("yp", 1), ("yn", -1)):
            name = f"line_box_{tag}_{ytag}"
            i = gid(model, name)
            assert model.geom_pos[i][1] == pytest.approx(sgn * P.PENALTY_W / 2)
            assert _world_extent(model, name) == pytest.approx(
                (P.PENALTY_D, P.LINE_W), abs=1e-9
            )


def test_centre_circle_segments_lie_on_the_circle():
    m = P.pitch_spec().compile()
    radii = []
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith("line_circle_"):
            radii.append(math.hypot(*m.geom_pos[i][:2]))
    assert len(radii) == P._CIRCLE_SEGMENTS
    # Each tangential segment is centred ON the circle (chord across it).
    assert np.allclose(radii, P.CENTER_CIRCLE_R, atol=1e-9)


# ── Contact budget ───────────────────────────────────────────────────────────
def test_goal_post_count_matches_two_goals(model):
    posts = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
    ]
    assert len([n for n in posts if n and n.startswith("goal_") and "post" in n]) == 4
