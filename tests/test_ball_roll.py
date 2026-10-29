"""Ball rolling-resistance calibration (P1-T6).

The whole point of ``BALL_ROLL_LOSS`` is that a kicked ball behaves like a real
ball on a real pitch: it travels a distance proportional to v^2, stops in finite
time, and then STAYS stopped. Two failure modes this test exists to catch:

  * **No / too little resistance** — the ball is an ice puck: it never stops
    inside an episode, so "where the ball is" stops being predictable and the
    value function cannot learn.
  * **Creep tail** — MuJoCo's ``geom.friction[2]`` (rolling friction) is a
    torque-limit constraint: it leaves a low-speed tail that takes minutes to
    settle (5e-4 -> 6.19 m but 76 s). ``dof_frictionloss`` is a true Coulomb
    term and stops in finite time; this test asserts that.

Physics sanity for the initial condition: for travel along +x, rolling without
slip needs angular velocity about +y of ``+v/r``. Using ``-v/r`` makes the ball
SKID, which decays ~10x faster and fakes any calibration.
"""

import mujoco
import pytest

from mjlab_microduck.soccer import pitch as P

_FLOOR_AND_WALL_FREE = """
<mujoco><compiler angle="radian"/><worldbody>
<geom name="floor" type="plane" size="0 0 0.05"/></worldbody></mujoco>
"""

# Measured with BALL_ROLL_LOSS = 2.15e-4 (docs/soccer_tasks.md §3).
D_1MS, T_1MS = 1.95, 3.9


def _world() -> mujoco.MjModel:
    """Floor + the real ball entity, nowhere to bump into (unbounded runway)."""
    spec = mujoco.MjSpec.from_string(_FLOOR_AND_WALL_FREE)
    spec.attach(P.ball_spec(), prefix="", frame=spec.worldbody.add_frame(pos=[0, 0, 0]))
    model = spec.compile()
    model.opt.timestep = P.SIM_TIMESTEP
    return model


def _roll(model: mujoco.MjModel, v0: float, tmax: float = 120.0):
    """Roll at ``v0`` (+x, no slip) and return (distance, time-to-stop, drift)."""
    data = mujoco.MjData(model)
    data.qpos[0:7] = [0.0, 0.0, P.BALL_RADIUS, 1.0, 0.0, 0.0, 0.0]
    data.qvel[0:6] = [v0, 0.0, 0.0, 0.0, v0 / P.BALL_RADIUS, 0.0]
    mujoco.mj_forward(model, data)

    t = 0.0
    while t < tmax and abs(data.qvel[0]) > 2e-3:
        mujoco.mj_step(model, data)
        t += model.opt.timestep
    x_stop = float(data.qpos[0])

    # 10 s of extra simulation: a Coulomb term must keep it at rest.
    for _ in range(int(10.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return x_stop, t, abs(float(data.qpos[0]) - x_stop)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return _world()


def test_ball_entity_carries_the_friction_loss():
    model = _world()
    assert model.nv == 6
    assert (model.dof_frictionloss == P.BALL_ROLL_LOSS).all()
    # ...and the ball is still the real 15 g asset (an ad-hoc spec that lost the
    # explicit `<inertial>` inherits a density-derived 180 g mass and would make
    # every roll measurement 12x off).
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    assert model.body_mass[ball] == pytest.approx(0.015, rel=1e-6)


def test_roll_distance_and_time_at_1_ms(model):
    d, t, _ = _roll(model, 1.0)
    assert d == pytest.approx(D_1MS, abs=0.10), f"1 m/s rolled {d:.3f} m"
    assert t == pytest.approx(T_1MS, abs=0.5), f"1 m/s took {t:.2f} s to stop"


def test_short_push_stops_quickly(model):
    """0.3 m/s is the dribbling regime: the ball must not run away from the duck."""
    d, t, _ = _roll(model, 0.3)
    assert d == pytest.approx(0.18, abs=0.02), f"0.3 m/s rolled {d:.3f} m"
    assert t < 2.0


def test_no_creep_after_stop(model):
    _, _, drift = _roll(model, 1.0)
    assert drift < 1e-3, f"ball crept {drift * 1e3:.2f} mm after stopping"


@pytest.mark.parametrize("v0", [0.2, 0.6, 2.0])
def test_distance_scales_with_v_squared(model, v0):
    """Coulomb resistance => d = A v^2 (the signature of real rolling friction;
    a torque-limit constraint or viscous drag does NOT follow this)."""
    d, _, _ = _roll(model, v0)
    assert d / v0**2 == pytest.approx(D_1MS / 1.0**2, rel=0.10), f"{v0} m/s -> {d:.3f} m"


def test_ball_stays_on_the_ground(model):
    _, _, _ = _roll(model, 1.0)
    # Sanity: no hopping / sinking beyond the soft contact penetration.
    data = mujoco.MjData(model)
    data.qpos[0:7] = [0.0, 0.0, P.BALL_RADIUS, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    assert data.qpos[2] == pytest.approx(P.BALL_RADIUS, abs=2e-3)


# ── End-to-end: a shot on goal, in the real pitch ────────────────────────────
_GOAL_LINE = P.PITCH_L / 2
_NET_BACK = P.PITCH_L / 2 + P.GOAL_D


def _pitch_with_ball() -> mujoco.MjModel:
    spec = P.pitch_spec()
    spec.attach(P.ball_spec(), prefix="", frame=spec.worldbody.add_frame(pos=[0, 0, 0]))
    model = spec.compile()
    model.opt.timestep = P.SIM_TIMESTEP
    return model


@pytest.mark.parametrize("aim_y", [0.0, 0.20, -0.20])
def test_shot_on_goal_ends_up_inside_the_goal(aim_y):
    """From 2 m out at 1.2 m/s: crosses the line and STAYS there for 5 s.

    This is the check that makes the goal event robust — a ball that crosses the
    line and then rolls back out through the mouth turns "score" into a
    single-frame race in the event handler.
    """
    model = _pitch_with_ball()
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    data = mujoco.MjData(model)
    start_x = _GOAL_LINE - 2.0
    data.qpos[0:7] = [start_x, aim_y, P.BALL_RADIUS, 1.0, 0.0, 0.0, 0.0]
    # Roll towards +x, aimed at the goal: spin about +y is +v/r.
    data.qvel[0:6] = [1.2, 0.0, 0.0, 0.0, 1.2 / P.BALL_RADIUS, 0.0]
    mujoco.mj_forward(model, data)
    assert model.body_mass[ball] == pytest.approx(0.015, rel=1e-6)

    for _ in range(int(8.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
    assert x > _GOAL_LINE + 0.1, f"shot stopped short at x={x:.3f}"
    assert x < _NET_BACK, f"ball tunnelled through the net (x={x:.3f})"
    assert abs(y) < P.GOAL_W / 2, f"ball missed the mouth (y={y:.3f})"
    assert z == pytest.approx(P.BALL_RADIUS, abs=3e-3)

    # And it must be at REST inside the goal (not rattling around).
    assert abs(data.qvel[0]) < 5e-2, f"ball still moving at {data.qvel[0]:.3f} m/s"
