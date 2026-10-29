"""The 22-duck world: 11 v 11 formation + one mjlab Scene config.

WHY 22 SEPARATE ENTITIES (and not one big spec)
  An mjlab ``Entity`` is *one articulation*: it takes the spec's first joint as
  the floating base and treats every other joint as an articulated DOF. Packing
  22 ducks into one spec would make mjlab read 21 more freejoints as hinge
  joints — silently broken. So each duck is its own ``EntityCfg`` (prefix
  ``d00`` … ``d21``), and the whole set is stitched together by ``Scene``.

  Consequences that the rest of the code depends on:
    * ``cfg.scene.entities`` insertion order == joint/actuator order in the
      compiled model (**duck-major**: ``d00/*`` 21 qpos / 14 ctrl, then ``d01``…).
      ``scene_entity_cfgs()`` is the single place that order is defined.
    * Every duck gets its own ``spec_fn`` call, i.e. 22 fresh ``MjSpec`` loads
      (~1.8 s total). Never share one spec object.
    * Model names are prefixed: ``d00/left_hip_yaw``. Any regex-based selector
      must be scoped per entity (``SceneEntityCfg("d07", ...)``) or match
      ``^d\\d\\d/`` explicitly. Never use a bare ``.*knee.*`` across the world.

FORMATION
  4-4-2, both teams inside their own half (a legal kickoff, and more importantly
  the only thing that works at 0.17 m/s: a duck needs 20 s to cross the pitch, so
  nobody may start far from the ball). Team 1 is team 0 rotated 180° about the
  centre spot: ``(x, y, yaw) -> (-x, -y, yaw + pi)``. Never mirror y alone — that
  hands team 1 the other kicking foot and produces two different skills under one
  policy.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mujoco
from mjlab.entity import EntityCfg

from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    MICRODUCK_ALLCOLLISIONS_XML,
)
from mjlab_microduck.soccer import pitch as P

# ── Spawn ─────────────────────────────────────────────────────────────────────
DUCK_SPAWN_Z = 0.115
"""Trunk height at HOME, measured in sim (same value the Standup/BallKick envs
use). A 5 mm error here turns the stand into an impossible target."""

# ── Formation (team 0 frame: defends -x, attacks +x) ──────────────────────────
# (x, y) per role line. x is measured from the centre spot.
_FORMATION_ROWS: tuple[tuple[float, tuple[float, ...]], ...] = (
    (-3.15, (0.0,)),                                   # GK
    (-2.45, (-1.55, -0.52, 0.52, 1.55)),               # DEF
    (-1.05, (-1.55, -0.52, 0.52, 1.55)),               # MID
    (-0.55, (-0.62, 0.62)),                            # FWD
)

# Small-sided kickoff layouts (x, y) for team 0, all inside its own half.
# Hand-tuned so that (a) nobody spawns within 0.35 m of a teammate, (b) the
# closest duck to the ball is ~0.8-1.6 m away — at 0.17 m/s that is 5-9 s of
# walking, which is the whole point of the small-sided curriculum stages: the
# ball must be reachable by ONE duck without a 20 s trek.
_SMALL_SIDED: dict[int, tuple[tuple[float, float], ...]] = {
    # 1 v 1 starts 0.45 m out: the ball is inside one leg-swing + one stumble,
    # so a policy that cannot walk yet can still earn its first touch.
    1: ((-0.45, 0.0),),
    2: ((-0.90, -0.40), (-0.50, 0.40)),
    3: ((-1.40, 0.0), (-0.70, -0.70), (-0.70, 0.70)),
    4: ((-1.80, -0.85), (-1.80, 0.85), (-0.60, -0.60), (-0.60, 0.60)),
    5: ((-2.40, 0.0), (-1.50, -1.00), (-1.50, 1.00), (-0.70, -0.60), (-0.70, 0.60)),
    6: ((-2.60, -0.90), (-2.60, 0.90), (-1.60, -1.30), (-1.60, 1.30),
        (-0.70, -0.55), (-0.70, 0.55)),
}
"""Kickoff layouts for the small-sided curriculum stages (S0-S3).

Anything not listed (7-10) falls back to an even subsample of the 11-a-side
4-4-2 rows, so ``n_team`` never has to be one of these numbers exactly.
"""

ROLE_ORDER: tuple[str, ...] = ("GK", "DEF", "DEF", "DEF", "DEF",
                               "MID", "MID", "MID", "MID",
                               "FWD", "FWD")
"""Role of duck index 0..10 within a team (used by the role one-hot in obs)."""

assert len(ROLE_ORDER) == P.NUM_DUCKS_PER_TEAM


def formation_team0(n_team: int = P.NUM_DUCKS_PER_TEAM) -> tuple[tuple[float, float], ...]:
    """``n_team`` (x, y) kickoff positions for the team that attacks +x."""
    if n_team in _SMALL_SIDED:
        return _SMALL_SIDED[n_team]
    rows = [(x, y) for x, ys in _FORMATION_ROWS for y in ys]      # 11 slots
    if n_team == len(rows):
        return tuple((x, y) for (x, y) in rows)
    if n_team <= 0:
        return ()
    # Fallback: even subsample of the 11-a-side layout.
    step = (len(rows) - 1) / (n_team - 1) if n_team > 1 else 0.0
    return tuple(rows[round(i * step)] for i in range(n_team))


def mirror(xy: tuple[float, float]) -> tuple[float, float]:
    """180° rotation about the centre spot (NOT a y-flip)."""
    return (-xy[0], -xy[1])


def formation(n_team: int = P.NUM_DUCKS_PER_TEAM) -> tuple[tuple[float, float, float], ...]:
    """All ``2*n_team`` spawn poses as ``(x, y, yaw)``.

    Index ``i`` spawns duck ``d{i:02d}``: team 0 = ``0..n_team-1`` (yaw 0,
    facing +x), team 1 = ``n_team..2n_team-1`` (yaw pi).
    """
    t0 = formation_team0(n_team)
    poses = [(x, y, 0.0) for (x, y) in t0]
    poses += [(mx, my, math.pi) for (mx, my) in map(mirror, t0)]
    assert len(poses) == 2 * n_team, len(poses)
    return tuple(poses)


def n_ducks(n_team: int = P.NUM_DUCKS_PER_TEAM) -> int:
    return 2 * n_team


def team_of(duck_index: int, n_team: int = P.NUM_DUCKS_PER_TEAM) -> int:
    """0 = attacks +x, 1 = attacks -x."""
    return 0 if duck_index < n_team else 1


def attack_sign(duck_index: int, n_team: int = P.NUM_DUCKS_PER_TEAM) -> int:
    """+1 if the duck attacks towards +x, -1 otherwise (canonical frame flip)."""
    return 1 if team_of(duck_index, n_team) == 0 else -1


def teammate_indices(duck_index: int, n_team: int = P.NUM_DUCKS_PER_TEAM) -> tuple[int, ...]:
    lo = team_of(duck_index, n_team) * n_team
    return tuple(i for i in range(lo, lo + n_team) if i != duck_index)


def opponent_indices(duck_index: int, n_team: int = P.NUM_DUCKS_PER_TEAM) -> tuple[int, ...]:
    lo = (1 - team_of(duck_index, n_team)) * n_team
    return tuple(range(lo, lo + n_team))


# ── Entity configs ────────────────────────────────────────────────────────────
def duck_entity_name(i: int) -> str:
    return f"d{i:02d}"


def duck_spec(i: int, n_team: int = P.NUM_DUCKS_PER_TEAM) -> mujoco.MjSpec:
    """One duck's spec, positioned at its spawn pose.

    The root body's ``pos``/``quat`` in the SPEC matters even for a floating
    base: MuJoCo derives ``model.qpos0`` for a freejoint from it, i.e. it is the
    DEFAULT state of the compiled model. With all 22 ducks left at the origin,
    ``Simulation.__init__`` runs ``mj_forward`` on 22 interpenetrating ducks ->
    ~80 000 contacts -> ``mj_stackAlloc: stack overflow`` before iteration 0.
    Setting it here makes the default state the formation.

    A fresh ``MjSpec.from_file`` per call is required (mjlab calls this once per
    entity and attaches the result).
    """
    assert 0 <= i < 2 * n_team
    x, y, yaw = formation(n_team)[i]
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    root = spec.body("trunk_base")
    assert root is not None, "robot XML must have a trunk_base root body"
    root.pos = (x, y, DUCK_SPAWN_Z)
    root.quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return spec


def duck_entity_cfg(i: int, n_team: int = P.NUM_DUCKS_PER_TEAM) -> EntityCfg:
    """One duck: the shared all-collision robot cfg with a per-duck spawn pose.

    Everything else (BAM actuators, collision classes, HOME joint pose, soft
    limits) is inherited from ``MICRODUCK_ALLCOLLISIONS_ROBOT_CFG``; only the
    spawn pose differs. ``init_state`` must agree with ``duck_spec`` or the first
    reset teleports the duck.
    """
    x, y, yaw = formation(n_team)[i]
    return replace(
        MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
        spec_fn=lambda i=i, nt=n_team: duck_spec(i, nt),
        init_state=replace(
            HOME_FRAME,
            pos=(x, y, DUCK_SPAWN_Z),
            rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
        ),
    )


def scene_entity_cfgs(n_team: int = P.NUM_DUCKS_PER_TEAM) -> dict[str, EntityCfg]:
    """Entity dict for ``SceneCfg.entities`` — THIS defines model ordering.

    Order = ducks ``d00``…``d21``, then ``ball``, then ``pitch``. Do not sort it
    or move ``pitch`` to the front "for tidiness": the qpos layout and every
    duck-major index in the MDP code follow this dict's insertion order.
    """
    cfgs: dict[str, EntityCfg] = {
        duck_entity_name(i): duck_entity_cfg(i, n_team) for i in range(2 * n_team)
    }
    cfgs["ball"] = P.BALL_ENTITY_CFG
    cfgs["pitch"] = P.PITCH_ENTITY_CFG
    return cfgs


# ── Verification helpers (used by tests and by the env at build time) ─────────
def verify_world(
    model: mujoco.MjModel, n_team: int = P.NUM_DUCKS_PER_TEAM
) -> dict[str, int]:
    """Assert the compiled world has the expected layout; return its size stats.

    Cheap (no simulation), so the env can call it at startup: a silently wrong
    ``nq``/``nu`` otherwise shows up as a shape error hours later, or worse, as
    actions applied to the wrong duck.
    """
    stats = {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "njnt": model.njnt,
    }
    n_ducks = 2 * n_team
    n_dof_per_duck = 21  # 7 free + 14 servo joints
    assert model.nq == n_ducks * n_dof_per_duck + 7, f"nq={model.nq} (expected {n_ducks * n_dof_per_duck + 7})"
    assert model.nu == n_ducks * 14, f"nu={model.nu}"
    assert model.nv == n_ducks * (n_dof_per_duck - 1) + 6, f"nv={model.nv}"

    # Duck-major joint order: d00's 15 joints, then d01's, ...
    for i in range(n_ducks):
        prefix = f"{duck_entity_name(i)}/"
        first = model.joint(i * 15).name
        assert first == f"{prefix}trunk_base_freejoint", (
            f"joint {i * 15} is {first!r}, expected {prefix}trunk_base_freejoint — "
            "entity order changed"
        )
    # Actuators follow joint order (BAM actuators are sorted by target joint).
    for i in range(n_ducks):
        prefix = f"{duck_entity_name(i)}/"
        names = [model.actuator(i * 14 + k).name for k in range(14)]
        target = [model.joint(model.actuator_trnid[i * 14 + k, 0]).name for k in range(14)]
        assert all(t.startswith(prefix) for t in target), (
            f"actuator block {i} targets a different duck: {target[:2]}"
        )
        assert len(set(names)) == 14

    # The ball is right after the ducks in qpos and is the only other free body.
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball/ball")
    assert ball >= 0, "ball entity missing"
    ball_joint = model.body_jntadr[ball]
    assert model.jnt_type[ball_joint] == mujoco.mjtJoint.mjJNT_FREE
    return stats


def duck_free_joint_ids(
    model: mujoco.MjModel, n_team: int = P.NUM_DUCKS_PER_TEAM
) -> tuple[int, ...]:
    """Joint (not dof) ids of the duck freejoints, in duck order."""
    ids = []
    for i in range(2 * n_team):
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{duck_entity_name(i)}/trunk_base_freejoint"
        )
        assert jid >= 0, f"missing freejoint for duck {i}"
        ids.append(jid)
    return tuple(ids)


def ball_joint_id(model: mujoco.MjModel) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball/ball_free")
    assert jid >= 0, "ball freejoint missing"
    return jid


def duck_qpos_slices(
    model: mujoco.MjModel, n_team: int = P.NUM_DUCKS_PER_TEAM
) -> tuple[slice, ...]:
    """Per-duck ``qpos`` slice covering root(7) + 14 joints (duck-major)."""
    out = []
    for jid in duck_free_joint_ids(model, n_team):
        start = int(model.jnt_qposadr[jid])
        out.append(slice(start, start + 21))
    return tuple(out)


def duck_dof_slices(
    model: mujoco.MjModel, n_team: int = P.NUM_DUCKS_PER_TEAM
) -> tuple[slice, ...]:
    """Per-duck ``qvel``/``qacc`` slice (6 root dofs + 14 joints)."""
    out = []
    for jid in duck_free_joint_ids(model, n_team):
        start = int(model.jnt_dofadr[jid])
        out.append(slice(start, start + 20))
    return tuple(out)


def duck_ctrl_slices(n_team: int = P.NUM_DUCKS_PER_TEAM) -> tuple[slice, ...]:
    """Per-duck ``ctrl`` slice — valid only for a duck-major actuator layout."""
    return tuple(slice(i * 14, (i + 1) * 14) for i in range(2 * n_team))


def ball_qpos_slice(model: mujoco.MjModel) -> slice:
    start = int(model.jnt_qposadr[ball_joint_id(model)])
    return slice(start, start + 7)


def ball_dof_slice(model: mujoco.MjModel) -> slice:
    start = int(model.jnt_dofadr[ball_joint_id(model)])
    return slice(start, start + 6)
