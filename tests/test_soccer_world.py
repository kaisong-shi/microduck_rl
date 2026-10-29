"""The 22-duck world (P2-T2/T7/T8).

Split in two:
  * layout tests  — pure ``mujoco`` + mjlab ``Scene`` compile, CPU, ~3 s. These
    lock the duck-major qpos/ctrl ordering that every MDP function indexes into.
  * GPU tests     — build the real scene through mjlab + stepping, marked
    ``skipif`` when no CUDA is available (they cost ~10 s).

The ordering tests matter more than they look: with 22 entities, a dict-order
slip (a "tidier" sort, a duck moved before the ball) silently re-indexes the
whole action vector — every duck would execute its neighbour's action and
nothing would raise.
"""

import math

import mujoco
import numpy as np
import pytest
import torch

from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer import scene as S


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


# ── Formation ────────────────────────────────────────────────────────────────
def test_formation_has_22_ducks_inside_the_pitch():
    poses = S.formation()
    assert len(poses) == P.NUM_DUCKS == 22
    half_l, half_w = P.PITCH_L / 2, P.PITCH_W / 2
    for x, y, _ in poses:
        assert abs(x) < half_l - 0.2, f"duck spawns on/behind the goal line: x={x}"
        assert abs(y) < half_w - 0.2, f"duck spawns on/outside the touchline: y={y}"


def test_both_teams_start_in_their_own_half():
    for i, (x, y, yaw) in enumerate(S.formation()):
        if S.team_of(i) == 0:
            assert x < 0, f"team0 duck {i} starts in the attacking half (x={x})"
            assert yaw == pytest.approx(0.0)
        else:
            assert x > 0, f"team1 duck {i} starts in the defending half (x={x})"
            assert yaw == pytest.approx(math.pi)


def test_team1_is_an_exact_180_rotation_not_a_y_flip():
    poses = S.formation()
    for i in range(P.NUM_DUCKS_PER_TEAM):
        x0, y0, _ = poses[i]
        x1, y1, _ = poses[i + P.NUM_DUCKS_PER_TEAM]
        assert (x1, y1) == pytest.approx((-x0, -y0))
        # A y-flip would put duck j at (x0, -y0) — i.e. it would keep the SAME
        # kicking foot pointing at the ball. Assert it does not happen.
        if abs(y0) > 1e-6:
            assert y1 != pytest.approx(y0)


def test_formation_is_not_self_overlapping():
    poses = S.formation()
    xy = np.array([(x, y) for x, y, _ in poses])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    # A duck is ~0.10 m wide; 0.35 m keeps spawns clear of each other.
    assert d.min() > 0.35, f"two ducks spawn {d.min():.3f} m apart"


def test_roles_cover_the_team():
    assert len(S.ROLE_ORDER) == P.NUM_DUCKS_PER_TEAM
    assert S.ROLE_ORDER.count("GK") == 1
    assert S.ROLE_ORDER.count("FWD") == 2
    assert set(S.ROLE_ORDER) == {"GK", "DEF", "MID", "FWD"}


def test_team_and_attack_helpers():
    for i in range(P.NUM_DUCKS):
        team = S.team_of(i)
        assert team in (0, 1)
        assert S.attack_sign(i) == (1 if team == 0 else -1)
        mates = S.teammate_indices(i)
        opps = S.opponent_indices(i)
        assert len(mates) == 10 and len(opps) == 11
        assert i not in mates and i not in opps
        assert all(S.team_of(j) == team for j in mates)
        assert all(S.team_of(j) != team for j in opps)


# ── Compiled world layout ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def world():
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    cfg = SceneCfg(
        num_envs=1,
        env_spacing=8.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=S.scene_entity_cfgs(),
    )
    scene = Scene(cfg, device="cpu")
    return scene.compile()


def test_world_sizes(world):
    stats = S.verify_world(world)
    assert stats["nq"] == 469   # 22 * 21 + 7 (ball)
    assert stats["nv"] == 446   # 22 * 20 + 6
    assert stats["nu"] == 308   # 22 * 14
    assert stats["njnt"] == 331  # 22 * 15 + 1


def test_qpos0_is_the_formation_not_the_origin(world):
    """A freejoint's ``qpos0`` comes from the root body's spec pose. If every
    duck defaulted to the origin, ``Simulation.__init__``'s ``mj_forward`` would
    run on 22 interpenetrating ducks (~80 000 contacts) and die with
    ``mj_stackAlloc: stack overflow`` before iteration 0."""
    root_z = [world.qpos0[s.start + 2] for s in S.duck_qpos_slices(world)]
    assert np.allclose(root_z, S.DUCK_SPAWN_Z)
    xy = np.array([[world.qpos0[s.start], world.qpos0[s.start + 1]]
                   for s in S.duck_qpos_slices(world)])
    want = np.array([(x, y) for x, y, _ in S.formation()])
    assert np.allclose(xy, want)
    # All distinct: no duck shares a spawn with another.
    assert len({tuple(np.round(p, 4)) for p in xy}) == P.NUM_DUCKS


def test_duck_major_joint_order(world):
    for i in range(P.NUM_DUCKS):
        assert world.joint(i * 15).name == f"d{i:02d}/trunk_base_freejoint"


def test_actuators_follow_duck_order(world):
    for i in range(P.NUM_DUCKS):
        targets = [
            world.joint(world.actuator_trnid[i * 14 + k, 0]).name for k in range(14)
        ]
        assert all(t.startswith(f"d{i:02d}/") for t in targets), targets
        # Servo joints only — never the freejoint, never a passive_* joint.
        assert not any("freejoint" in t or "passive_" in t for t in targets)


def test_qpos_and_ctrl_slices_are_disjoint_and_cover_everything(world):
    qslices = S.duck_qpos_slices(world)
    dslices = S.duck_dof_slices(world)
    cslices = S.duck_ctrl_slices()
    assert len(qslices) == len(dslices) == len(cslices) == P.NUM_DUCKS
    covered_q = sorted(i for s in qslices for i in range(s.start, s.stop))
    bs = S.ball_qpos_slice(world)
    covered_q += list(range(bs.start, bs.stop))
    assert covered_q == list(range(world.nq))
    covered_d = sorted(i for s in dslices for i in range(s.start, s.stop))
    bd = S.ball_dof_slice(world)
    covered_d += list(range(bd.start, bd.stop))
    assert covered_d == list(range(world.nv))
    covered_c = sorted(i for s in cslices for i in range(s.start, s.stop))
    assert covered_c == list(range(world.nu))


def test_duck_root_bodies_are_where_the_formation_says(world):
    data = mujoco.MjData(world)
    mujoco.mj_forward(world, data)
    for i, (x, y, yaw) in enumerate(S.formation()):
        start = S.duck_qpos_slices(world)[i].start
        assert data.qpos[start] == pytest.approx(x, abs=1e-6)
        assert data.qpos[start + 1] == pytest.approx(y, abs=1e-6)
        assert data.qpos[start + 2] == pytest.approx(S.DUCK_SPAWN_Z, abs=1e-4)
        # Quaternion encodes the facing direction.
        w, _, _, z = data.qpos[start + 3:start + 7]
        assert math.cos(yaw / 2) == pytest.approx(w, abs=1e-6)
        assert math.sin(yaw / 2) == pytest.approx(z, abs=1e-6)


def test_home_pose_is_applied_per_duck(world):
    """The spawn pose lives in the merged ``init_state`` keyframe, not in
    ``qpos0``: mjlab writes ``init_state.joint_pos`` into the key at build time
    and the env applies it at reset. NOTE qpos0 intentionally keeps zero joints
    (that is the state ``Simulation.__init__`` forward-passes over — straight
    legs, no interpenetration), so do not "fix" this by copying HOME into
    qpos0 unless you re-run the world build."""
    assert world.nkey == 1
    key = np.array(world.key_qpos[0])
    assert key.shape == (world.nq,)

    ref = None
    for i, s in enumerate(S.duck_qpos_slices(world)):
        # Root pose comes from the formation.
        x, y, _ = S.formation()[i]
        assert key[s.start] == pytest.approx(x, abs=1e-6)
        assert key[s.start + 1] == pytest.approx(y, abs=1e-6)
        assert key[s.start + 2] == pytest.approx(S.DUCK_SPAWN_Z, abs=1e-4)
        # Joints: the same HOME pose for every duck (the 180 deg yaw is at the
        # root, so team 1 is not mirrored in the joints).
        q = key[s.start + 7:s.stop]
        if ref is None:
            ref = q
        assert np.allclose(q, ref), f"duck {i} spawns with a different joint pose"
    # HOME, not zeros: left_hip_roll = -0.0873, left_hip_pitch = -0.4579.
    assert abs(ref[1]) == pytest.approx(0.0873, abs=1e-3)
    assert abs(ref[2]) == pytest.approx(0.4579, abs=1e-3)


# ── GPU: the real world steps cleanly ────────────────────────────────────────
@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_world_steps_without_overflow_or_nan():
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg

    n = 4
    cfg = SceneCfg(
        num_envs=n,
        env_spacing=8.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=S.scene_entity_cfgs(),
    )
    scene = Scene(cfg, device="cuda:0")
    mj_model = scene.compile()
    S.verify_world(mj_model)
    sim = Simulation(
        num_envs=n,
        cfg=SimulationCfg(
            njmax=P.NJMAX,
            nconmax=P.NCONMAX,
            mujoco=MujocoCfg(timestep=P.SIM_TIMESTEP),
        ),
        model=mj_model,
        device="cuda:0",
    )
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    scene.reset(slice(None))

    # The ball's rolling resistance must be present in EVERY world after the
    # per-world field expansion (this is the array BAM writes into).
    sim.expand_model_fields(("dof_frictionloss", "dof_damping"))
    fl = sim.model.dof_frictionloss
    assert fl.stride(0) != 0, "dof_frictionloss was not expanded per world"
    bd = S.ball_dof_slice(sim.mj_model)
    ball_fl = fl[:, bd.start:bd.stop]
    assert torch.allclose(ball_fl, torch.full_like(ball_fl, P.BALL_ROLL_LOSS))

    for _ in range(250):
        sim.step()
    torch.cuda.synchronize()
    assert not torch.isnan(sim.data.qpos).any()
    assert not torch.isnan(sim.data.qvel).any()
    # Ducks may fall over (no policy yet) but must stay inside the world.
    assert sim.data.qpos.abs().max().item() < 50.0


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_bam_never_touches_the_ball_dofs():
    """BAM writes friction per env, but only into ITS OWN ``dof_ids``.

    The invariant that matters: no duck actuator's target dof is a ball dof.
    (If one were, joint-friction DR would randomise the ball's rolling
    resistance every reset and the tuned 2.15e-4 would be gone.)
    """
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    cfg = SceneCfg(
        num_envs=1,
        env_spacing=8.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=S.scene_entity_cfgs(),
    )
    scene = Scene(cfg, device="cpu")
    model = scene.compile()
    ball_dofs = set(range(*S.ball_dof_slice(model).indices(model.nv)))
    ball_qpos = set(range(*S.ball_qpos_slice(model).indices(model.nq)))

    # Every actuator is a joint transmission; collect the dof each one drives.
    targets = set()
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        targets.add(int(model.jnt_dofadr[joint]))
    assert not (targets & ball_dofs), "an actuator targets a ball dof"
    assert len(targets) == 22 * 14
    # And the ball's qpos block is never written by a servo either.
    assert not (targets & ball_qpos)
