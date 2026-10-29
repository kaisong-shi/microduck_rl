"""Per-duck observation contract (P3-T3/T4).

The symmetry test is the important one. It is the ONLY check that the body-frame
canonicalisation is right, and it is cheap: take any state, rotate the whole
world 180° about the centre spot, swap the teams, and every duck's 97-dim
observation must come out bit-identical. If it fails, the policy is learning two
different tasks wearing one set of weights (one per team) — which trains, looks
plausible on video, and then behaves like a different animal after the 180° flip.
"""

import math

import pytest
import torch

from mjlab_microduck.soccer import obs as O
from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer import scene as S

N_TEAM = P.NUM_DUCKS_PER_TEAM
N = P.NUM_DUCKS


def _state(e: int = 2, seed: int = 0, n_team: int = N_TEAM, n: int = None):
    """A random but geometry-plausible world state."""
    n = n or 2 * n_team
    g = torch.Generator().manual_seed(seed)
    root_pos = torch.zeros(e, n, 3)
    root_pos[..., 0].uniform_(-P.PITCH_L / 2 + 0.3, P.PITCH_L / 2 - 0.3, generator=g)
    root_pos[..., 1].uniform_(-P.PITCH_W / 2 + 0.3, P.PITCH_W / 2 - 0.3, generator=g)
    root_pos[..., 2].uniform_(0.08, 0.2, generator=g)
    yaw = torch.zeros(e, n).uniform_(-math.pi, math.pi, generator=g)
    quat = torch.stack(
        (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)),
        dim=-1,
    )
    ball_pos = torch.zeros(e, 3)
    ball_pos[:, 0].uniform_(-1.0, 1.0, generator=g)
    ball_pos[:, 1].uniform_(-1.0, 1.0, generator=g)
    ball_pos[:, 2] = P.BALL_RADIUS
    return dict(
        root_pos_w=root_pos,
        root_quat_w=quat,
        root_lin_vel_w=torch.zeros(e, n, 3).uniform_(-0.5, 0.5, generator=g),
        joint_pos_rel=torch.zeros(e, n, O.N_JOINTS).uniform_(-0.5, 0.5, generator=g),
        joint_vel_rel=torch.zeros(e, n, O.N_JOINTS).uniform_(-4.0, 4.0, generator=g),
        base_ang_vel_b=torch.zeros(e, n, 3).uniform_(-2.0, 2.0, generator=g),
        projected_gravity_b=torch.zeros(e, n, 3).uniform_(-1.0, 1.0, generator=g),
        last_action=torch.zeros(e, n, O.N_JOINTS).uniform_(-1.0, 1.0, generator=g),
        ball_pos_w=ball_pos,
        ball_vel_w=torch.zeros(e, 3).uniform_(-2.0, 2.0, generator=g),
        own_goal_pos_w=_own_goals(n, n_team),
        role=O.role_onehot(n, n_team),
        time_remaining=torch.zeros(e).uniform_(0.0, 1.0, generator=g),
        goal_diff=torch.zeros(e).uniform_(-3.0, 3.0, generator=g),
        last_touch=torch.zeros(e).uniform_(0.0, 1.0, generator=g),
        n_team=n_team,
    )


def _own_goals(n: int, n_team: int) -> torch.Tensor:
    """Own goal centre per duck, in world coordinates."""
    team = O.team_ids(n, n_team)
    sign = O.team_attack_sign(n, n_team)
    out = torch.zeros(n, 3)
    out[:, 0] = -sign * P.PITCH_L / 2   # team 0 defends -x
    out[:, 2] = P.GOAL_H / 2
    del team
    return out


def _rotate_180(state: dict, n_team: int) -> dict:
    """Rotate the world 180 deg about the centre spot and swap the teams.

    Mapping: original duck ``i`` <-> rotated duck ``sigma(i) = (i + n_team) % n``,
    and ``sigma(i)`` is on the OTHER team. Everything that is a world vector gets
    flipped in xy; body-frame quantities (ang vel, projected gravity, joint
    state) are untouched, because the duck was rotated with the world.
    """
    n = state["root_pos_w"].shape[1]
    flip = torch.ones(n, 3)
    flip[:, :2] = -1.0

    s = dict(state)
    order = torch.cat((torch.arange(n_team, n), torch.arange(0, n_team)))

    # Reorder AND rotate in one step (doing them in the wrong order silently
    # discards the rotation - it cost one debugging round).
    s["root_pos_w"] = (state["root_pos_w"] * flip)[:, order, :]
    s["root_lin_vel_w"] = (state["root_lin_vel_w"] * flip)[:, order, :]
    q = state["root_quat_w"][:, order, :]
    # qz(pi) (x) q -> (w, x, y, z) = (-z, -y, x, w).  Component order is
    # (w, x, y, z), so -z is index 3 and w is index 0 — getting that mapping
    # wrong silently produces a rotation about y instead of z.
    s["root_quat_w"] = torch.stack((-q[..., 3], -q[..., 2], q[..., 1], q[..., 0]), dim=-1)
    flip_ball = flip.new_tensor([-1.0, -1.0, 1.0])
    s["ball_pos_w"] = state["ball_pos_w"] * flip_ball
    s["ball_vel_w"] = state["ball_vel_w"] * flip_ball

    for key in ("joint_pos_rel", "joint_vel_rel", "base_ang_vel_b",
                "projected_gravity_b", "last_action", "role"):
        v = state[key]
        s[key] = v[:, order, :] if v.dim() == 3 else v[order, :]

    # Own goal is DERIVED from the duck's index (= its team), so it is NOT
    # reordered: rotated duck j is on team 1 and therefore defends +x, which is
    # exactly the 180 deg image of what duck i (= order[j]) defended.
    s["own_goal_pos_w"] = _own_goals(n, n_team)
    # Contest state mirrors.
    s["goal_diff"] = -state["goal_diff"]
    s["last_touch"] = 1.0 - state["last_touch"]
    return s


# ── Contract ─────────────────────────────────────────────────────────────────
def test_obs_dim_and_layout():
    assert O.OBS_DIM == 97
    slices = O.obs_slices()
    assert slices["joint_pos_rel"].start == 0
    assert slices["own_goal_rel_pos_b"].stop == 97
    # Blocks tile the vector with no gap or overlap.
    covered = []
    for sl in slices.values():
        covered.extend(range(sl.start, sl.stop))
    assert covered == list(range(97))
    # Order is append-only: the base block must stay at the front (a warm start
    # from any earlier policy depends on it).
    names = list(slices)
    assert names[:5] == ["joint_pos_rel", "joint_vel_rel", "base_ang_vel_b",
                         "projected_gravity_b", "last_action"]


def test_build_obs_shape_and_finiteness():
    st = _state()
    obs = O.build_obs(**st)
    assert obs.shape == (2, N, O.OBS_DIM)
    assert torch.isfinite(obs).all()


def test_large_state_stays_finite():
    """NaN guard: absurd values must be scrubbed, not propagated."""
    st = _state()
    st["root_pos_w"][0, 0, 0] = float("nan")
    st["ball_vel_w"][0] = float("inf")
    obs = O.build_obs(**st)
    assert torch.isfinite(obs).all()


def test_base_block_is_passed_through_untouched():
    st = _state()
    obs = O.build_obs(**st)
    sl = O.SLICES
    assert torch.equal(obs[..., sl["joint_pos_rel"]], st["joint_pos_rel"])
    assert torch.equal(obs[..., sl["joint_vel_rel"]], st["joint_vel_rel"])
    assert torch.equal(obs[..., sl["last_action"]], st["last_action"])
    assert torch.equal(obs[..., sl["base_ang_vel_b"]], st["base_ang_vel_b"])


def test_ball_is_in_the_duck_body_frame():
    """A ball 1 m straight ahead of a duck facing +y must read (0, 1, dz)."""
    st = _state(e=1)
    n = N
    st["root_pos_w"][:] = 0.0
    yaw = torch.full((1, n), math.pi / 2)   # facing +y
    st["root_quat_w"] = torch.stack(
        (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1
    )
    st["root_lin_vel_w"][:] = 0.0        # ball velocity is reported RELATIVE to the duck
    st["ball_pos_w"][0] = torch.tensor([0.0, 1.0, 0.1])
    st["ball_vel_w"][0] = torch.tensor([0.0, 2.0, 0.0])
    obs = O.build_obs(**st)
    ball_rel = obs[0, 0, O.SLICES["ball_rel_pos_b"]]
    assert ball_rel[:2] == pytest.approx(torch.tensor([1.0, 0.0]), abs=1e-5)
    ball_vel = obs[0, 0, O.SLICES["ball_rel_vel_b"]]
    assert ball_vel[:2] == pytest.approx(torch.tensor([2.0, 0.0]), abs=1e-5)


def test_ball_distance_block_matches_the_xy_distance():
    st = _state()
    obs = O.build_obs(**st)
    d = obs[..., O.SLICES["ball_distance"]].squeeze(-1)
    want = (st["ball_pos_w"].unsqueeze(1) - st["root_pos_w"])[..., :2].norm(dim=-1)
    assert torch.allclose(d, want / O.MAX_BALL_DIST, atol=1e-5)
    assert d.max() <= 1.0 + 1e-6


def test_role_onehot_and_self_position():
    st = _state()
    obs = O.build_obs(**st)
    role = obs[0, :, O.SLICES["role_onehot"]]
    assert torch.allclose(role.sum(-1), torch.ones(N))
    # The 11 v 11 template: exactly one GK, four DEF, four MID, two FWD per team.
    for t in (0, 1):
        block = role[t * N_TEAM:(t + 1) * N_TEAM]
        assert block[:, O.ROLE_NAMES.index("GK")].sum() == 1
        assert block[:, O.ROLE_NAMES.index("DEF")].sum() == 4
        assert block[:, O.ROLE_NAMES.index("MID")].sum() == 4
        assert block[:, O.ROLE_NAMES.index("FWD")].sum() == 2
    self_pos = obs[0, :, O.SLICES["self_pos_xy"]]
    assert self_pos.abs().max() <= 1.5


def test_nearest_three_are_actually_the_nearest():
    st = _state(e=1, seed=7)
    obs = O.build_obs(**st)
    pos = st["root_pos_w"][0, :, :2]
    mates = obs[0, 0, O.SLICES["mates_rel_xy_b"]].reshape(3, 2)
    # Reverse the body rotation to compare in world space: distances are
    # rotation invariant, so just check the magnitude set.
    got = mates.norm(dim=-1).sort().values
    d = (pos - pos[0]).norm(dim=-1)
    mate_ids = [i for i in range(N) if S.team_of(i) == S.team_of(0) and i != 0]
    want = d[mate_ids].sort().values[:3]
    assert torch.allclose(got, want, atol=1e-5)


def test_small_sided_teams_zero_pad_and_stay_finite():
    """1 v 1: no mates, one opponent -> mate slots all zero, layout unchanged."""
    st = _state(e=1, n_team=1, n=2)
    obs = O.build_obs(**st)
    assert obs.shape == (1, 2, O.OBS_DIM)
    assert torch.isfinite(obs).all()
    assert torch.all(obs[0, 0, O.SLICES["mates_rel_xy_b"]] == 0)
    assert torch.all(obs[0, 0, O.SLICES["mates_rel_vel_xy_b"]] == 0)
    assert torch.all(obs[0, 0, O.SLICES["mates_centroid_xy_b"]] == 0)
    # The single opponent is reported (non-zero unless exactly overlapping).
    assert obs[0, 0, O.SLICES["opps_rel_xy_b"]][:2].abs().sum() > 0


@pytest.mark.parametrize("n_team", [1, 2, 3, 4, 6, 11])
def test_roles_adapt_to_team_size(n_team):
    roles = O.roles_for_team(n_team)
    assert len(roles) == n_team
    assert set(roles) <= set(O.ROLE_NAMES)
    oh = O.role_onehot(2 * n_team, n_team)
    assert torch.allclose(oh.sum(-1), torch.ones(2 * n_team))


# ── The symmetry contract ────────────────────────────────────────────────────
def test_observation_is_invariant_to_a_180_rotation_plus_team_swap():
    """THE test for the canonical frame (P3-T3).

    Rotate the world 180° about the centre spot, swap the two teams, and every
    duck's observation must be IDENTICAL to its new counterpart's. Otherwise the
    shared policy is silently learning two mirrored tasks.
    """
    st = _state(e=3, seed=11)
    obs = O.build_obs(**st)

    rot = _rotate_180(st, N_TEAM)
    obs_rot = O.build_obs(**rot)

    # duck i in the original == duck (i + N_TEAM) % N in the rotated world.
    order = torch.cat((torch.arange(N_TEAM, N), torch.arange(0, N_TEAM)))
    err = (obs - obs_rot[:, order]).abs().max().item()
    assert err < 1e-4, f"observation changed under 180 deg rotation + team swap (max err {err:.2e})"


def test_symmetry_holds_for_small_sided_worlds():
    st = _state(e=2, seed=3, n_team=2, n=4)
    obs = O.build_obs(**st)
    rot = _rotate_180(st, 2)
    obs_rot = O.build_obs(**rot)
    order = torch.cat((torch.arange(2, 4), torch.arange(0, 2)))
    err = (obs - obs_rot[:, order]).abs().max().item()
    assert err < 1e-4, f"max err {err:.2e}"
