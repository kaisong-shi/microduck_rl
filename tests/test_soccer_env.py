"""Soccer env + lane wrapper contract (P4-T1/T3/T4).

The single most dangerous bug class here is *silent mis-routing*: if the flat
action vector is permuted, every duck executes its neighbour's action. Nothing
raises, training still "works", and the policy learns a smeared average — this
is what `test_actions_are_routed_to_the_right_duck` exists to prevent.

GPU-only tests are skipped without CUDA.
"""

import math

import pytest
import torch

from mjlab_microduck.soccer import env as ENV
from mjlab_microduck.soccer import obs as O
from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer import scene as S

N = P.NUM_DUCKS
N_TEAM = P.NUM_DUCKS_PER_TEAM


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")


@pytest.fixture(scope="module")
def env():
    if not _has_cuda():
        pytest.skip("needs CUDA")
    e = ENV.SoccerEnv(ENV.SoccerEnvCfg(num_worlds=2, device="cuda:0", seed=0))
    yield e
    e.close()


# ── Pure logic (CPU) ────────────────────────────────────────────────────────
def test_episode_length_math():
    cfg = ENV.SoccerEnvCfg(num_worlds=1, episode_s=30.0)
    assert cfg.max_episode_length == int(30.0 / (0.005 * 4))
    assert cfg.num_agents == N


def test_lane_world_mapping_is_documented_and_consistent():
    """lane = world * 22 + duck (the layout the wrapper relies on)."""
    assert ENV.N == 22
    for world in (0, 1, 7):
        for duck in (0, 21):
            lane = world * N + duck
            assert lane // N == world and lane % N == duck


# ── Env smoke ───────────────────────────────────────────────────────────────
@needs_cuda
def test_shapes_and_finiteness(env):
    obs = env.reset()
    assert obs["actor"].shape == (2 * N, O.OBS_DIM)
    assert obs["critic"].shape == (2 * N, O.OBS_DIM + env.cfg.critic_extra_dim)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(obs["critic"]).all()

    a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(5):
        obs, rew, done, extras = env.step(a)
    assert rew.shape == (2 * N,)
    assert done.shape == (2 * N,)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(rew).all()
    assert set(extras["Episode_Reward"]) >= {"touch", "ball_goal_progress",
                                             "goal_event", "teammate_spacing",
                                             "fallen", "no_progress"}


@needs_cuda
def test_reset_puts_ducks_in_the_formation_and_the_ball_in_the_patch(env):
    env.reset()
    pos = torch.stack([d.data.root_link_pos_w[0] for d in env.ducks])   # (22, 3)
    for i, (x, y, _) in enumerate(S.formation()):
        assert pos[i, 0].item() == pytest.approx(x, abs=ENV.EV.DUCK_SPAWN_JITTER + 1e-3)
        assert pos[i, 1].item() == pytest.approx(y, abs=ENV.EV.DUCK_SPAWN_JITTER + 1e-3)
        assert pos[i, 2].item() == pytest.approx(S.DUCK_SPAWN_Z, abs=2e-2)
    ball = env.ball.data.root_link_pos_w[0]
    assert ball[:2].norm().item() <= ENV.EV.BALL_SPAWN_RADIUS + 1e-3
    assert ball[2].item() == pytest.approx(P.BALL_RADIUS, abs=1e-2)


@needs_cuda
def test_home_pose_is_the_ctrl_zero_action(env):
    """action = 0 must mean "hold HOME", not "go to 0 rad"."""
    env.reset()
    a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(10):
        env.step(a)
    q = env.ducks[0].data.joint_pos[0]
    assert torch.allclose(q, env.default_joint_pos, atol=0.25), (
        f"zero action drifted from HOME: max err {(q - env.default_joint_pos).abs().max():.3f}"
    )


@needs_cuda
def test_actions_are_routed_to_the_right_duck(env):
    """THE mis-routing test: a pure +1 action on duck 0 must move duck 0's
    joints and must NOT move the other 21.

    Differential, not absolute: ducks settle under gravity even with a zero
    action (~0.08 rad of joint drift in 0.3 s on all 22 at once), so the
    comparison must be world 0 (action) vs world 1 (no action). An absolute
    threshold would either pass a permuted layout or fail on the settling drift.
    """
    env.reset()
    a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    a[0] = 1.0                       # world 0, duck 0. World 1 = control.
    before = torch.stack([d.data.joint_pos.clone() for d in env.ducks])   # (22, E, 14)
    for _ in range(15):
        env.step(a)
    after = torch.stack([d.data.joint_pos.clone() for d in env.ducks])
    delta = (after - before).abs().amax(dim=-1)      # (22, E) per duck per world

    moved_0 = delta[0, 0].item() - delta[0, 1].item()
    assert moved_0 > 0.05, f"duck 0 ignored its own action (extra motion {moved_0:.4f})"

    others = (delta[1:, 0] - delta[1:, 1]).abs()
    assert others.max() < 0.02, (
        f"an action aimed at duck 0 changed another duck's motion "
        f"(max {others.max():.4f}) — the flat action vector is permuted"
    )


@needs_cuda
def test_episode_times_out_and_resets():
    cfg = ENV.SoccerEnvCfg(num_worlds=2, episode_s=0.4, device="cuda:0", seed=1)
    e = ENV.SoccerEnv(cfg)
    assert e.max_episode_length == int(0.4 / 0.02)
    a = torch.zeros(e.num_envs, e.num_actions, device=e.device)
    seen_done = False
    for _ in range(e.max_episode_length + 2):
        _, _, done, extras = e.step(a)
        if done.any():
            seen_done = True
            assert extras["truncated"].all(), "short episode should end by timeout"
            assert not extras["terminated"].any()
            break
    assert seen_done, "episode never timed out"
    # Lanes reset: every lane's episode counter restarts.
    assert e.episode_length_buf.max().item() <= 1
    e.close()


@needs_cuda
def test_random_policy_produces_non_degenerate_rewards(env):
    """The reward stack must actually be alive: a flailing policy falls over and
    touches the ball sometimes. All-zero rewards would train nothing."""
    env.reset()
    torch.manual_seed(0)
    a = torch.randn(env.num_envs, env.num_actions, device=env.device).clamp(-1, 1)
    acc = {}
    for _ in range(40):
        _, rew, _, extras = env.step(a)
        for k, v in extras["Episode_Reward"].items():
            acc.setdefault(k, []).append(v)
    means = {k: sum(v) / len(v) for k, v in acc.items()}
    # Random actions knock the ducks over -> the fall penalty must fire.
    assert means["fallen"] < 0, f"no fall penalty with random actions: {means}"
    # And every penalty stays non-positive (the wandb audit rule, in-sim).
    for k in ("teammate_spacing", "fallen", "no_progress"):
        assert means[k] <= 1e-6, f"{k} averaged {means[k]:.4f} > 0"
    assert torch.isfinite(rew).all()


# ── Lane wrapper ────────────────────────────────────────────────────────────
@needs_cuda
def test_vecenv_lane_contract(env):
    from mjlab_microduck.soccer.vecenv import SoccerVecEnv

    v = SoccerVecEnv(env)
    assert v.num_envs == env.num_worlds * N
    assert v.num_actions == N_JOINTS_EXPECTED
    assert v.max_episode_length == env.max_episode_length
    assert v.obs_groups == {"actor": ["actor"], "critic": ["critic"]}

    obs = v.get_observations()
    assert obs["actor"].shape == (v.num_envs, O.OBS_DIM)
    assert obs.batch_size == torch.Size([v.num_envs])

    a = torch.zeros(v.num_envs, v.num_actions, device=v.device)
    obs, rew, dones, extras = v.step(a)
    assert rew.shape == (v.num_envs,)
    assert dones.shape == (v.num_envs,)
    assert dones.dtype == torch.long
    assert extras["time_outs"].shape == (v.num_envs,)

    # episode_length_buf: repeated for the getter, one authoritative copy per
    # world for the setter (rsl_rl writes it with randint_like).
    elb = v.episode_length_buf
    assert elb.shape == (v.num_envs,)
    assert torch.equal(elb[0:N], elb[:N])
    assert elb[0].item() == elb[N - 1].item()
    v.episode_length_buf = torch.arange(v.num_envs, device=v.device)
    assert env.episode_length_buf[0].item() == 0
    assert env.episode_length_buf[1].item() == N
    assert torch.equal(v.episode_length_buf[:N], torch.zeros(N, dtype=torch.long,
                                                             device=v.device))


N_JOINTS_EXPECTED = 14


# ── Small-sided stages (S0 1v1 ... S3 6v6) ──────────────────────────────────
@pytest.mark.parametrize("n_team", [1, 2, 3, 4, 5, 6, 11])
def test_formation_scales_with_team_size(n_team):
    poses = S.formation(n_team)
    assert len(poses) == 2 * n_team
    half_l, half_w = P.PITCH_L / 2, P.PITCH_W / 2
    for i, (x, y, yaw) in enumerate(poses):
        assert abs(x) < half_l - 0.2 and abs(y) < half_w - 0.2
        if i < n_team:
            assert x < 0 and yaw == pytest.approx(0.0), "team 0 must be in its own half"
        else:
            assert x > 0 and yaw == pytest.approx(math.pi)
    # Mirror pairs and a sane minimum spacing.
    for i in range(n_team):
        x0, y0, _ = poses[i]
        x1, y1, _ = poses[i + n_team]
        assert (x1, y1) == pytest.approx((-x0, -y0))
    xy = torch.tensor([(x, y) for x, y, _ in poses])
    d = torch.cdist(xy, xy) + torch.eye(2 * n_team) * 10
    assert d.min().item() > 0.35, f"spawns only {d.min():.2f} m apart"


def test_small_sided_duck_count_and_budgets():
    for n_team in (1, 2, 6, 11):
        cfg = ENV.SoccerEnvCfg(num_worlds=1, n_team=n_team)
        assert cfg.n_ducks == 2 * n_team
        assert cfg.num_agents == 2 * n_team
    # Contact budget is scaled: 1 v 1 must not reserve a 22-duck budget.
    small = ENV.SoccerEnvCfg(num_worlds=1, n_team=1)
    full = ENV.SoccerEnvCfg(num_worlds=1, n_team=11)
    assert max(400, int(round(P.NJMAX * small.n_ducks / P.NUM_DUCKS))) < P.NJMAX
    assert max(300, int(round(P.NCONMAX * full.n_ducks / P.NUM_DUCKS))) == P.NCONMAX


@needs_cuda
def test_one_v_one_env_smoke():
    """S0: 1 duck per team, so no teammates and no keeper — the observation
    must still be exactly 97 dims with zero-padded mate slots."""
    e = ENV.SoccerEnv(ENV.SoccerEnvCfg(num_worlds=2, n_team=1, device="cuda:0", seed=2))
    try:
        assert e.n_ducks == 2
        assert e.num_envs == 4
        assert len(e.ducks) == 2
        obs = e.reset()
        assert obs["actor"].shape == (4, O.OBS_DIM)
        sl = O.SLICES
        # No teammates in 1 v 1 -> every mate slot is zero.
        assert torch.all(obs["actor"][:, sl["mates_rel_xy_b"]] == 0)
        assert torch.all(obs["actor"][:, sl["mates_centroid_xy_b"]] == 0)
        # But the single opponent IS visible.
        assert obs["actor"][:, sl["opps_rel_xy_b"]][:, :2].abs().sum() > 0
        a = torch.zeros(e.num_envs, e.num_actions, device=e.device)
        for _ in range(5):
            obs, rew, done, extras = e.step(a)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(rew).all()
        # The spacing penalty has nothing to complain about with one duck/team.
        assert extras["Episode_Reward"]["teammate_spacing"] == 0.0
    finally:
        e.close()


@needs_cuda
def test_spawn_uses_the_matching_small_sided_formation():
    """Regression: `spawn_poses` used to ignore the team size, so a 1 v 1 world
    spawned its two ducks on the 11-a-side keeper line (3 m from the ball) and
    every episode stalemated before contact. The spawn distance to the ball is
    what makes S0 learnable at all."""
    e = ENV.SoccerEnv(ENV.SoccerEnvCfg(num_worlds=2, n_team=1, device="cuda:0", seed=3))
    try:
        e.reset()
        ball = e.ball.data.root_link_pos_w[:, :2]
        duck = torch.stack([d.data.root_link_pos_w[:, :2] for d in e.ducks], dim=1)
        d = (duck - ball[:, None, :]).norm(dim=-1)
        # 1 v 1 spawns are +-0.8 m from the centre; allow the ball patch + jitter.
        assert d.max().item() < 1.3, f"closest spawn is {d.min():.2f} m, farthest {d.max():.2f} m"
        # And they are on opposite sides of the centre spot.
        assert duck[0, 0, 0].item() * duck[0, 1, 0].item() < 0
    finally:
        e.close()
