"""Reward term contract (P3-T5/T6).

Three things are locked here, in the order of how much they have cost this repo:
  1. **Sign.** Every penalty returns <= 0 (so its weight is positive). A single
     sign slip is not a small bug: a negative weight on a self-negating penalty
     double-negates into a reward for the violation, and the policy farms it
     until the run is worthless.
  2. **Unfarmable progress.** The dense task term must pay exactly ZERO when the
     ball does not move. A discounted potential form pays more the FURTHER the
     ball is from the goal — an incentive to keep the ball in your own half.
  3. **Team relativism.** The same world event must pay one team and charge the
     other, otherwise "shoot anywhere" is optimal.
"""

import math

import pytest
import torch

from mjlab_microduck.soccer import obs as O
from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer import rewards as R
from mjlab_microduck.soccer import scene as S

N_TEAM = P.NUM_DUCKS_PER_TEAM
N = P.NUM_DUCKS
SIGN = O.team_attack_sign(N, N_TEAM)
W = R.BALL_APPROACH_WEIGHT


# ── Goal detection ───────────────────────────────────────────────────────────
def test_ball_goal_side():
    in_pos = torch.tensor([P.PITCH_L / 2 + 0.05, 0.0, P.BALL_RADIUS])
    in_neg = torch.tensor([-P.PITCH_L / 2 - 0.05, 0.0, P.BALL_RADIUS])
    centre = torch.zeros(3)
    wide = torch.tensor([P.PITCH_L / 2 + 0.05, P.GOAL_W, P.BALL_RADIUS])
    high = torch.tensor([P.PITCH_L / 2 + 0.05, 0.0, P.GOAL_H + 0.05])
    side = R.ball_goal_side(torch.stack((in_pos, in_neg, centre, wide, high)))
    assert side.tolist() == [1, -1, 0, 0, 0]


def test_goal_event_is_edge_triggered():
    prev = torch.tensor([0, 0, 1, 1, -1])
    cur = torch.tensor([1, 0, 1, 0, -1])
    ev = R.goal_events(prev, cur)
    # Only 0->1 fires; 0->0, 1->1, 1->0 and -1->-1 do not.
    assert ev.tolist() == [1, 0, 0, 0, 0]


def test_goal_reward_is_team_relative():
    ev = torch.tensor([1, -1, 0])
    r = R.goal_event_reward(ev, SIGN)          # (3, 22)
    assert r.shape == (3, N)
    # Ball in the +x goal: team 0 (attacks +x) is paid, team 1 is charged.
    assert (r[0, :N_TEAM] > 0).all()
    assert (r[0, N_TEAM:] < 0).all()
    # Ball in the -x goal: mirrored.
    assert (r[1, :N_TEAM] < 0).all()
    assert (r[1, N_TEAM:] > 0).all()
    assert (r[2] == 0).all()


# ── Ball progress: the unfarmable dense term ─────────────────────────────────
def test_progress_pays_forward_and_zero_when_static():
    target = torch.tensor([P.PITCH_L / 2, 0.0])
    prev = torch.tensor([[-1.0, 0.0]])         # 1 m back, on the axis
    toward = torch.tensor([[-0.9, 0.0]])       # 10 cm closer
    away = torch.tensor([[-1.1, 0.0]])         # 10 cm further
    static = prev.clone()

    fwd = R.ball_goal_progress(toward, prev, target, SIGN)
    back = R.ball_goal_progress(away, prev, target, SIGN)
    still = R.ball_goal_progress(static, prev, target, SIGN)

    assert fwd.shape == (1, N)
    # Team 0 attacks +x -> moving the ball +x is progress; team 1 the opposite.
    assert (fwd[0, :N_TEAM] > 0).all()
    assert (fwd[0, N_TEAM:] < 0).all()
    assert (back[0, :N_TEAM] < 0).all()
    assert (back[0, N_TEAM:] > 0).all()
    # THE anti-jackpot assertion: no motion -> exactly zero, for everybody.
    assert torch.all(still == 0)


def test_progress_cannot_be_farmed_by_parking_far_from_the_goal():
    """A discounted potential would pay d*(1-gamma) per step. Verify we don't."""
    target = torch.tensor([P.PITCH_L / 2, 0.0])
    far_prev = torch.tensor([[-3.0, 0.0]])
    far_now = torch.tensor([[-3.0, 0.0]])
    near_prev = torch.tensor([[0.0, 0.0]])
    near_now = torch.tensor([[0.0, 0.0]])
    assert torch.all(R.ball_goal_progress(far_now, far_prev, target, SIGN) == 0)
    assert torch.all(R.ball_goal_progress(near_now, near_prev, target, SIGN) == 0)


def test_progress_round_trip_cancels():
    """Going 20 cm forward then 20 cm back must net ~0 (potential shaping)."""
    target = torch.tensor([P.PITCH_L / 2, 0.0])
    a = torch.tensor([[-1.0, 0.3]])
    b = torch.tensor([[-0.8, 0.3]])
    fwd = R.ball_goal_progress(b, a, target, SIGN)
    back = R.ball_goal_progress(a, b, target, SIGN)
    assert torch.allclose(fwd + back, torch.zeros_like(fwd), atol=1e-6)


def test_progress_is_bounded_per_step():
    target = torch.tensor([P.PITCH_L / 2, 0.0])
    a = torch.tensor([[-3.4, 2.2]])
    b = torch.tensor([[3.4, -2.2]])
    r = R.ball_goal_progress(b, a, target, SIGN)
    assert r.abs().max() <= 1.0


# ── Touch ────────────────────────────────────────────────────────────────────
def test_touch_reward_respects_cooldown():
    touched = torch.ones(1, N, dtype=torch.bool)
    just_now = torch.zeros(1, N)               # touched a moment ago
    long_ago = torch.full((1, N), 10.0)
    assert torch.all(R.touch_reward(touched, just_now) == 0)
    assert torch.allclose(R.touch_reward(touched, long_ago),
                          torch.full((1, N), R.TOUCH_BONUS))
    # Not touching -> nothing, even when eligible.
    assert torch.all(R.touch_reward(~touched, long_ago) == 0)


def test_touch_reward_is_capped_by_the_cooldown():
    """A duck touching the ball every step cannot out-earn one clean touch/5 s."""
    one_touch = R.touch_reward(torch.ones(1, N, dtype=torch.bool),
                               torch.full((1, N), 99.0)).sum().item()
    many = 0.0
    since = torch.full((1, N), 99.0)
    dt = 0.02
    for _ in range(int(5.0 / dt)):             # 5 s of continuous contact
        r = R.touch_reward(torch.ones(1, N, dtype=torch.bool), since)
        many += r.sum().item()
        since = torch.where(r > 0, torch.zeros_like(since), since + dt)
    assert many == pytest.approx(one_touch)   # exactly one credited touch per 5 s


# ── Penalties: every one must be <= 0 ───────────────────────────────────────
def test_teammate_spacing_penalty_is_zero_when_spread_out():
    pos = torch.zeros(1, N, 3)
    for i, (x, y, _) in enumerate(S.formation()):
        pos[0, i, 0], pos[0, i, 1] = x, y
    pen = R.teammate_spacing_penalty(pos, SIGN)
    assert torch.all(pen == 0), "the kickoff formation must not be taxed"


def test_teammate_spacing_penalty_punishes_crowding():
    pos = torch.zeros(1, 4, 3)
    pos[0, 0, :2] = torch.tensor([0.0, 0.0])
    pos[0, 1, :2] = torch.tensor([0.10, 0.0])    # 10 cm away, same team
    pos[0, 2, :2] = torch.tensor([2.0, 0.0])
    pos[0, 3, :2] = torch.tensor([2.0, 0.5])
    sign = O.team_attack_sign(4, 2)
    pen = R.teammate_spacing_penalty(pos, sign)
    assert (pen <= 0).all()
    assert pen[0, 0] < 0 and pen[0, 1] < 0
    assert pen[0, 2] == 0 and pen[0, 3] == 0
    # Two ducks 10 cm apart on a 25 cm threshold -> 15 cm of overlap each.
    assert pen[0, 0].item() == pytest.approx(-0.15, abs=1e-5)


def test_teammate_spacing_ignores_opponents():
    """Opponents SHOULD be close (that is the contest) — no penalty for that."""
    pos = torch.zeros(1, 2, 3)
    pos[0, 0, :2] = torch.tensor([0.0, 0.0])
    pos[0, 1, :2] = torch.tensor([0.05, 0.0])
    pen = R.teammate_spacing_penalty(pos, O.team_attack_sign(2, 1))
    assert torch.all(pen == 0)


@pytest.mark.parametrize("tilt_deg,expect_zero", [(0, True), (30, True), (59, True),
                                                  (61, False), (90, False)])
def test_fallen_penalty_onset(tilt_deg, expect_zero):
    g = torch.tensor([0.0, math.sin(math.radians(tilt_deg)), -math.cos(math.radians(tilt_deg))])
    pen = R.fallen_penalty(g.view(1, 1, 3))
    assert (pen <= 0).all()
    if expect_zero:
        assert pen.item() == pytest.approx(0.0, abs=1e-6)
    else:
        assert pen.item() < 0


def test_fallen_penalty_is_bounded():
    flat_on_back = torch.tensor([[[0.0, 1.0, 0.0]]])   # 90 deg over
    assert R.fallen_penalty(flat_on_back).item() == pytest.approx(-1.0, abs=1e-6)
    upside_down = torch.tensor([[[0.0, 0.0, 1.0]]])    # fully inverted
    assert R.fallen_penalty(upside_down).item() <= -1.0


def test_no_progress_penalty_fires_only_when_stuck():
    ball = torch.tensor([[0.0, 0.0, P.BALL_RADIUS]])
    start = torch.tensor([[0.0, 0.0]])
    # Window not elapsed yet.
    assert torch.all(R.no_progress_penalty(ball, start, torch.tensor([1.0]),
                                           torch.tensor([False]), N) == 0)
    # Elapsed, ball moved enough -> no penalty.
    moved = torch.tensor([[0.30, 0.0, P.BALL_RADIUS]])
    assert torch.all(R.no_progress_penalty(moved, start, torch.tensor([6.0]),
                                           torch.tensor([False]), N) == 0)
    # Elapsed, ball static, but someone touched it -> not a stalemate.
    assert torch.all(R.no_progress_penalty(ball, start, torch.tensor([6.0]),
                                           torch.tensor([True]), N) == 0)
    # The real stalemate: elapsed, static, untouched -> BOTH teams docked.
    pen = R.no_progress_penalty(ball, start, torch.tensor([6.0]),
                                torch.tensor([False]), N)
    assert pen.shape == (1, N)
    assert torch.all(pen == -1.0)


# ── Whole-stack audit ───────────────────────────────────────────────────────
def test_every_penalty_is_non_positive_on_a_random_state():
    g = torch.Generator().manual_seed(0)
    e = 4
    pos = torch.zeros(e, N, 3)
    pos[..., 0].uniform_(-3.0, 3.0, generator=g)
    pos[..., 1].uniform_(-2.0, 2.0, generator=g)
    pos[..., 2].uniform_(0.03, 0.25, generator=g)
    gvec = torch.zeros(e, N, 3)
    gvec[..., 2].uniform_(-1.0, 0.5, generator=g)
    ball = torch.zeros(e, 3)
    ball[..., 0].uniform_(-3.4, 3.4, generator=g)
    ball[..., 1].uniform_(-2.2, 2.2, generator=g)
    ball[..., 2] = P.BALL_RADIUS
    prev = ball[..., :2].clone()

    values = {
        "teammate_spacing": R.teammate_spacing_penalty(pos, SIGN),
        "fallen": R.fallen_penalty(gvec),
        "no_progress": R.no_progress_penalty(
            ball, prev, torch.full((e,), 10.0), torch.zeros(e, dtype=torch.bool), N
        ),
    }
    for name, v in values.items():
        assert v.shape == (e, N), name
        assert (v <= 1e-6).all(), f"{name} went positive: {v.max().item()}"
    # And the guard the env can call every step must accept this stack.
    R.assert_penalty_signs(values)


def test_penalty_sign_guard_catches_a_bad_term():
    with pytest.raises(AssertionError):
        R.assert_penalty_signs({"fallen": torch.tensor([[0.1]])})


def test_weights_are_positive_for_penalties():
    for name in R.PENALTY_TERMS:
        assert R.WEIGHTS[name] > 0, (
            f"{name} returns <= 0, so a negative weight would double-negate into "
            "a reward for the violation"
        )


# ── Duck -> ball approach (S0's enabling gradient) ──────────────────────────
def test_approach_pays_the_nearest_duck_and_nothing_when_static():
    ball = torch.tensor([[0.0, 0.0, P.BALL_RADIUS]])
    # n_team=2: team 0 = ducks 0,1 ; team 1 = ducks 2,3.
    duck = torch.tensor([[[-1.0, 0.0, 0.115], [-2.0, 0.0, 0.115],
                          [2.0, 0.0, 0.115], [3.0, 0.0, 0.115]]])
    pm = R.team_min_ball_distance(duck, ball, 2)
    assert pm.shape == (1, 2)
    # 3D distance: the +-0.08 m height difference is included.
    assert pm[0, 0] == pytest.approx(1.0, abs=0.01)
    assert pm[0, 1] == pytest.approx(2.0, abs=0.01)

    # Static: exactly zero for everybody (no farmable offset).
    assert torch.all(R.ball_approach_progress(duck, ball, pm, 2) == 0)

    # Duck 0 (team 0's nearest) walks 0.1 m closer -> only it is paid.
    duck2 = duck.clone()
    duck2[0, 0, 0] = -0.9
    r = R.ball_approach_progress(duck2, ball, pm, 2)
    assert r[0, 0] > 0, "the duck that approached was not paid"
    assert r[0, 2] == 0 and r[0, 3] == 0, "the other team was paid for nothing"
    assert r[0, 1] == 0, "a non-nearest teammate was paid"

    # Retreat is charged symmetrically.
    duck3 = duck.clone()
    duck3[0, 0, 0] = -1.1
    assert R.ball_approach_progress(duck3, ball, pm, 2)[0, 0] < 0


def test_approach_is_bounded_and_team_relative():
    ball = torch.tensor([[0.0, 0.0, P.BALL_RADIUS]])
    duck = torch.tensor([[[-3.0, 0.0, 0.115], [0.5, 0.0, 0.115]]])
    pm = R.team_min_ball_distance(duck, ball, 1)
    near = duck.clone()
    near[0, 0, 0] = -0.1            # team 0 closes a huge gap in one step
    r = R.ball_approach_progress(near, ball, pm, 1)
    assert r.abs().max() <= 1.0
    assert r[0, 0] > 0 and r[0, 1] == 0


def test_approach_weight_is_positive_and_registered():
    assert R.WEIGHTS["ball_approach"] > 0
    assert "ball_approach" not in R.PENALTY_TERMS


# ── NaN safety (a diverged contact solve must not kill the whole PPO run) ────
def test_approach_reward_survives_inf_and_nan_inputs():
    ball = torch.tensor([[0.0, 0.0, P.BALL_RADIUS]])
    duck = torch.tensor([[[-1.0, 0.0, 0.115], [2.0, 0.0, 0.115]]])
    inf_prev = torch.tensor([[float("inf"), float("inf")]])
    r = R.ball_approach_progress(duck, ball, inf_prev, 1)
    assert torch.isfinite(r).all(), "inf*0 leaked a NaN into the approach reward"

    duck_nan = duck.clone()
    duck_nan[0, 0, 0] = float("nan")
    r2 = R.ball_approach_progress(duck_nan, ball, inf_prev, 1)
    assert torch.isfinite(r2).all(), "a NaN duck position leaked through"


def test_all_terms_are_finite_on_a_diverged_state():
    """Every reward term must return finite values even when the state is junk;
    rsl_rl aborts the entire training run on one NaN reward."""
    e, n = 2, 4
    pos = torch.full((e, n, 3), float("nan"))
    ball = torch.full((e, 3), float("nan"))
    g = torch.full((e, n, 3), float("nan"))
    sign = O.team_attack_sign(n, 2)
    vals = {
        "spacing": R.teammate_spacing_penalty(pos, sign),
        "fallen": R.fallen_penalty(g),
        "ball_side": R.ball_goal_side(ball).float(),
        "progress": R.ball_goal_progress(ball, ball[..., :2], torch.zeros(2), sign),
    }
    for name, v in vals.items():
        assert torch.isfinite(v).all(), f"{name} produced non-finite values"
