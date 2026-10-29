"""Soccer reward terms (P3-T5), as pure functions.

SIGN CONVENTION — read this before adding a term
  Every *penalty* in this module returns **≤ 0** (self-negating, the
  ``microduck_mdp`` style), so its configured weight is **POSITIVE**. A penalty
  returning ≥ 0 with a negative weight is a double negation that PAYS for the
  violation, and a policy will farm it (this repo has paid for that lesson four
  times). The audit rule from AGENTS.md still applies at run time: in wandb every
  ``Episode_Reward/<penalty>`` must read ≤ 0.

NO JACKPOTS
  The only dense "do the task" term is a *potential difference* on the ball's
  distance to the opponent goal (``ball_goal_progress``): advancing pays,
  holding still pays exactly zero, retreating pays negative. It cannot be farmed
  by parking anywhere, and it is bounded per step by the pitch diagonal.
  A naive ``-gamma*Phi' - Phi`` (discounted) form would NOT be used: with
  ``gamma < 1`` it pays ``d*(1-gamma)`` every step, i.e. it pays MORE the further
  the ball is from the goal — an incentive to keep the ball in your own half.
  We therefore use the undiscounted difference (``gamma = 1``), which is exactly
  the "pay Δprogress" rule in AGENTS.md.

GOALS
  ``goal_event`` pays ±1 scaled per duck: scoring pays the scoring team, conceding
  pays its opponent, and both are TEAM-RELATIVE (see ``sign`` below), so a policy
  cannot learn "shoot at +x" as a shortcut.
"""

from __future__ import annotations

import math

import torch

from mjlab_microduck.soccer import obs as O
from mjlab_microduck.soccer import pitch as P

# ── Weights (documented in docs/soccer_tasks.md §5.2) ────────────────────────
#
# REWARD MASS — the numbers below are not free parameters, they are ordered by
# what one step of each behaviour is worth. Measured magnitudes at 50 Hz:
#
#   duck walking 0.17 m/s, pushing the ball towards goal:
#       d(Phi) = (0.17/50)/8 = 4.2e-4  ->  x15 = +0.006/step  (~+4 over a
#       full 13 s possession)
#   one clean kick (1.9 m of ball travel in a single step):
#       (1.9/8) x 15 = +3.5 in one step      <- the "kick" incentive
#   a goal: +50                              <- 12 possessions
#   a credited touch (max once per 5 s): +0.15
#   lying on the ground: -0.02/step          <- -20 over 20 s down
#   two ducks fully stacked (0.25 m overlap): -0.0125/step
#
# The FIRST version of this module used FALLEN=2.0 and SPACING=1.0, i.e. 300x
# the task gradient. A 3-iteration smoke test showed it immediately: value loss
# 0.002 -> 10 -> 131, because a random policy falls at t=0, stays down for all
# 1500 steps and returns a flat -3000. "Do nothing" then beats "play football",
# which is the classic do-nothing basin. Penalties are nudges, not the task.
GOAL_WEIGHT = 50.0
BALL_PROGRESS_WEIGHT = 15.0
BALL_APPROACH_WEIGHT = 15.0
TOUCH_WEIGHT = 0.5
SPACING_WEIGHT = 0.05
FALLEN_WEIGHT = 0.02
NO_PROGRESS_WEIGHT = 0.5
"""Penalty weights are POSITIVE because the functions return <= 0."""

TOUCH_BONUS = 0.3
"""Raw per-touch payoff (weighted by TOUCH_WEIGHT -> ~0.6/step at first touch)."""

TOUCH_COOLDOWN_S = 5.0
"""One credited touch per 5 s per duck — a dribbler touching the ball every step
must not out-earn a defender who intercepts once."""

SPACING_MIN_M = 0.25
"""Teammates closer than this are getting in each other's way (a duck is ~0.1 m
wide and needs ~0.15 m to swing a leg)."""

FALLEN_TILT_DEG = 60.0
"""Tilt beyond which a duck counts as fallen (trunk axis vs vertical)."""

NO_PROGRESS_WINDOW_S = 5.0
NO_PROGRESS_MIN_MOVE = 0.15
"""No-progress watchdog: if the ball has not moved this far in this long AND
nobody touched it, the episode is a stalemate (22 ducks wedged on the ball)."""


# ── Goal / scoring ───────────────────────────────────────────────────────────
def ball_goal_side(ball_pos_w: torch.Tensor) -> torch.Tensor:
    """+1 if the ball is in the +x goal, -1 if in the -x goal, 0 otherwise.

    "In the goal" = past the goal line, inside the mouth, below the crossbar.
    The net is SOLID (see pitch.py), so once the ball is in it stays in: the
    caller can treat this as a level, not a single-frame event.
    """
    x, y, z = ball_pos_w[..., 0], ball_pos_w[..., 1], ball_pos_w[..., 2]
    inside_y = y.abs() < P.GOAL_W / 2
    below_bar = z < P.GOAL_H
    return torch.where(
        (x > P.PITCH_L / 2) & inside_y & below_bar,
        torch.ones_like(x, dtype=torch.long),
        torch.where(
            (x < -P.PITCH_L / 2) & inside_y & below_bar,
            -torch.ones_like(x, dtype=torch.long),
            torch.zeros_like(x, dtype=torch.long),
        ),
    )


def goal_events(prev_side: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Rising-edge goal per world: +1 team 0 scored, -1 team 1 scored, 0 none.

    Edge-triggered (not a level) so a ball resting in the net pays exactly once.
    """
    return torch.where(side != prev_side, side, torch.zeros_like(side))


def goal_event_reward(events: torch.Tensor, team_attack_sign: torch.Tensor) -> torch.Tensor:
    """(E,) events + (N,) signs -> (E, N) reward, TEAM-RELATIVE.

    ``events`` is +1 when the ball entered the +x goal (team 0 scored). For team
    0 that is a reward (+1), for team 1 a penalty (-1): multiply by the duck's
    attack sign. Without this, both teams would be rewarded for the same goal
    and the policy would learn to shoot into its own net.
    """
    return events.reshape(-1, 1).to(torch.float32) * team_attack_sign.reshape(1, -1)


# ── Ball progress (potential-based, unfarmable) ──────────────────────────────
def ball_goal_distance(ball_pos_w: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
    """Normalised XY distance from the ball to a goal centre. (E,) in [0, ~1]."""
    d = (ball_pos_w[..., :2] - goal_xy).norm(dim=-1)
    return d / 8.0  # pitch diagonal is 8.32 m


def ball_goal_progress(
    ball_pos_w: torch.Tensor,
    prev_ball_xy: torch.Tensor,
    target_goal_xy: torch.Tensor,
    team_attack_sign: torch.Tensor,
) -> torch.Tensor:
    """Δ(progress toward the duck's attacking goal), per duck, in [-1, 1].

    Potential Phi = -d/8 with an UNDISCOUNTED difference, so:
      * ball moves 10 cm toward the target goal -> +0.0125 * weight
      * ball does not move                       -> exactly 0 (no farmable offset)
      * ball moves toward your OWN goal          -> negative
    Team-relative via ``team_attack_sign`` so both teams are rewarded for moving
    the ball toward the goal they attack.
    """
    prev_d = (prev_ball_xy - target_goal_xy).norm(dim=-1) / 8.0
    cur_d = (ball_pos_w[..., :2] - target_goal_xy).norm(dim=-1) / 8.0
    return torch.nan_to_num(
        (prev_d - cur_d).reshape(-1, 1) * team_attack_sign.reshape(1, -1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


# ── Touch ────────────────────────────────────────────────────────────────────
# ── Duck -> ball approach (the ENABLING skill, S0's whole point) ────────────
def team_min_ball_distance(
    duck_pos_w: torch.Tensor,   # (E, N, 3)
    ball_pos_w: torch.Tensor,   # (E, 3)
    n_team: int,
) -> torch.Tensor:
    """(E, 2): distance from each team's CLOSEST duck to the ball."""
    d = (duck_pos_w - ball_pos_w[:, None, :]).norm(dim=-1)          # (E, N)
    e = d.shape[0]
    teams = torch.arange(d.shape[1], device=d.device) // n_team
    out = torch.full((e, 2), float("inf"), device=d.device, dtype=d.dtype)
    for k in range(2):
        out[:, k] = d.masked_fill(teams != k, float("inf")).min(dim=-1).values
    return out


def ball_approach_progress(
    duck_pos_w: torch.Tensor,
    ball_pos_w: torch.Tensor,
    prev_team_min_dist: torch.Tensor,   # (E, 2) or (E, N) broadcast-able
    n_team: int,
) -> torch.Tensor:
    """(E, N) reward for closing the distance to the ball, per team.

    WHY THIS EXISTS
      Without it S0 has no gradient at all: the only dense task term was
      "ball -> opponent goal", which requires touching the ball first, which
      requires walking, which is exactly what a from-scratch policy cannot do.
      A 300-iteration 1 v 1 run measured 0 touches, 0 goals, value loss 0.0001
      and a growing action std — the policy sat still and collected the stalemate
      penalty. This term pays the team that gets closer to the BALL.

    FORM
      Team-level potential (the team's minimum duck->ball distance), UNDISCOUNTED
      difference — same anti-farm reasoning as ``ball_goal_progress``: no motion
      pays exactly zero, approaching pays positive, retreating pays negative.

    WHO GETS PAID
      Only the duck that is currently closest (one-hot per team). Paying every
      teammate would hand a duck a reward it did not cause, which is pure credit
      assignment noise for the other 10. The mask does make this "almost" a
      potential (it can jump when the nearest duck changes), but the jump is
      bounded by one step of walking (~0.0034 m / 8), i.e. ~1e-4 of reward.
    """
    d = (duck_pos_w - ball_pos_w[:, None, :]).norm(dim=-1)          # (E, N)
    e, n = d.shape
    teams = torch.arange(n, device=d.device) // n_team
    cur_min = team_min_ball_distance(duck_pos_w, ball_pos_w, n_team)  # (E, 2)
    prev = prev_team_min_dist
    if prev.shape[-1] != 2:
        prev = torch.stack(
            [prev.masked_fill(teams != k, float("inf")).min(dim=-1).values for k in range(2)],
            dim=-1,
        )
    # inf - finite = inf, and inf * 0 = NaN: guard before the one-hot multiply
    # (a team whose ducks all spawned at the ball, or a diverged sim, can leave
    # an inf in `prev`). NaN here poisons the whole PPO batch.
    delta = torch.nan_to_num((prev - cur_min) / 8.0, nan=0.0, posinf=0.0, neginf=0.0)
    # One-hot payout to the nearest duck of each team.
    nearest = torch.zeros(e, n, device=d.device)
    for k in range(2):
        idx = d.masked_fill(teams != k, float("inf")).argmin(dim=-1)
        nearest[torch.arange(e, device=d.device), idx] = 1.0
    team_of = teams.view(1, n)
    delta_per_duck = torch.gather(delta, 1, team_of.expand(e, n).reshape(e, -1).reshape(e, n).long())
    return torch.nan_to_num(delta_per_duck * nearest, nan=0.0, posinf=0.0, neginf=0.0)


def touch_reward(
    touched: torch.Tensor,          # (E, N) bool: this duck touched the ball now
    since_last_touch_s: torch.Tensor,  # (E, N) seconds since this duck last scored a touch
    cooldown_s: float = TOUCH_COOLDOWN_S,
    bonus: float = TOUCH_BONUS,
) -> torch.Tensor:
    """(E, N) reward for a credited first touch; 0 while on cooldown.

    ``since_last_touch_s`` is caller-owned state (initialise to +inf / a large
    number so the very first touch is always credited).
    """
    eligible = touched & (since_last_touch_s >= cooldown_s)
    return torch.nan_to_num(eligible.to(torch.float32) * bonus, nan=0.0)


# ── Penalties (all <= 0) ─────────────────────────────────────────────────────
def teammate_spacing_penalty(
    root_pos_w: torch.Tensor,       # (E, N, 3)
    team_attack_sign: torch.Tensor,  # (N,)
    min_dist: float = SPACING_MIN_M,
) -> torch.Tensor:
    """(E, N) <= 0: how much this duck crowds its nearest teammate(s).

    Uses a soft-ish sum of overlaps with every teammate (not just the nearest)
    so three ducks in a pile is worse than two. Returns EXACTLY 0 when nothing
    is closer than ``min_dist``, so a well-spread formation is not taxed.
    """
    e, n, _ = root_pos_w.shape
    pos = root_pos_w[..., :2]
    d = torch.cdist(pos, pos)                                   # (E, N, N)
    team = team_attack_sign.reshape(1, n) < 0                   # True for team 1
    same = team == team.transpose(0, 1)                         # (N, N)
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    pair_ok = same & ~eye
    overlap = torch.clamp(min_dist - d, min=0.0)                # >= 0
    overlap = overlap.masked_fill(~pair_ok.unsqueeze(0), 0.0)
    # NaN positions (diverged solve) must not leak through [nan - x = nan].
    return torch.nan_to_num(-overlap.sum(-1), nan=0.0, posinf=0.0, neginf=0.0)


def fallen_penalty(
    projected_gravity_b: torch.Tensor,   # (E, N, 3) body frame, ~ (0,0,-1) upright
    max_tilt_deg: float = FALLEN_TILT_DEG,
) -> torch.Tensor:
    """(E, N) <= 0: 0 while upright, growing once the trunk tilts past the limit.

    ``projected_gravity_b[2] = -cos(tilt)``, so the onset is at
    ``-cos(max_tilt)``; the penalty is the excess, normalised by 1 rad-ish so it
    stays O(1) at "lying on its back" (tilt 90 deg).
    """
    gz = projected_gravity_b[..., 2].clamp(-1.0, 1.0)
    # Upright: gz = -1. Lying on its back: gz = 0. Past the limit means gz has
    # risen ABOVE -cos(max_tilt).
    onset = -math.cos(math.radians(max_tilt_deg))
    return torch.nan_to_num(
        -torch.clamp(gz - onset, min=0.0) / (1.0 - math.cos(math.radians(max_tilt_deg))),
        nan=0.0,
    )


def no_progress_penalty(
    ball_pos_w: torch.Tensor,            # (E, 3)
    window_start_ball_xy: torch.Tensor,  # (E, 2) ball position when the window opened
    window_s: torch.Tensor,              # (E,) seconds elapsed in the current window
    touched_in_window: torch.Tensor,     # (E,) bool
    n_ducks: int,
    window_length_s: float = NO_PROGRESS_WINDOW_S,
    min_move_m: float = NO_PROGRESS_MIN_MOVE,
) -> torch.Tensor:
    """(E, N) <= 0: both teams are docked while the game is stuck.

    Fires only when the window has ELAPSED, the ball has not travelled
    ``min_move_m`` in it, and nobody touched the ball. Applied to BOTH teams:
    a stalemate is a failure of both, and paying only one side would teach the
    other side to freeze (the cheapest way to win).
    """
    moved = (ball_pos_w[..., :2] - window_start_ball_xy).norm(dim=-1)
    stuck = (window_s >= window_length_s) & (moved < min_move_m) & ~touched_in_window
    return torch.nan_to_num(
        -stuck.to(torch.float32).reshape(-1, 1).expand(-1, n_ducks), nan=0.0
    )


# ── Aggregate weights (single place the env cfg reads) ───────────────────────
WEIGHTS: dict[str, float] = {
    "goal_event": GOAL_WEIGHT,
    "ball_goal_progress": BALL_PROGRESS_WEIGHT,
    "ball_approach": BALL_APPROACH_WEIGHT,
    "touch": TOUCH_WEIGHT,
    "teammate_spacing": SPACING_WEIGHT,
    "fallen": FALLEN_WEIGHT,
    "no_progress": NO_PROGRESS_WEIGHT,
}
"""Configured reward weights. Penalties keep POSITIVE weights (see module doc)."""

PENALTY_TERMS: tuple[str, ...] = ("teammate_spacing", "fallen", "no_progress")


def assert_penalty_signs(values: dict[str, torch.Tensor]) -> None:
    """Cheap runtime guard mirroring the wandb audit (used by tests + env)."""
    for name in PENALTY_TERMS:
        if name in values:
            v = values[name]
            if v.numel() and (v > 1e-9).any():
                raise AssertionError(
                    f"penalty {name!r} returned a positive value — it will pay for "
                    "the violation once multiplied by its positive weight"
                )
