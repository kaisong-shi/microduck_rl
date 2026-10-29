"""Per-duck observation builder for the 11 v 11 soccer task (P3-T1).

    build_obs(...) -> (E, 22, 97)

Pure torch, no simulator objects: P4's lane wrapper feeds it tensors extracted
once per control step, and every rule here is unit-testable without a GPU.

FRAME — the one decision everything else follows
  Every *relative* quantity is expressed in the duck's own BODY frame
  (``v_b = R^T v_world``, full rotation, not just yaw). That single choice buys
  team symmetry for free: rotate the world 180° about the centre spot and swap
  the teams, and a duck's observation is bit-identical, because
  ``R' = R·Rz(pi)`` and ``v' = Rz(pi) v`` cancel. No team one-hot, no mirroring
  table, half the learning problem. ``tests/test_soccer_obs.py`` asserts it.

  World-anchored quantities (own position on the pitch) are the exception: they
  are multiplied by ``team_sign`` (+1 team 0, -1 team 1) so both teams read the
  same value in the mirrored state.

  The consequence to remember: the policy knows its heading implicitly (body
  frame) but NOT its absolute yaw. Anything that needs an absolute direction
  must be given as a relative vector (the own-goal block exists for that).

LAYOUT (97 dims) — append-only, single source of truth is ``_BLOCKS``
   0..14   joint_pos_rel      14   (servo joints, HOME-relative)
  14..28   joint_vel_rel      14
  28..31   base_ang_vel_b      3
  31..34   projected_gravity_b 3
  34..48   last_action        14
  48..51   ball_rel_pos_b      3
  51..54   ball_rel_vel_b      3
  54..55   ball_distance       1   (normalised by 8 m)
  55..61   mates rel xy        6   (3 nearest x 2)
  61..67   mates rel vel xy    6
  67..73   opps  rel xy        6
  73..79   opps  rel vel xy    6
  79..81   mates centroid xy   2
  81..83   opps  centroid xy   2
  83..85   own position xy     2   (canonical, normalised to ~[-1, 1])
  85..86   own height          1   (trunk z / 0.25 m)
  86..90   role one-hot        4   (GK, DEF, MID, FWD)
  90..91   time remaining      1
  91..92   goal difference     1   (own - opponent, /3, clamped)
  92..94   last touch one-hot  2   (self team / opponent team)
  94..97   own goal rel pos_b  3

Small-sided stages (S0-S3) shrink the world; missing players are simply never
selected (their distance is inf) and their slots stay ZERO — the layout is
identical at every stage, so a warm start sees the same input space.
"""

from __future__ import annotations

import torch

# ── Layout ───────────────────────────────────────────────────────────────────
N_NEAREST = 3
ROLE_NAMES: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
N_ROLES = len(ROLE_NAMES)

_BLOCKS: tuple[tuple[str, int], ...] = (
    ("joint_pos_rel", 14),
    ("joint_vel_rel", 14),
    ("base_ang_vel_b", 3),
    ("projected_gravity_b", 3),
    ("last_action", 14),
    ("ball_rel_pos_b", 3),
    ("ball_rel_vel_b", 3),
    ("ball_distance", 1),
    ("mates_rel_xy_b", 2 * N_NEAREST),
    ("mates_rel_vel_xy_b", 2 * N_NEAREST),
    ("opps_rel_xy_b", 2 * N_NEAREST),
    ("opps_rel_vel_xy_b", 2 * N_NEAREST),
    ("mates_centroid_xy_b", 2),
    ("opps_centroid_xy_b", 2),
    ("self_pos_xy", 2),
    ("self_height", 1),
    ("role_onehot", N_ROLES),
    ("time_remaining", 1),
    ("goal_diff", 1),
    ("last_touch_onehot", 2),
    ("own_goal_rel_pos_b", 3),
)

OBS_DIM = sum(d for _, d in _BLOCKS)
assert OBS_DIM == 97, OBS_DIM


def obs_slices() -> dict[str, slice]:
    """Name -> slice into the 97-dim vector."""
    out: dict[str, slice] = {}
    i = 0
    for name, dim in _BLOCKS:
        out[name] = slice(i, i + dim)
        i += dim
    return out


SLICES = obs_slices()

# Normalisation constants.
MAX_BALL_DIST = 8.0      # pitch diagonal is 8.32 m; the walls keep us inside
MAX_SELF_HEIGHT = 0.25   # a duck standing tall is ~0.25 m
MAX_GOAL_DIFF = 3.0      # first to 3 goals
PITCH_HALF_L = 3.5
PITCH_HALF_W = 2.25

# Joint block layout (must match the actuator order in the compiled model).
N_JOINTS = 14
assert SLICES["joint_pos_rel"].stop == N_JOINTS


# ── Math helpers ─────────────────────────────────────────────────────────────
def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(…, 4) wxyz -> (…, 3, 3) rotation matrix (world-from-body)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(dim=-1)
    o = torch.ones_like(w)
    row0 = torch.stack((o - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1)
    row1 = torch.stack((2 * (x * y + z * w), o - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1)
    row2 = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), o - 2 * (x * x + y * y)), -1)
    return torch.stack((row0, row1, row2), dim=-2)


def to_body(v_world: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """World vectors (E, N, 3) -> body frame, given (E, N, 3, 3) matrices."""
    return torch.einsum("enji,enj->eni", rot, v_world)


def team_ids(n_ducks: int, n_team: int, device=None) -> torch.Tensor:
    """0 for the first ``n_team`` ducks (attack +x), 1 for the rest."""
    return torch.arange(n_ducks, device=device) // n_team


def team_attack_sign(n_ducks: int, n_team: int, device=None) -> torch.Tensor:
    """+1 for team 0 (attacks +x), -1 for team 1."""
    return torch.where(
        team_ids(n_ducks, n_team, device) == 0,
        torch.ones(n_ducks, device=device),
        -torch.ones(n_ducks, device=device),
    )


def roles_for_team(n_team: int) -> tuple[str, ...]:
    """Roles for a small-sided team, subsampled evenly from the 4-4-2 template.

    11 players -> the real 4-4-2 role list. Fewer players -> an even subsample
    (2 v 2 -> GK + FWD, 4 v 4 -> GK + 2x MID + FWD), so the role one-hot keeps
    meaning something at every curriculum stage instead of collapsing to a
    constant.
    """
    from mjlab_microduck.soccer.scene import ROLE_ORDER

    if n_team <= 0:
        return ()
    if n_team >= len(ROLE_ORDER):
        return tuple(ROLE_ORDER)
    if n_team == 1:
        return (ROLE_ORDER[len(ROLE_ORDER) // 2],)
    step = (len(ROLE_ORDER) - 1) / (n_team - 1)
    return tuple(ROLE_ORDER[round(i * step)] for i in range(n_team))


def role_onehot(n_ducks: int, n_team: int, device=None) -> torch.Tensor:
    """(N, 4) role one-hot for every duck (duck-major; team 1 mirrors team 0)."""
    per_team = roles_for_team(n_team)
    ids = torch.tensor(
        [ROLE_NAMES.index(per_team[i % n_team]) for i in range(n_ducks)],
        dtype=torch.long, device=device,
    )
    out = torch.zeros(n_ducks, N_ROLES, dtype=torch.float32, device=device)
    out.scatter_(1, ids.view(-1, 1), 1.0)
    return out


def _nearest(pos_xy: torch.Tensor, mask: torch.Tensor, k: int):
    """Indices + validity of the k nearest allowed partners, per duck.

    ALWAYS returns exactly ``k`` columns (zero-padded when the team is smaller
    than k+1). A variable column count would change OBS_DIM between curriculum
    stages — the one thing the shared layout must never do.
    """
    e, n, _ = pos_xy.shape
    d = torch.cdist(pos_xy, pos_xy).masked_fill(~mask, float("inf"))
    kk = max(min(k, n - 1), 1)
    dist, idx = d.topk(kk, dim=-1, largest=False)
    valid = torch.isfinite(dist)
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    if kk < k:
        pad = k - kk
        idx = torch.cat((idx, torch.zeros(e, n, pad, dtype=idx.dtype, device=idx.device)), -1)
        valid = torch.cat((valid, torch.zeros(e, n, pad, dtype=torch.bool, device=idx.device)), -1)
    return idx, valid


def _gather_pairs(src: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(E, N, N, C) + (E, N, k) -> (E, N, k*C), zeroed where invalid."""
    pick = idx.unsqueeze(-1).expand(-1, -1, -1, src.shape[-1])
    out = torch.gather(src, 2, pick)                     # (E, N, k, C)
    out = out * valid.unsqueeze(-1).to(out.dtype)
    e, n = out.shape[0], out.shape[1]
    return out.reshape(e, n, -1)


# ── Main builder ─────────────────────────────────────────────────────────────
def build_obs(
    *,
    root_pos_w: torch.Tensor,           # (E, N, 3)
    root_quat_w: torch.Tensor,          # (E, N, 4) wxyz
    root_lin_vel_w: torch.Tensor,       # (E, N, 3)
    joint_pos_rel: torch.Tensor,        # (E, N, 14)
    joint_vel_rel: torch.Tensor,        # (E, N, 14)
    base_ang_vel_b: torch.Tensor,       # (E, N, 3)  already body frame
    projected_gravity_b: torch.Tensor,  # (E, N, 3)  already body frame
    last_action: torch.Tensor,          # (E, N, 14)
    ball_pos_w: torch.Tensor,           # (E, 3)
    ball_vel_w: torch.Tensor,           # (E, 3)
    own_goal_pos_w: torch.Tensor,       # (N, 3) this duck's own goal centre
    role: torch.Tensor,                 # (N, 4)
    time_remaining: torch.Tensor,       # (E,) 1 = kickoff, 0 = end
    goal_diff: torch.Tensor,            # (E,) own - opponent
    last_touch: torch.Tensor,           # (E,) 1 if our team touched last else 0
    n_team: int | None = None,
    nearest: int = N_NEAREST,
) -> torch.Tensor:
    """Assemble the (E, N, 97) per-duck observation. See the module docstring."""
    e, n, _ = root_pos_w.shape
    n_team = n_team or n // 2
    dev = root_pos_w.device

    rot = quat_to_matrix(root_quat_w)
    pos_xy = root_pos_w[..., :2]
    team = team_ids(n, n_team, dev)

    # ── Ball (3D, body frame) ────────────────────────────────────────────────
    ball_p = ball_pos_w.unsqueeze(1).expand(-1, n, -1)
    ball_v = ball_vel_w.unsqueeze(1).expand(-1, n, -1)
    ball_rel = to_body(ball_p - root_pos_w, rot)
    ball_rel_vel = to_body(ball_v - root_lin_vel_w, rot)
    ball_dist = (ball_p[..., :2] - pos_xy).norm(dim=-1, keepdim=True) / MAX_BALL_DIST

    # ── Nearest mates / opponents ────────────────────────────────────────────
    eye = torch.eye(n, dtype=torch.bool, device=dev).view(1, n, n)
    same = team.view(1, 1, n) == team.view(1, n, 1)                # (1, N, N)
    mate_idx, mate_ok = _nearest(pos_xy, same & ~eye, nearest)
    opp_idx, opp_ok = _nearest(pos_xy, ~same, nearest)

    rel_xy = pos_xy.unsqueeze(2) - pos_xy.unsqueeze(1)              # (E, N, N, 2)
    rel_vxy = root_lin_vel_w[..., :2].unsqueeze(2) - root_lin_vel_w[..., :2].unsqueeze(1)

    mates_rel_w = _gather_pairs(rel_xy, mate_idx, mate_ok)
    mates_vel_w = _gather_pairs(rel_vxy, mate_idx, mate_ok)
    opps_rel_w = _gather_pairs(rel_xy, opp_idx, opp_ok)
    opps_vel_w = _gather_pairs(rel_vxy, opp_idx, opp_ok)

    def _xy_pairs_to_body(xy: torch.Tensor) -> torch.Tensor:
        """(E, N, 2k) world xy pairs -> (E, N, 2k) body xy pairs."""
        k = xy.shape[-1] // 2
        pairs = xy.reshape(e, n, k, 2)
        v = torch.zeros(e, n, k, 3, device=dev, dtype=xy.dtype)
        v[..., :2] = pairs
        v = v.reshape(e, n * k, 3)
        rot_k = rot.unsqueeze(2).expand(-1, -1, k, -1, -1).reshape(e, n * k, 3, 3)
        return to_body(v, rot_k)[..., :2].reshape(e, n, k * 2)

    mates_rel = _xy_pairs_to_body(mates_rel_w)
    mates_vel = _xy_pairs_to_body(mates_vel_w)
    opps_rel = _xy_pairs_to_body(opps_rel_w)
    opps_vel = _xy_pairs_to_body(opps_vel_w)

    # ── Team centroids (relative, body frame) ────────────────────────────────
    same_f = same.view(n, n).to(root_pos_w.dtype)
    opp_f = 1.0 - same_f
    # pos_xy is (E, N, 2): player index is the MIDDLE dim, coord the last one.
    mate_c = torch.einsum("ij,ejc->eic", same_f, pos_xy) / same_f.sum(-1).clamp_min(1).view(1, n, 1)
    opp_c = torch.einsum("ij,ejc->eic", opp_f, pos_xy) / opp_f.sum(-1).clamp_min(1).view(1, n, 1)

    def _xy_to_body(s: torch.Tensor) -> torch.Tensor:
        v = torch.zeros(*s.shape[:-1], 3, device=dev, dtype=s.dtype)
        v[..., :2] = s
        return to_body(v, rot)[..., :2]

    mate_c_rel = _xy_to_body(mate_c - pos_xy)
    opp_c_rel = _xy_to_body(opp_c - pos_xy)

    # ── World-anchored, canonicalised by team ────────────────────────────────
    ts = team_attack_sign(n, n_team, dev).view(1, n, 1)
    self_xy = root_pos_w[..., :2] * ts
    self_xy = torch.stack(
        (self_xy[..., 0] / PITCH_HALF_L, self_xy[..., 1] / PITCH_HALF_W), dim=-1
    ).clamp(-1.5, 1.5)
    self_h = (root_pos_w[..., 2:3] / MAX_SELF_HEIGHT).clamp(0.0, 1.5)

    goal_rel = to_body(own_goal_pos_w.view(1, n, 3) - root_pos_w, rot)

    # ── Contest state ────────────────────────────────────────────────────────
    # These are per-WORLD scalars, but the policy must read them from ITS OWN
    # side: "am I ahead?" and "did MY team touch last?" both flip sign for the
    # other team. Broadcasting the world value would hand both teams the same
    # answer (and break the 180-degree symmetry).
    def _col(v: torch.Tensor) -> torch.Tensor:
        return v.reshape(-1, 1, 1).expand(e, n, 1)

    team_is_zero = (team == 0).to(root_pos_w.dtype).view(1, n, 1)
    sign = ts                                   # (1, N, 1), +1 team 0 / -1 team 1
    time_rem = _col(time_remaining)
    gdiff = (_col(goal_diff) * sign / MAX_GOAL_DIFF).clamp(-1.0, 1.0)
    touch_world = _col(last_touch).clamp(0.0, 1.0)          # 1 = team 0 touched last
    touch_own = touch_world * team_is_zero + (1.0 - touch_world) * (1.0 - team_is_zero)
    touch2 = torch.cat((touch_own, 1.0 - touch_own), dim=-1)

    obs = torch.cat(
        (
            joint_pos_rel,
            joint_vel_rel,
            base_ang_vel_b,
            projected_gravity_b,
            last_action,
            ball_rel,
            ball_rel_vel,
            ball_dist,
            mates_rel,
            mates_vel,
            opps_rel,
            opps_vel,
            mate_c_rel,
            opp_c_rel,
            self_xy,
            self_h,
            role.view(1, n, N_ROLES).expand(e, -1, -1),
            time_rem,
            gdiff,
            touch2,
            goal_rel,
        ),
        dim=-1,
    )
    assert obs.shape == (e, n, OBS_DIM), obs.shape
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
