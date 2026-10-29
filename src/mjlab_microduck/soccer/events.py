"""Episode reset logic for the soccer task (P3-T7), as pure functions.

The env cfg calls these from reset events; keeping them torch-only means the
spawn rules are unit-testable without building a 22-duck world.

WHY THE SPAWN IS DELIBERATELY ASYMMETRIC
  Symmetric self-play has a stable, worthless fixed point: both teams mirror each
  other exactly, the ball never leaves the centre spot, and the gradient is
  identically zero everywhere. The fixes, all applied here at t=0:

    * the ball starts anywhere in a +-0.3 m patch around the centre spot, not
      exactly on it;
    * every duck gets its own +-5 cm spawn jitter (so a 4-4-2 mirror match does
      not produce bit-identical mirrored states);
    * the team that starts closer to the ball is the one that "has possession"
      (used to seed the ``last_touch`` observation), which itself is random
      because of the two above.
"""

from __future__ import annotations

import torch

from mjlab_microduck.soccer import pitch as P

BALL_SPAWN_RADIUS = 0.30
"""Uniform disc radius around the centre spot where the ball may start (m)."""

DUCK_SPAWN_JITTER = 0.05
"""Per-axis uniform spawn jitter per duck (m). Must stay well under half the
minimum formation spacing (0.35 m) so no two ducks can spawn overlapping."""


def reset_ball_pos(
    n_worlds: int,
    generator: torch.Generator | None = None,
    radius: float = BALL_SPAWN_RADIUS,
    device=None,
) -> torch.Tensor:
    """(E, 3) ball spawn inside the centre circle, on the ground."""
    xy = torch.zeros(n_worlds, 2, device=device)
    # Uniform over a disc: r ~ U(0, R^2)^0.5 keeps the density flat (a naive
    # r ~ U(0,R) piles the ball up in the middle and quietly removes the
    # asymmetry we are buying this for).
    r = (torch.rand(n_worlds, generator=generator, device=device) * radius**2).sqrt()
    theta = torch.rand(n_worlds, generator=generator, device=device) * (2 * torch.pi)
    xy[:, 0] = r * torch.cos(theta)
    xy[:, 1] = r * torch.sin(theta)
    pos = torch.zeros(n_worlds, 3, device=device)
    pos[:, :2] = xy
    pos[:, 2] = P.BALL_RADIUS
    return pos


def spawn_jitter(
    n_worlds: int,
    n_ducks: int,
    generator: torch.Generator | None = None,
    jitter: float = DUCK_SPAWN_JITTER,
    device=None,
) -> torch.Tensor:
    """(E, N, 2) per-duck xy jitter, uniform in +-``jitter`` per axis."""
    return (torch.rand(n_worlds, n_ducks, 2, generator=generator, device=device) * 2 - 1) * jitter


def possession(
    ball_pos_w: torch.Tensor,     # (E, 3)
    duck_pos_w: torch.Tensor,     # (E, N, 3)
    team_attack_sign: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """(E,) 1.0 if team 0 is closer to the ball at spawn, else 0.0.

    Used to seed the ``last_touch`` observation so both branches of that
    one-hot are seen from step 0 (a slot that is always 1 has dead weights
    forever).
    """
    d = (duck_pos_w[..., :2] - ball_pos_w[..., None, :2]).norm(dim=-1)   # (E, N)
    team0 = (team_attack_sign.reshape(1, -1) > 0).expand_as(d)
    d0 = d.masked_fill(~team0, float("inf")).min(dim=-1).values
    d1 = d.masked_fill(team0, float("inf")).min(dim=-1).values
    return (d0 <= d1).to(torch.float32)


def spawn_poses(
    n_worlds: int,
    generator: torch.Generator | None = None,
    jitter: float = DUCK_SPAWN_JITTER,
    device=None,
    n_team: int | None = None,
) -> torch.Tensor:
    """(E, 2*n_team, 3) spawn positions: formation + jitter, z = trunk height.

    Reads the formation from ``soccer.scene`` so the two can never drift apart.
    ``n_team`` MUST match the world being reset: the default (11) puts 1 v 1
    ducks on the 11-a-side goalkeeper line, i.e. 3 m from the ball, and the
    episode is over before anybody reaches it.
    """
    from mjlab_microduck.soccer import pitch as P
    from mjlab_microduck.soccer import scene as S

    n_team = P.NUM_DUCKS_PER_TEAM if n_team is None else n_team
    base = torch.tensor(
        [(x, y) for x, y, _ in S.formation(n_team)], device=device, dtype=torch.float32
    )
    pos = base.view(1, -1, 2) + spawn_jitter(n_worlds, base.shape[0], generator, jitter, device)
    out = torch.zeros(n_worlds, base.shape[0], 3, device=device)
    out[..., :2] = pos
    out[..., 2] = S.DUCK_SPAWN_Z
    return out


def respawn_within_pitch(
    pos: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Clamp (E, N, 3) spawn xy inside the touchlines (jitter must not push a
    duck into the wall)."""
    pos = pos.clone()
    pos[..., 0] = pos[..., 0].clamp(-P.PITCH_L / 2 + margin, P.PITCH_L / 2 - margin)
    pos[..., 1] = pos[..., 1].clamp(-P.PITCH_W / 2 + margin, P.PITCH_W / 2 - margin)
    return pos
