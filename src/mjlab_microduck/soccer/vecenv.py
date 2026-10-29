"""rsl_rl adapter for ``SoccerEnv`` — the 22-lane trick, in one place.

rsl_rl knows nothing about teams: it sees ``num_envs`` independent MDPs, each
with ``num_actions`` outputs and one scalar reward. So we present the soccer
world as ``E * 22`` lanes:

    lane = world * 22 + duck_index

and this wrapper is the ONLY place that reshapes between lane space and world
space. Everything upstream (PPO, the rollout buffer, the actor/critic nets) is
completely standard single-agent code, which is exactly the point: parameter
sharing across 22 ducks is just "one policy, many lanes".

Consequences that need care, and where they are handled:
  * ``dones``  — a duck cannot be "done" alone; the whole world resets. So the
    per-world done is broadcast to all 22 lanes.
  * ``episode_length_buf`` — rsl_rl reads it (for timeouts) and sometimes
    writes it (``randint_like``). The getter repeats the world buffer 22x; the
    setter writes back only lane ``world*22`` so a single write cannot produce 22
    inconsistent copies.
  * ``extras["time_outs"]`` — must be per-lane for rsl_rl's bootstrap masking.
"""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer.env import SoccerEnv


class SoccerVecEnv(VecEnv):
    """VecEnv view of a ``SoccerEnv``: ``E * 22`` lanes, 14 actions each."""

    def __init__(self, env: SoccerEnv, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        self.cfg = env.cfg

    # ── VecEnv interface ─────────────────────────────────────────────────────
    @property
    def unwrapped(self) -> SoccerEnv:
        return self.env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf.repeat_interleave(self.env.n_ducks)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # Keep only one authoritative copy per world (lane 0 of each block).
        world_ids = torch.arange(self.env.num_worlds, device=value.device)
        self.env.episode_length_buf = value[world_ids * self.env.n_ducks].clone()

    def seed(self, seed: int = -1) -> int:
        if seed == -1:
            seed = int(torch.randint(0, 10_000, (1,)).item())
        torch.manual_seed(seed)
        return seed

    def get_observations(self) -> TensorDict:
        return TensorDict(self.env.obs, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs = self.env.reset()
        return TensorDict(obs, batch_size=[self.num_envs]), {}

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        obs, rew, done, extras = self.env.step(actions)
        extras = dict(extras)
        # rsl_rl expects long/bool dones and a per-lane timeout mask.
        dones = done.to(torch.long)
        extras["time_outs"] = extras["time_outs"].to(torch.bool)
        return TensorDict(obs, batch_size=[self.num_envs]), rew, dones, extras

    def close(self) -> None:
        self.env.close()


def make_soccer_env(
    num_worlds: int = P.NUM_WORLDS,
    device: str = "cuda:0",
    **overrides,
) -> SoccerVecEnv:
    """Convenience constructor used by the training / eval scripts."""
    from mjlab_microduck.soccer.env import SoccerEnvCfg

    cfg = SoccerEnvCfg(num_worlds=num_worlds, device=device, **overrides)
    return SoccerVecEnv(SoccerEnv(cfg, device=device))
