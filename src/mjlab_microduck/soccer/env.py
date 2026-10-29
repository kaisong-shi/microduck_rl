"""The soccer environment: 22 independent duck agents in one MuJoCo world.

WHY NOT ``ManagerBasedRlEnv``
  mjlab's manager layer is built around ONE robot per world: one action term, one
  observation group, one reward vector of shape (E,). Soccer needs 22 agents per
  world, per-agent rewards, and a 97-dim observation *per duck*. Expressing that
  through the managers means ~300 observation terms and 22 action terms, and the
  reward manager still could not return an (E, 22) tensor. So this env owns its
  own step loop and uses mjlab's lower layers directly (``Scene`` +
  ``Simulation`` + per-entity actuators). What we keep from mjlab is the valuable
  part: the BAM actuators, the CUDA-graph sim, per-world field expansion, and the
  entity API (``set_joint_position_target`` / ``write_data_to_sim`` /
  ``update(dt)``).

WHAT ONE CONTROL STEP DOES (mirrors ManagerBasedRlEnv.step exactly)
    for each of ``decimation`` physics substeps:
        set_joint_position_target(HOME + action * scale)   # per duck
        entity.write_data_to_sim()                          # BAM -> ctrl
        sim.step()
        entity.update(dt)                                   # BAM internal state
    sim.forward()                                           # refresh derived qty
    rewards / terminations / observation

AGENT ↔ LANE MAPPING
  lane = world * 22 + duck, i.e. the flat action/obs tensors are duck-major
  within a world. That is the layout ``soccer/vecenv.py`` hands to rsl_rl.

EPISODE
  Ends on a goal, on a no-progress stalemate, on NaN, or at ``episode_s``. A goal
  ends the episode (field resets) rather than continuing the match: that turns
  the ±50 into a clean terminal signal instead of a mid-episode bump, which is
  what PPO needs to learn a sparse task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import torch

from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.soccer import events as EV
from mjlab_microduck.soccer import obs as OBS
from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer import rewards as RW
from mjlab_microduck.soccer import scene as S

N_TEAM = P.NUM_DUCKS_PER_TEAM
N = P.NUM_DUCKS
N_JOINTS = 14


@dataclass
class SoccerEnvCfg:
    """Everything the env needs. Numbers come from ``pitch.py`` (measured)."""

    num_worlds: int = P.NUM_WORLDS
    n_team: int = P.NUM_DUCKS_PER_TEAM
    """Ducks per team. 1 = S0 (1 v 1), 11 = the full match. Total ducks =
    ``2 * n_team``; every other dimension (actions, lanes, obs) follows."""

    @property
    def n_ducks(self) -> int:
        return 2 * self.n_team

    episode_s: float = P.EPISODE_S
    decimation: int = P.CONTROL_DECIMATION
    timestep: float = P.SIM_TIMESTEP
    njmax: int | None = None
    """None -> P.NJMAX scaled by team size (a 1 v 1 world needs ~10x less contact
    budget than 22 ducks; leaving it at the full value wastes VRAM that could be
    spent on more parallel matches)."""
    nconmax: int | None = None
    action_scale: float = 1.0
    """Action scale in RADIANS per unit action, on top of the HOME offset.

    1.0 matches the repo's recipes (velocity / ball-kick both use scale=1.0), i.e.
    a +-1 action is a +-1 rad (57 deg) joint deviation — a real leg swing.
    An earlier 0.25 rad was a 4x-too-narrow action space: the policy could not
    physically take a step, so it learned to LEAN at ~51 deg towards the ball
    instead (measured mean projected-gravity z = -0.63), and S0 produced 0.45
    touches/episode with 78 % of the time past the 60 deg fall threshold."""
    clip_actions: float | None = 1.0
    touch_radius: float = 0.16
    """Ball counts as touched when its centre is this close to a duck's trunk
    root. Duck half-width ~0.05 m + ball radius 0.035 m + a margin for the
    protruding foot. Sim ground truth, no contact sensor (22 ducks would blow
    past ``contact_sensor_maxmatch``)."""
    obs_noise_std: float = 0.0
    """0 for now: sim-only training. Kept as a knob for the S5 noise stage."""
    critic_extra_dim: int = 10
    """Critic-only global block appended to each duck's 97 dims:
    ball world pos 3 + ball world vel 3 + ball speed 1 + score diff 1 +
    clock 1 + last-touch 1 = 10."""
    device: str = "cuda:0"
    seed: int | None = None

    @property
    def num_agents(self) -> int:
        return self.num_worlds * self.n_ducks

    @property
    def max_episode_length(self) -> int:
        return int(round(self.episode_s / (self.timestep * self.decimation)))


class SoccerEnv:
    """Sim + 22-agent MDP. Exposes a lane-space API (``num_envs = E*22``)."""

    def __init__(self, cfg: SoccerEnvCfg | None = None, device: str | None = None):
        self.cfg = cfg or SoccerEnvCfg()
        self.device = device or self.cfg.device
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        e = self.cfg.num_worlds
        n = self.cfg.n_ducks
        if self.cfg.njmax is None:
            self.cfg.njmax = max(400, int(round(P.NJMAX * n / P.NUM_DUCKS)))
        if self.cfg.nconmax is None:
            self.cfg.nconmax = max(300, int(round(P.NCONMAX * n / P.NUM_DUCKS)))
        self.n_ducks = n
        self.n_team = self.cfg.n_team
        self.num_worlds = e
        self.num_envs = e * n
        self.num_actions = N_JOINTS
        self.max_episode_length = self.cfg.max_episode_length
        self.step_dt = self.cfg.timestep * self.cfg.decimation
        self.physics_dt = self.cfg.timestep

        scene_cfg = SceneCfg(
            num_envs=e,
            env_spacing=10.0,
            terrain=TerrainEntityCfg(
                terrain_type="plane",
                textures=(P.PITCH_TERRAIN_TEXTURE,),
                materials=(P.PITCH_TERRAIN_MATERIAL,),
            ),
            entities=S.scene_entity_cfgs(self.cfg.n_team),
        )
        self.scene = Scene(scene_cfg, device=self.device)
        self.mj_model = self.scene.compile()
        S.verify_world(self.mj_model, self.cfg.n_team)

        self.sim = Simulation(
            num_envs=e,
            cfg=SimulationCfg(
                njmax=self.cfg.njmax,
                nconmax=self.cfg.nconmax,
                mujoco=MujocoCfg(timestep=self.cfg.timestep),
            ),
            model=self.mj_model,
            device=self.device,
        )
        self.scene.initialize(
            mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data
        )
        # REQUIRED by BAM: per-env friction arrays. This is what the
        # `expand_bam_friction_fields` startup event does in manager-based envs.
        self.sim.expand_model_fields(("dof_frictionloss", "dof_damping"))

        # Entity handles.
        self.ducks = [self.scene[S.duck_entity_name(i)] for i in range(n)]
        self.ball = self.scene["ball"]
        self.pitch = self.scene["pitch"]

        dev = self.device
        self.team_sign = OBS.team_attack_sign(n, self.cfg.n_team, dev)
        self.role = OBS.role_onehot(n, self.cfg.n_team, dev)
        self.own_goal_pos = torch.zeros(n, 3, device=dev)
        self.own_goal_pos[:, 0] = -self.team_sign * P.PITCH_L / 2
        self.own_goal_pos[:, 2] = P.GOAL_H / 2
        self.target_goal_xy = torch.tensor(
            [P.PITCH_L / 2, 0.0], device=dev
        )  # the goal team 0 attacks

        # Default joint pose (servo order) — the action is an offset from HOME.
        self.default_joint_pos = self.ducks[0].data.default_joint_pos[0].clone()
        assert self.default_joint_pos.shape == (N_JOINTS,)

        # Per-episode / per-step state.
        zero_e = torch.zeros(e, 1, device=dev)
        self.episode_length_buf = torch.zeros(e, dtype=torch.long, device=dev)
        self.common_step_counter = 0
        self.last_action = torch.zeros(e, n, N_JOINTS, device=dev)
        self.prev_ball_xy = torch.zeros(e, 2, device=dev)
        self.prev_team_min_dist = torch.zeros(e, 2, device=dev)
        self.goal_side = torch.zeros(e, dtype=torch.long, device=dev)
        self.last_touch_team0 = torch.zeros(e, device=dev)
        self.since_touch = torch.full((e, n), 1e9, device=dev)
        self.no_progress_start_xy = torch.zeros(e, 2, device=dev)
        self.no_progress_time = torch.zeros(e, device=dev)
        self.touched_in_window = torch.zeros(e, dtype=torch.bool, device=dev)
        self.scores = torch.zeros(e, 2, dtype=torch.long, device=dev)
        self.episode_return = torch.zeros(e, n, device=dev)
        self.term_reason = {
            "goal": torch.zeros(e, dtype=torch.bool, device=dev),
            "no_progress": torch.zeros(e, dtype=torch.bool, device=dev),
            "nan": torch.zeros(e, dtype=torch.bool, device=dev),
            "timeout": torch.zeros(e, dtype=torch.bool, device=dev),
        }
        self.reward_terms: dict[str, torch.Tensor] = {}
        self.nan_reward_steps: dict[str, int] = {}
        self.obs = None
        self.reset()

    # ── Public API (lane space) ──────────────────────────────────────────────
    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        self._reset_idx(
            torch.arange(self.num_worlds, device=self.device) if env_ids is None else env_ids
        )
        self._refresh()
        self.obs = self._compute_obs()
        return self.obs

    def step(self, actions: torch.Tensor):
        a = actions.reshape(self.num_worlds, self.n_ducks, N_JOINTS)
        if self.cfg.clip_actions is not None:
            a = a.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        self.last_action = a

        target = self.default_joint_pos.view(1, 1, -1) + a * self.cfg.action_scale
        for i, duck in enumerate(self.ducks):
            duck.set_joint_position_target(target[:, i, :])

        for _ in range(self.cfg.decimation):
            for duck in self.ducks:
                duck.write_data_to_sim()
            self.sim.step()
            for duck in self.ducks:
                duck.update(self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        rew = self._compute_rewards()
        terminated, truncated = self._compute_terminations()
        # Compute `done` BEFORE resetting: `_reset_idx` clears the per-reason
        # flags, and `truncated` is a tensor that the flags may alias. Building
        # `done` first makes the step ordering explicit and alias-proof (an
        # aliased version silently reported done=False for timeouts, which kills
        # rsl_rl's `extras["time_outs"]` bootstrap masking).
        done = (terminated | truncated).repeat_interleave(self.n_ducks)
        # Build `extras` BEFORE the reset: `_reset_idx` clears the per-reason
        # masks, and an evaluator needs to know WHY an episode ended.
        extras = {
            "terminated": terminated.repeat_interleave(self.n_ducks),
            "truncated": truncated.repeat_interleave(self.n_ducks),
            "time_outs": truncated.repeat_interleave(self.n_ducks),
            "term_reason": {
                k: v.repeat_interleave(self.n_ducks) for k, v in self.term_reason.items()
            },
            "Episode_Reward": {k: v.mean().item() for k, v in self.reward_terms.items()},
            "nan_reward_terms": dict(self.nan_reward_steps),
        }
        done_worlds = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(done_worlds):
            self._reset_idx(done_worlds)
            # Only a reset needs an explicit forward: freshly written qpos would
            # otherwise be unusable until the next mj_step.
            self._refresh()

        self.obs = self._compute_obs()
        return self.obs, rew.reshape(self.num_envs), done, extras

    # ── Physics refresh ──────────────────────────────────────────────────────
    def _refresh(self) -> None:
        """Recompute derived quantities from qpos/qvel.

        Called after a RESET only. On a normal step the derived arrays lag
        qpos/qvel by one 5 ms physics substep — mjlab's own env documents this lag
        as consistent and therefore harmless (every env, every step sees exactly
        the same one-substep lag), and calling ``sim.forward()`` every control
        step costs ~30 ms, i.e. ~8 % of the whole step at 512 worlds.
        """
        self.sim.forward()

    # ── Reset ────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        e = len(env_ids)
        gen = None

        # Ducks: formation + jitter, HOME joints, zero velocity.
        poses = EV.respawn_within_pitch(
            EV.spawn_poses(e, generator=gen, device=self.device, n_team=self.cfg.n_team)
        )
        assert poses.shape[1] == self.n_ducks, (
            f"spawn produced {poses.shape[1]} ducks for a {self.n_ducks}-duck world"
        )
        yaw = torch.tensor(
            [y for _, _, y in S.formation(self.cfg.n_team)],
            device=self.device, dtype=torch.float32,
        )
        quat = torch.stack(
            (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)),
            dim=-1,
        ).expand(e, -1, -1)
        for i in range(self.n_ducks):
            root = torch.zeros(e, 13, device=self.device)
            root[:, 0:3] = poses[:, i, :]
            root[:, 3:7] = quat[:, i, :]
            self.ducks[i].write_root_state_to_sim(root, env_ids=env_ids)
            jp = self.default_joint_pos.view(1, -1).expand(e, -1)
            self.ducks[i].write_joint_state_to_sim(
                jp, torch.zeros_like(jp), env_ids=env_ids
            )
            self.ducks[i].reset(env_ids)      # clear BAM internal state

        # Ball: centre-circle patch.
        ball_pos = EV.reset_ball_pos(e, generator=gen, device=self.device)
        ball_root = torch.zeros(e, 13, device=self.device)
        ball_root[:, 0:3] = ball_pos
        ball_root[:, 3] = 1.0
        self.ball.write_root_state_to_sim(ball_root, env_ids=env_ids)
        self.ball.reset(env_ids)

        # Episode state.
        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.prev_ball_xy[env_ids] = ball_pos[:, :2]
        self.prev_team_min_dist[env_ids] = RW.team_min_ball_distance(
            torch.stack(
                [self.ducks[i].data.root_link_pos_w[env_ids] for i in range(self.n_ducks)],
                dim=1,
            ),
            ball_pos,
            self.cfg.n_team,
        )
        self.goal_side[env_ids] = 0
        self.since_touch[env_ids] = 1e9
        self.no_progress_start_xy[env_ids] = ball_pos[:, :2]
        self.no_progress_time[env_ids] = 0.0
        self.touched_in_window[env_ids] = False
        self.scores[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        for v in self.term_reason.values():
            v[env_ids] = False

        # Who "has the ball" at kickoff (seeds the last_touch observation so
        # both branches of that one-hot are exercised from step 0).
        duck_pos = torch.stack(
            [self.ducks[i].data.root_link_pos_w[env_ids] for i in range(self.n_ducks)],
            dim=1,
        )
        self.last_touch_team0[env_ids] = EV.possession(ball_pos, duck_pos, self.team_sign)

    # ── Signals ──────────────────────────────────────────────────────────────
    def _stack(self, attr: str) -> torch.Tensor:
        """(E, 3) per entity -> (E, N, 3)."""
        return torch.stack([getattr(d.data, attr) for d in self.ducks], dim=1)

    def _ball_pos(self) -> torch.Tensor:
        return self.ball.data.root_link_pos_w

    def _ball_vel(self) -> torch.Tensor:
        return self.ball.data.root_link_lin_vel_w

    def _touch_mask(self, ball_pos: torch.Tensor, duck_pos: torch.Tensor) -> torch.Tensor:
        d = (ball_pos[:, None, :2] - duck_pos[..., :2]).norm(dim=-1)
        return d < self.cfg.touch_radius

    def _compute_obs(self) -> dict[str, torch.Tensor]:
        duck_pos = self._stack("root_link_pos_w")
        quat = self._stack("root_link_quat_w")
        lin_vel = self._stack("root_link_lin_vel_w")
        joint_pos = torch.stack([d.data.joint_pos for d in self.ducks], dim=1)
        joint_vel = torch.stack([d.data.joint_vel for d in self.ducks], dim=1)
        ang_vel_b = torch.stack([d.data.root_link_ang_vel_b for d in self.ducks], dim=1)
        grav_b = torch.stack([d.data.projected_gravity_b for d in self.ducks], dim=1)

        e = self.num_worlds
        ball_pos = self._ball_pos()
        time_rem = 1.0 - self.episode_length_buf.float() / max(self.max_episode_length, 1)
        goal_diff = (self.scores[:, 0] - self.scores[:, 1]).float()

        actor = OBS.build_obs(
            root_pos_w=duck_pos,
            root_quat_w=quat,
            root_lin_vel_w=lin_vel,
            joint_pos_rel=joint_pos - self.default_joint_pos.view(1, 1, -1),
            joint_vel_rel=joint_vel,
            base_ang_vel_b=ang_vel_b,
            projected_gravity_b=grav_b,
            last_action=self.last_action,
            ball_pos_w=ball_pos,
            ball_vel_w=self._ball_vel(),
            own_goal_pos_w=self.own_goal_pos,
            role=self.role,
            time_remaining=time_rem.clamp(0.0, 1.0),
            goal_diff=goal_diff,
            last_touch=self.last_touch_team0,
            n_team=self.cfg.n_team,
        )
        if self.cfg.obs_noise_std > 0:
            actor = actor + torch.randn_like(actor) * self.cfg.obs_noise_std

        # Critic-only global block (12 dims): ball world state, ball speed,
        # score, clock, and the two in-goal flags.
        ball_v = self._ball_vel()
        extra = torch.cat(
            (
                ball_pos,
                ball_v,
                ball_v.norm(dim=-1, keepdim=True),
                (self.scores[:, 0] - self.scores[:, 1]).float().view(e, 1),
                self.episode_length_buf.float().view(e, 1) / max(self.max_episode_length, 1),
                self.last_touch_team0.view(e, 1),
            ),
            dim=-1,
        )
        assert extra.shape[-1] == self.cfg.critic_extra_dim, extra.shape
        critic = torch.cat(
            (actor, extra.unsqueeze(1).expand(-1, self.n_ducks, -1)), dim=-1
        )
        lanes = e * self.n_ducks
        return {"actor": actor.reshape(lanes, -1), "critic": critic.reshape(lanes, -1)}

    def _compute_rewards(self) -> torch.Tensor:
        e = self.num_worlds
        ball_pos = self._ball_pos()
        duck_pos = self._stack("root_link_pos_w")
        grav_b = torch.stack([d.data.projected_gravity_b for d in self.ducks], dim=1)

        # ── Touch bookkeeping (also feeds the observation) ───────────────────
        touched = self._touch_mask(ball_pos, duck_pos)               # (E, N)
        any_touch = touched.any(dim=-1)
        team0_touch = touched[:, : self.cfg.n_team].any(dim=-1)
        team1_touch = touched[:, self.cfg.n_team:].any(dim=-1)
        self.last_touch_team0 = torch.where(
            team0_touch & ~team1_touch, torch.ones_like(self.last_touch_team0),
            torch.where(
                team1_touch & ~team0_touch, torch.zeros_like(self.last_touch_team0),
                self.last_touch_team0,
            ),
        )
        self.touched_in_window = self.touched_in_window | any_touch

        terms: dict[str, torch.Tensor] = {}
        terms["touch"] = RW.touch_reward(touched, self.since_touch) * RW.TOUCH_WEIGHT
        self.since_touch = torch.where(
            (touched & (self.since_touch >= RW.TOUCH_COOLDOWN_S)),
            torch.zeros_like(self.since_touch),
            self.since_touch + self.step_dt,
        )

        # ── Ball progress (potential difference, unfarmable) ─────────────────
        terms["ball_goal_progress"] = (
            RW.ball_goal_progress(
                ball_pos, self.prev_ball_xy, self.target_goal_xy, self.team_sign
            )
            * RW.BALL_PROGRESS_WEIGHT
        )

        # ── Duck -> ball approach (S0's enabling gradient) ───────────────────
        terms["ball_approach"] = (
            RW.ball_approach_progress(
                duck_pos, ball_pos, self.prev_team_min_dist, self.cfg.n_team
            )
            * RW.BALL_APPROACH_WEIGHT
        )
        self.prev_team_min_dist = RW.team_min_ball_distance(
            duck_pos, ball_pos, self.cfg.n_team
        )

        # ── Goals (rising edge; the ball is trapped by the solid net) ────────
        side = RW.ball_goal_side(ball_pos)
        events = RW.goal_events(self.goal_side, side)
        self.goal_side = side
        self.scores[:, 0] += (events > 0).long()
        self.scores[:, 1] += (events < 0).long()
        terms["goal_event"] = (
            RW.goal_event_reward(events, self.team_sign) * RW.GOAL_WEIGHT
        )

        # ── Penalties (all <= 0, positive weights) ───────────────────────────
        terms["teammate_spacing"] = (
            RW.teammate_spacing_penalty(duck_pos, self.team_sign) * RW.SPACING_WEIGHT
        )
        terms["fallen"] = RW.fallen_penalty(grav_b) * RW.FALLEN_WEIGHT

        # No-progress watchdog.
        self.no_progress_time += self.step_dt
        moved = (ball_pos[..., :2] - self.no_progress_start_xy).norm(dim=-1)
        window_done = self.no_progress_time >= RW.NO_PROGRESS_WINDOW_S
        self._stalemate = window_done & (moved < RW.NO_PROGRESS_MIN_MOVE) & ~self.touched_in_window
        terms["no_progress"] = (
            RW.no_progress_penalty(
                ball_pos,
                self.no_progress_start_xy,
                self.no_progress_time,
                self.touched_in_window,
                self.n_ducks,
            )
            * RW.NO_PROGRESS_WEIGHT
        )
        # Roll the window whenever it closes (progress or not).
        roll = window_done
        if roll.any():
            self.no_progress_start_xy[roll] = ball_pos[roll, :2]
            self.no_progress_time[roll] = 0.0
            self.touched_in_window[roll] = False

        # NaN guard. MuJoCo's contact solver can produce one divergent step
        # before `nan_state` terminates the episode; rsl_rl aborts the WHOLE run
        # on a single NaN reward, so scrub per term and count it (repo practice:
        # every microduck reward function is nan_to_num-safe).
        for k, v in terms.items():
            if not torch.isfinite(v).all():
                self.nan_reward_steps[k] = self.nan_reward_steps.get(k, 0) + 1
                terms[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        RW.assert_penalty_signs(terms)

        total = torch.stack(tuple(terms.values()), dim=0).sum(0)     # (N_terms, E, N)
        self.reward_terms = {k: v.detach() for k, v in terms.items()}
        self.episode_return += total
        self.prev_ball_xy = ball_pos[..., :2].clone()
        del e
        return total

    def _compute_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        bad = torch.isnan(qpos).any(dim=-1) | torch.isnan(qvel).any(dim=-1)
        nan_state = bad.any(dim=-1) if bad.dim() > 1 else bad

        scored = (self.goal_side != 0)
        timed_out = self.episode_length_buf >= self.max_episode_length
        stalemate = getattr(self, "_stalemate", torch.zeros_like(timed_out))

        terminated = (scored | stalemate | nan_state) & ~timed_out
        truncated = timed_out.clone()
        # Diagnostics only — store CLONES so `_reset_idx` clearing them cannot
        # mutate the tensors the caller is about to use.
        self.term_reason["goal"] = scored.clone()
        self.term_reason["no_progress"] = stalemate.clone()
        self.term_reason["nan"] = nan_state.clone()
        self.term_reason["timeout"] = timed_out.clone()
        return terminated, truncated

    # ── Misc ─────────────────────────────────────────────────────────────────
    @property
    def action_dim(self) -> int:
        return N_JOINTS

    def close(self) -> None:
        pass
