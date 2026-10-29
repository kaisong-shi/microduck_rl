#!/usr/bin/env python
"""Headless evaluation of a soccer checkpoint (P6-lite).

    python scripts/soccer_eval.py --checkpoint logs/soccer/S0/.../model_300.pt \
        --n-team 1 --episodes 256

Reports what the rollouts ACTUALLY show, per the repo rule "measure before
theorising": ball displacement, touches, goals, falls, and the done reasons.
No video, no viewer — a table you can compare across checkpoints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.rl import MjlabOnPolicyRunner

from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer.rl_cfg import SoccerRlCfg
from mjlab_microduck.soccer.vecenv import make_soccer_env


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-team", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=256,
                    help="parallel matches to run (all start together)")
    ap.add_argument("--max-control-steps", type=int, default=None,
                    help="cap per episode (default: the env's own limit)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of the mean")
    ap.add_argument("--out", default=None, help="write the metrics as JSON here")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    env = make_soccer_env(num_worlds=args.episodes, n_team=args.n_team,
                          device=args.device, seed=args.seed)
    agent_cfg = asdict(SoccerRlCfg)
    runner = MjlabOnPolicyRunner(env, agent_cfg, None, args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)
    policy.eval()

    e = env.unwrapped
    n = e.n_ducks
    worlds = e.num_worlds
    limit = args.max_control_steps or e.max_episode_length

    # The actor takes the whole observation TensorDict (it selects its
    # obs_groups itself), not the raw "actor" tensor.
    obs = env.get_observations()
    prev_ball_xy = e.ball.data.root_link_pos_w[:, :2].clone()

    acc = {
        "ball_travel": torch.zeros(worlds, device=e.device),
        "touches": torch.zeros(worlds, device=e.device),
        "fall_s": torch.zeros(worlds, device=e.device),
        "steps": torch.zeros(worlds, device=e.device),
        "min_ball_dist": torch.full((worlds,), float("inf"), device=e.device),
    }
    first_touch = torch.full((worlds,), float("nan"), device=e.device)
    totals = {"episodes": 0, "goals": 0, "stalemates": 0, "nans": 0,
              "timeouts": 0, "ball_travel": 0.0, "touches": 0.0,
              "fall_s": 0.0, "elapsed_s": 0.0, "min_ball_dist": [],
              "unexplained": 0}
    first_touch_list: list[float] = []

    with torch.no_grad():
        for _ in range(limit):
            actions = policy(obs)
            obs, _rew, done, extras = env.step(actions)
            done_w = done.reshape(worlds, n)[:, 0].bool()

            ball_xy = e.ball.data.root_link_pos_w[:, :2]
            acc["ball_travel"] += (ball_xy - prev_ball_xy).norm(dim=-1)
            prev_ball_xy = ball_xy.clone()

            touched = (e.reward_terms["touch"] > 0).any(dim=-1)
            acc["touches"] += touched.float()
            new = touched & torch.isnan(first_touch)
            first_touch[new] = e.episode_length_buf.float()[new] * e.step_dt
            acc["fall_s"] += (e.reward_terms["fallen"] < -1e-6).any(dim=-1).float() * e.step_dt
            acc["steps"] += 1
            # Closest any duck got to the ball during this episode.
            d = (e.ball.data.root_link_pos_w[:, None, :2]
                 - torch.stack([dk.data.root_link_pos_w[..., :2] for dk in e.ducks], 1)).norm(dim=-1)
            acc["min_ball_dist"] = torch.minimum(acc["min_ball_dist"], d.min(dim=-1).values)

            if done_w.any():
                idx = done_w.nonzero(as_tuple=False).squeeze(-1)
                # term_reason is per LANE; collapse to one row per world first.
                reason = {
                    k: v.reshape(-1, n)[:, 0] for k, v in extras["term_reason"].items()
                }
                totals["episodes"] += int(len(idx))
                lane0 = lambda key: reason[key][idx]
                totals["goals"] += int(lane0("goal").sum().item())
                totals["stalemates"] += int(lane0("no_progress").sum().item())
                totals["nans"] += int(lane0("nan").sum().item())
                totals["timeouts"] += int(lane0("timeout").sum().item())
                totals["ball_travel"] += acc["ball_travel"][idx].sum().item()
                totals["touches"] += acc["touches"][idx].sum().item()
                totals["fall_s"] += acc["fall_s"][idx].sum().item()
                # NOTE: read the length from OUR counter, not from
                # episode_length_buf — the env already reset it to 0 for these
                # worlds (that mistake reported every episode as 0.02 s).
                totals["elapsed_s"] += float((acc["steps"][idx] * e.step_dt).sum().item())
                totals["min_ball_dist"].extend(acc["min_ball_dist"][idx].tolist())
                flagged = (
                    lane0("goal") | lane0("no_progress")
                    | lane0("nan") | lane0("timeout")
                )
                totals["unexplained"] += int((~flagged).sum().item())
                fin = first_touch[idx]
                first_touch_list.extend(fin[~torch.isnan(fin)].tolist())
                acc["ball_travel"][idx] = 0
                acc["touches"][idx] = 0
                acc["fall_s"][idx] = 0
                acc["steps"][idx] = 0
                acc["min_ball_dist"][idx] = float("inf")
                first_touch[idx] = float("nan")

    ep = max(totals["episodes"], 1)
    ft = sorted(first_touch_list)
    out = {
        "checkpoint": str(args.checkpoint),
        "n_team": args.n_team,
        "episodes": totals["episodes"],
        "goals": totals["goals"],
        "goals_per_episode": round(totals["goals"] / ep, 3),
        "ball_travel_m_per_episode": round(totals["ball_travel"] / ep, 3),
        "touches_per_episode": round(totals["touches"] / ep, 2),
        "episodes_with_a_touch": round(len(ft) / ep, 3),
        "first_touch_s_median": round(ft[len(ft) // 2], 2) if ft else None,
        "fall_frac_of_time": round(totals["fall_s"] / max(totals["elapsed_s"], 1e-6), 3),
        "episode_len_s": round(totals["elapsed_s"] / ep, 2),
        "min_duck_ball_dist_m": round(
            sum(totals["min_ball_dist"]) / max(len(totals["min_ball_dist"]), 1), 3
        ),
        "unexplained_ends": totals["unexplained"],
        "end_reasons_per_episode": {
            "goal": round(totals["goals"] / ep, 3),
            "stalemate": round(totals["stalemates"] / ep, 3),
            "nan": round(totals["nans"] / ep, 3),
            "timeout": round(totals["timeouts"] / ep, 3),
        },
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
