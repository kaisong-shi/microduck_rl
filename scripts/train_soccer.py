#!/usr/bin/env python
"""Train the 22-duck (11 v 11) soccer policy.

    python scripts/train_soccer.py --num-worlds 512 --iterations 3000
    python scripts/train_soccer.py --num-worlds 8 --iterations 5      # smoke

Why a standalone script and not `uv run train <TASK_ID>`: mjlab's task registry
is built around ``ManagerBasedRlEnv`` (one robot per world, one action term, one
observation group). The soccer env is a multi-agent world with its own step loop
(see ``soccer/env.py`` for why), so it cannot be expressed as a task id without
pretending to be something it is not. Everything downstream — env wrapper, PPO,
runner, checkpointing — is the standard mjlab/rsl_rl stack.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.rl import MjlabOnPolicyRunner

from mjlab_microduck.soccer import pitch as P
from mjlab_microduck.soccer.rl_cfg import SoccerRlCfg
from mjlab_microduck.soccer.vecenv import make_soccer_env


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-worlds", type=int, default=P.NUM_WORLDS,
                    help="parallel matches (each is 22 agents); default 512")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override max_iterations")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override num_steps_per_env")
    ap.add_argument("--n-team", type=int, default=P.NUM_DUCKS_PER_TEAM,
                    help="ducks per team; 1 = S0 (1v1), 11 = full match. The "
                         "curriculum stages are S0 1, S1 2, S2 4, S3 6, S4 11")
    ap.add_argument("--experiment-name", default=None,
                    help="override the wandb/log experiment name (e.g. soccer_S0)")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="PPO entropy bonus. Actions are clamped to +-1, so a "
                         "Gaussian std that saturates the clamp (measured 4.19 "
                         "with the 0.01 default) makes the log-prob inconsistent "
                         "with what the env executed")
    ap.add_argument("--init-std", type=float, default=None,
                    help="initial action std of the Gaussian policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-dir", default=None,
                    help="default logs/soccer/<experiment>/<timestamp>")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path to resume from")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", choices=("online", "offline", "disabled"),
                    default="offline",
                    help="wandb mode; 'disabled' skips wandb entirely")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.wandb == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = args.wandb

    torch.manual_seed(args.seed)

    env = make_soccer_env(num_worlds=args.num_worlds, n_team=args.n_team,
                          device=args.device, seed=args.seed)
    cfg_env = env.unwrapped.cfg
    print(
        f"[soccer] {args.n_team} v {args.n_team}: {args.num_worlds} worlds x "
        f"{cfg_env.n_ducks} ducks = {env.num_envs} policy lanes, "
        f"{env.num_actions} actions each, episode {cfg_env.episode_s:.0f} s "
        f"({env.max_episode_length} control steps), "
        f"njmax={cfg_env.njmax} nconmax={cfg_env.nconmax}"
    )

    agent_cfg = asdict(SoccerRlCfg)
    if args.experiment_name:
        agent_cfg["experiment_name"] = args.experiment_name
        if not args.run_name:
            agent_cfg["run_name"] = args.experiment_name
    if args.iterations is not None:
        agent_cfg["max_iterations"] = args.iterations
    if args.num_steps is not None:
        agent_cfg["num_steps_per_env"] = args.num_steps
    if args.run_name:
        agent_cfg["run_name"] = args.run_name
    if args.entropy_coef is not None:
        agent_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
    if args.init_std is not None:
        agent_cfg["actor"]["distribution_cfg"]["init_std"] = args.init_std

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(
        args.log_dir
        or Path("logs") / "soccer" / agent_cfg["experiment_name"]
        / f"{stamp}_{agent_cfg['run_name']}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[soccer] log dir: {log_dir}")

    runner = MjlabOnPolicyRunner(env, agent_cfg, str(log_dir), args.device)
    if args.resume:
        print(f"[soccer] resuming from {args.resume}")
        runner.load(args.resume)

    runner.learn(
        num_learning_iterations=agent_cfg["max_iterations"],
        init_at_random_ep_len=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
