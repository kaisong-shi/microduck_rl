"""PPO configuration for the soccer task.

Hyper-parameters that differ from the repo's walking recipe, and why:
  * ``gamma = 0.995`` (walking uses 0.99). Soccer reward is sparse-ish: the only
    dense task signal is ball progress, and a goal is 50x any single step. At
    50 Hz, 0.99 gives an effective horizon of ~2 s — too short to value "kick it
    now, score in 4 s". 0.995 is ~4 s, which matches one kick -> roll -> arrival
    arc.
  * ``num_steps_per_env = 24`` (same as the repo). 11264 lanes * 24 = 270k
    transitions per iteration.
  * ``entropy_coef = 0.003`` (walking recipe: 0.01). Measured on S0, 300
    iterations, everything else equal:

        entropy  init_std   std@300   touch%   ball m/ep   fall%   goals/ep
        0.01     1.0        1.92      98.5     4.35        40.0    0
        0.003    1.0        1.03      100.0    3.69        19.3    0.046
        0.01     0.5        1.18      100.0    1.64        14.2    0
        0.003    0.5        0.46      100.0    4.61        19.4    0

    The 0.01 default walks the policy std up (1.92 -> 4.19 by iteration 800)
    until ~80 % of samples hit the +-1 action clamp: the log-prob PPO trains on
    then no longer matches the action the env executed, and (empirically) the
    ducks spend 40 % of the time past the 60 deg fall threshold. 0.003 keeps the
    std near 1 (full-range actions, real exploration) and roughly halves the
    falling. ``init_std`` is left at 1.0: 0.5 explores too narrowly and the
    coupled variant (0.003 + 0.5) drops the std to 0.46, which is a poor place to
    start a curriculum that later needs 2 v 2 and 11 v 11.
  * No symmetry augmentation: the 97-dim observation is already
    canonicalised per team (see ``obs.py``), so the mirror trick would be
    redundant at best and wrong for the asymmetric role one-hot at worst.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

GAMMA = 0.995
LAM = 0.95
NUM_STEPS_PER_ENV = 24
LEARNING_RATE = 3.0e-4
"""3e-4 with 270k transitions/iteration is the stable end of the range; the
walking recipe's 1e-3 + adaptive KL is tuned for 1 duck per world and a much
denser reward."""

SoccerRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=LEARNING_RATE,
        schedule="fixed",
        gamma=GAMMA,
        lam=LAM,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="soccer_11v11",
    run_name="soccer_11v11",
    save_interval=100,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=10_000,
)
