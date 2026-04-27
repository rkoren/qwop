"""
Training with some gait rewards.

Adds bonuses on top of distance:

  alt     — rewards actually running (one foot up as the other go down)
  stride  — rewards leg separation (left/right thigh angle difference)
  lean    — rewards forward torso lean
  flight  — rewards foot height alternation (one foot off the ground)

for now bonuses are small relative to distance

Usage:
    python train_gait.py

From an existing model:
    python train_gait.py --load models/phase1/phase1_final.zip

Monitor with TensorBoard:
  tensorboard --logdir logs/
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from qwop_env import QWOPEnv
from train import EpisodeStatsCallback

TOTAL_STEPS = 7_000_000  # more steps — gait shaping makes the problem harder

# Observation layout: 12 parts × 5 values [pos_x, pos_y, angle, vel_x, vel_y]
# Parts: torso(0) head(1) l_arm(2) r_arm(3) l_forearm(4) r_forearm(5)
#        l_thigh(6) r_thigh(7) l_calf(8) r_calf(9) l_foot(10) r_foot(11)
_TORSO_ANGLE  = 2
_L_THIGH_ANGLE = 6 * 5 + 2   # 32
_R_THIGH_ANGLE = 7 * 5 + 2   # 37
_L_FOOT_Y      = 10 * 5 + 1  # 51
_R_FOOT_Y      = 11 * 5 + 1  # 56
_L_FOOT_VEL_Y  = 10 * 5 + 4  # 54
_R_FOOT_VEL_Y  = 11 * 5 + 4  # 59


class GaitQWOPEnv(QWOPEnv):
    """QWOPEnv with gait-shaping bonuses layered on top of the distance reward."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward += _gait_bonus(obs)
        return obs, reward, terminated, truncated, info


def _gait_bonus(obs: np.ndarray) -> float:
    # Stride: reward angular separation between the two thighs
    stride = abs(float(obs[_L_THIGH_ANGLE]) - float(obs[_R_THIGH_ANGLE]))

    # Lean: reward forward torso lean (positive normalized angle = leaning toward finish)
    lean = max(0.0, float(obs[_TORSO_ANGLE]))

    # Flight: reward when feet are at different heights (one lifted = actual stride)
    flight = abs(float(obs[_L_FOOT_Y]) - float(obs[_R_FOOT_Y]))

    # Alternation: reward anti-phase foot swing — one foot rising as the other falls.
    # Product of vertical velocities is negative when they oppose each other (true running),
    # and positive when they move in unison (scoot/hop). Weighted highest as the core
    # mechanical signature of a running gait.
    alt = max(0.0, -(float(obs[_L_FOOT_VEL_Y]) * float(obs[_R_FOOT_VEL_Y])))

    return 0.01 * stride + 0.005 * lean + 0.01 * flight + 0.03 * alt


def train(args: argparse.Namespace) -> None:
    save_dir = "models/gait"
    os.makedirs(save_dir, exist_ok=True)

    env = GaitQWOPEnv(phase="distance", render_mode="browser")

    if args.load:
        print(f"Loading model from {args.load} ...")
        model = PPO.load(args.load, env=env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log="logs/",
            policy_kwargs={"net_arch": [256, 256]},
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            learning_rate=3e-4,
            ent_coef=0.01,
        )

    callbacks = [
        EpisodeStatsCallback(),
        CheckpointCallback(
            save_freq=100_000,
            save_path=save_dir,
            name_prefix="ppo_qwop_gait",
            verbose=1,
        ),
    ]

    print(f"Gait training: {TOTAL_STEPS:,} steps ...")
    print(f"Checkpoints: {save_dir}/")
    print("TensorBoard: tensorboard --logdir logs/")
    print()

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.load is None),
    )

    final_path = os.path.join(save_dir, "gait_final")
    model.save(final_path)
    print(f"\nTraining complete. Model saved: {final_path}.zip")
    print("\nTo watch the result:")
    print(f"  python play.py --model {final_path}.zip --fps 60")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on QWOP with gait shaping")
    parser.add_argument(
        "--load", default=None, metavar="MODEL_PATH",
        help="Path to a saved .zip model to resume or fine-tune",
    )
    train(parser.parse_args())
