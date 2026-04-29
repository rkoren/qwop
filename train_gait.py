"""
Training with alternating foot reward.

Give a bonus each time a foot makes contact with the ground (foot_y >= 8)
and it's the opposite foot from the last contact. The bonus scales linearly
with current distance.

Usage:
    python train_gait.py

From an existing model:
    python train_gait.py --load models/phase1/phase1_final.zip

Monitor with TensorBoard:
  tensorboard --logdir logs/
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from qwop_env import QWOPEnv
from train import EpisodeStatsCallback

TOTAL_STEPS = 7_000_000  # more steps — gait shaping makes the problem harder

# Observation layout: 12 parts × 5 values [pos_x, pos_y, angle, vel_x, vel_y]
# Parts: torso(0) head(1) l_arm(2) r_arm(3) l_forearm(4) r_forearm(5)
#        l_thigh(6) r_thigh(7) l_calf(8) r_calf(9) l_foot(10) r_foot(11)
_TORSO_X  = 0
_L_FOOT_Y = 10 * 5 + 1  # 51
_R_FOOT_Y = 11 * 5 + 1  # 56

GROUND_Y = 8.0   # foot_y >= this → on the ground (confirmed from replay observation)
ALT_BASE = 0.5   # bonus per alternating contact at distance 0
ALT_SCALE = 0.1  # additional bonus per metre of distance


class GaitQWOPEnv(QWOPEnv):
    """Env that rewards alternating foot contacts, scaled by distance."""

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_foot: str | None = None  # 'L' or 'R'
        self._l_was_down = float(obs[_L_FOOT_Y]) >= GROUND_Y
        self._r_was_down = float(obs[_R_FOOT_Y]) >= GROUND_Y
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        l_down = float(obs[_L_FOOT_Y]) >= GROUND_Y
        r_down = float(obs[_R_FOOT_Y]) >= GROUND_Y

        l_contact = l_down and not self._l_was_down
        r_contact = r_down and not self._r_was_down

        distance = max(0.0, float(obs[_TORSO_X]))
        step_value = ALT_BASE + ALT_SCALE * distance

        if l_contact:
            if self._last_foot == 'R':
                reward += step_value
            self._last_foot = 'L'
        if r_contact:
            if self._last_foot == 'L':
                reward += step_value
            self._last_foot = 'R'

        self._l_was_down = l_down
        self._r_was_down = r_down

        return obs, reward, terminated, truncated, info


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
