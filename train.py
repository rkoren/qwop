"""
Training an agent to play QWOP w/ PPO

Phase 1 (distance): Reward = forward progress only.
Phase 2 (speed): Reward = forward progress minus time cost.
- Load phase 1 weights and fine-tune for faster running.


Usage:
  Phase 1 (start fresh):
    python train.py

  Phase 2 (fine-tune for speed):
    python train.py --phase 2 --load models/phase1/phase1_final.zip

  Resume interrupted training:
    python train.py --load models/phase1/ppo_qwop_1000000_steps.zip

Monitor training with TensorBoard:
  tensorboard --logdir logs/
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from qwop_env import QWOPEnv


# Total timesteps per phase. Adjust up/down based on convergence.
PHASE_STEPS = {1: 5_000_000, 2: 2_000_000}


class EpisodeStatsCallback(BaseCallback):
    """
    Prints per-episode results to the console and logs them to TensorBoard.
    """

    def __init__(self):
        super().__init__(verbose=0)
        self._distances: list[float] = []
        self._successes: list[bool] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            dist = info.get("distance", 0.0)
            success = info.get("is_success", False)
            t = info.get("time", 0.0)
            self._distances.append(dist)
            self._successes.append(success)
            result = "FINISHED" if success else "fell"
            print(
                f"  step {self.num_timesteps:>9,} | {result:8s} | "
                f"{dist:5.1f} m | {t:5.1f} s"
            )
        return True

    def _on_rollout_end(self) -> None:
        if not self._distances:
            return
        self.logger.record("episode/mean_distance", np.mean(self._distances))
        self.logger.record("episode/success_rate", np.mean(self._successes))
        self._distances.clear()
        self._successes.clear()


def make_model(env: QWOPEnv) -> PPO:
    """Create a fresh PPO model with hyperparameters tuned for QWOP locomotion."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="logs/",
        # Larger network than default [64, 64] — QWOP needs capacity to learn gaits
        policy_kwargs={"net_arch": [256, 256]},
        # Rollout: collect 2048 steps, then update
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        # Discount and advantage estimation
        gamma=0.99,
        gae_lambda=0.95,
        # PPO clip — keep policy updates conservative
        clip_range=0.2,
        # Learning rate
        learning_rate=3e-4,
        # Entropy bonus — encourages exploration, important for discovering gaits
        ent_coef=0.01,
    )


def train(args: argparse.Namespace) -> None:
    phase_name = "distance" if args.phase == 1 else "speed"
    save_dir = f"models/phase{args.phase}"
    os.makedirs(save_dir, exist_ok=True)

    env = QWOPEnv(phase=phase_name, render_mode="browser")

    if args.load:
        print(f"Loading model from {args.load} ...")
        model = PPO.load(args.load, env=env)
    else:
        model = make_model(env)

    callbacks = [
        EpisodeStatsCallback(),
        CheckpointCallback(
            save_freq=100_000,
            save_path=save_dir,
            name_prefix="ppo_qwop",
            verbose=1,
        ),
    ]

    total_steps = PHASE_STEPS[args.phase]
    print(f"Phase {args.phase} ({phase_name}): training for {total_steps:,} steps ...")
    print(f"Checkpoints: {save_dir}/")
    print("TensorBoard: tensorboard --logdir logs/")
    print()

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.load is None),
    )

    final_path = os.path.join(save_dir, "phase{}_final".format(args.phase))
    model.save(final_path)
    print(f"\nTraining complete. Model saved: {final_path}.zip")

    if args.phase == 1:
        print("\nTo train phase 2 (speed optimization):")
        print(f"  python train.py --phase 2 --load {final_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on QWOP")
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Training phase: 1=finish race, 2=run faster (default: 1)",
    )
    parser.add_argument(
        "--load", default=None, metavar="MODEL_PATH",
        help="Path to a saved .zip model to resume or fine-tune",
    )
    train(parser.parse_args())
