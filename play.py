"""
Watch a trained PPO agent play QWOP.

Usage:
  python play.py --model models/phase1/phase1_final.zip
  python play.py --model models/phase2/phase2_final.zip --episodes 10 --fps 30
"""

import argparse
import time

from stable_baselines3 import PPO

from qwop_env import QWOPEnv


def play(args: argparse.Namespace) -> None:
    env = QWOPEnv(render_mode="browser")
    model = PPO.load(args.model, env=env)

    print(f"Loaded model: {args.model}")
    print(f"Running {args.episodes} episode(s) at {args.fps} fps ...\n")

    step_delay = 1.0 / args.fps

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(step_delay)

        result = "FINISHED" if info.get("is_success") else "fell"
        dist = info.get("distance", 0.0)
        t = info.get("time", 0.0)
        speed = info.get("avgspeed", 0.0)
        print(f"Episode {ep:3d}: {result:8s} | {dist:6.1f} m | {t:6.1f} s | {speed:.2f} m/s")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained PPO agent play QWOP")
    parser.add_argument("--model", required=True, metavar="MODEL_PATH",
                        help="Path to saved model .zip file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to watch (default: 5)")
    parser.add_argument("--fps", type=float, default=60,
                        help="Playback speed in steps/sec — 60=real-time, 30=half-speed (default: 60)")
    play(parser.parse_args())
