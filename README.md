# QWOP RL

Reinforcement learning agent to play [QWOP](https://www.foddy.net/Athletics.html)
Uses [PPO](https://stable-baselines3.readthedocs.io) and the [qwop-gym](https://github.com/smanolloff/qwop-gym) environment, to play the game in browser game and view the runner's physics state.

Each step it chooses one of 16 key combinations from **Q**, **W**, **O**, **P** and receives a reward based on forward progress.

**Two-phase training:**
- **Phase 1** — reward = meters gained, so trying to reach 100m without falling.
- **Phase 2** — reward = meters gained minus a time cost, so trying to run faster.

## Setup

**Requirements:** Python 3.10+, [Google Chrome](https://www.google.com/chrome/), [chromedriver](https://googlechromelabs.github.io/chrome-for-testing/)

```bash
# Install chromedriver (macOS)
brew install chromedriver

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download and patch the QWOP game (one-time)
python setup.py
```

## Training

```bash
# Phase 1 — learn to finish the race (~3–10M steps, several hours)
python train.py

# Phase 2 — fine-tune for speed
python train.py --phase 2 --load models/phase1/phase1_final.zip

# Gait training — reward alternating foot contacts scaled by distance
python train_gait.py

# Resume an interrupted run
python train.py --load models/phase1/ppo_qwop_1000000_steps.zip
python train_gait.py --load models/gait/ppo_qwop_gait_1000000_steps.zip
```

Progress is printed per episode and logged to TensorBoard:
```
step   198,144 | fell      |  34.7 m | 22.1 s
step   200,192 | FINISHED  | 100.0 m | 67.3 s
```

```bash
tensorboard --logdir logs/
```

Checkpoints are saved to `models/phase1/` every 100k steps.

## Watch the agent play

```bash
python play.py --model models/phase1/phase1_final.zip
python play.py --model models/phase2/phase2_final.zip --episodes 10
```

## Files

| File | Purpose |
|------|---------|
| `setup.py` | One-time setup: verify Chrome/chromedriver, patch QWOP |
| `qwop_env.py` | Gymnasium environment wrapper with reward shaping |
| `train.py` | PPO training with episode logging and checkpoints |
| `train_gait.py` | Gait training with alternating foot-contact reward |
| `play.py` | Load a saved model and watch it play |
