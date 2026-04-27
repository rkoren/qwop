"""
Env to use qwop-gym (https://github.com/smanolloff/qwop-gym)
  - Body physics state from QWOP.min.js
  - Host the game locally via WebSocket
  - Returns body state via observations below

Observations (60 floats, normalized to [-1, 1]):
  12 body parts × [pos_x, pos_y, angle, vel_x, vel_y]
  Parts: torso, head, left_arm, right_arm, left_forearm, right_forearm,
         left_thigh, right_thigh, left_calf, right_calf, left_foot, right_foot

Keys:
  Q = left hip/shoulder  W = right hip/shoulder
  O = right knee         P = left knee

Actions:
  All combinations of the four QWOP keys.
  0=none, 1=Q, 2=W, 3=O, 4=P, 5=QW, 6=QO, 7=QP, 8=WO, 9=WP, 10=OP,
  11=QWO, 12=QWP, 13=QOP, 14=WOP, 15=QWOP

Reward modes:
  "distance"  Reward = meters gained each step. Train agent to get to 100m.
  "speed"     Reward = meters gained minus small time cost. Trains agent to run faster.
"""

import os
import shutil
import gymnasium as gym
import qwop_gym  # noqa: F401  registers "QWOP-v1" with gymnasium as a side effect


# Chrome-based browser paths to search on macOS
_MACOS_BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def find_browser() -> str:
    """Return path to a Chrome-based browser, or raise."""
    for path in _MACOS_BROWSERS:
        if os.path.exists(path):
            return path
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "No Chrome browser found. Install Google Chrome: https://www.google.com/chrome/"
    )


def find_chromedriver() -> str:
    """Return path to chromedriver, or raise."""
    found = shutil.which("chromedriver")
    if found:
        return found
    raise RuntimeError(
        "chromedriver not found.\n"
        "  macOS: brew install chromedriver\n"
        "  Or: https://googlechromelabs.github.io/chrome-for-testing/"
    )


class QWOPEnv(gym.Wrapper):
    """
    QWOP with selectable reward shaping for two-phase training.

    phase="distance"  Only reward forward progress — best for learning to finish.
    phase="speed"     Add a small time penalty — use after phase 1 to optimize pace.
    """

    def __init__(
        self,
        phase: str = "distance",
        browser: str = None,
        driver: str = None,
        render_mode: str = "browser",
    ):
        env = gym.make(
            "QWOP-v1",
            browser=browser or find_browser(),
            driver=driver or find_chromedriver(),
            render_mode=render_mode,
            stat_in_browser=(render_mode == "browser"),
            loglevel="INFO",
        )
        super().__init__(env)
        self.phase = phase
        self._distance = 0.0
        self._time = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._distance = info.get("distance", 0.0)
        self._time = info.get("time", 0.0)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(info, terminated)
        self._distance = info.get("distance", self._distance)
        self._time = info.get("time", self._time)
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info: dict, terminated: bool) -> float:
        distance_now = info.get("distance", self._distance)
        time_now = info.get("time", self._time)

        reward = distance_now - self._distance  # meters gained this step

        if self.phase == "speed":
            dt = time_now - self._time
            reward -= 0.001 * dt  # small time penalty to encourage faster running

        if terminated and not info.get("is_success", False):
            reward -= 1.0  # fell before reaching 100m

        return reward
