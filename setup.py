"""
Run once before training:
    python setup.py
"""

import os
import shutil
import subprocess
import sys


MACOS_CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def check_browser() -> str:
    for path in MACOS_CHROME_PATHS:
        if os.path.exists(path):
            print(f"  browser: {path}")
            return path
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            print(f"  browser: {path}")
            return path
    print("ERROR: No Chrome-based browser found.")
    print("  Install Google Chrome: https://www.google.com/chrome/")
    sys.exit(1)


def check_chromedriver() -> str:
    path = shutil.which("chromedriver")
    if path:
        print(f"  chromedriver: {path}")
        return path
    print("ERROR: chromedriver not found.")
    print("  macOS: brew install chromedriver")
    print("  Or download: https://googlechromelabs.github.io/chrome-for-testing/")
    sys.exit(1)


def patch_qwop():
    """Download QWOP.min.js and apply RL instrumentation patch via qwop-gym CLI."""
    print("  Downloading and patching QWOP.min.js from foddy.net ...")
    result = subprocess.run(
        "curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch",
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR patching QWOP:\n{result.stderr or result.stdout}")
        print("Make sure you've run: pip install -r requirements.txt")
        sys.exit(1)
    print("  QWOP.min.js patched OK")


def main():
    print("=== QWOP RL Setup ===")
    print()
    print("[1/3] Checking browser...")
    check_browser()
    print()
    print("[2/3] Checking chromedriver...")
    check_chromedriver()
    print()
    print("[3/3] Patching game...")
    patch_qwop()
    print()
    print("Setup complete.")
    print()
    print("Start training:      python train.py")
    print("Watch trained agent: python play.py --model models/phase1/phase1_final.zip")


if __name__ == "__main__":
    main()
