"""Prepare the seven-source Milling runtime and launch/resume GradNorm training."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def checked(*arguments: str) -> None:
    subprocess.run([sys.executable, "-u", *arguments], cwd=str(HERE), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runtime_status = HERE / "runtime/milling/status.json"
    if not runtime_status.is_file():
        checked(str(HERE / "prepare_runtime.py"))
    trainer = HERE / "three_domain/code/train_joint.py"
    command = [str(trainer)]
    if args.resume and (HERE / "three_domain/training/last.pt").is_file():
        command.append("--resume")
    checked(*command)


if __name__ == "__main__":
    main()
