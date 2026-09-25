"""Prepare canonical Milling inputs, then train Milling-only and three-domain serially."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
STATUS = HERE / "queue_status.json"


def write(value) -> None:
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def run_stage(name: str, command: list[str], cwd: Path, log_path: Path) -> None:
    write(dict(state="running", stage=name, pid=os.getpid()))
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    try:
        run_stage("prepare_runtime", [sys.executable, "-u", str(HERE / "prepare_runtime.py")], HERE,
                  HERE / "prepare_runtime.log")
        completed = []
        for name in ("milling_only", "three_domain"):
            package = HERE / name
            status_path = package / "training/status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("state") == "complete":
                completed.append(name)
                continue
            command = [sys.executable, "-u", str(package / "code/train_joint.py")]
            if (package / "training/last.pt").exists():
                command.append("--resume")
            run_stage(name, command, package, package / "stdout.log")
            completed.append(name)
        write(dict(state="complete", completed=completed, pid=os.getpid()))
    except Exception as error:
        write(dict(state="failed", error=repr(error), traceback=traceback.format_exc(), pid=os.getpid()))
        raise


if __name__ == "__main__":
    main()
