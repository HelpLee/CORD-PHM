"""Train both original-four-source all-channel upstreams, then run 10% Milling probes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
STATUS = HERE / "status.json"


def write(value) -> None:
    temporary = STATUS.with_name(f"{STATUS.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(temporary, STATUS)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.25)


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("state") == "complete"


def run_stage(name: str, command: list[str], cwd: Path, log_path: Path) -> None:
    write({"state": "running", "stage": name, "pid": os.getpid()})
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    completed = []
    try:
        for name in ("milling_only", "three_domain"):
            package = HERE / name
            status = package / "training/status.json"
            if not complete(status):
                command = [sys.executable, "-u", str(package / "code/train_joint.py")]
                if (package / "training/last.pt").is_file():
                    command.append("--resume")
                run_stage(name, command, package, package / "stdout.log")
            completed.append(name)
        downstream = HERE / "downstream_milling_10pct"
        if not complete(downstream / "status.json"):
            run_stage("downstream_milling_10pct", [sys.executable, "-u", str(downstream / "run_comparison.py")],
                      downstream, downstream / "stdout.log")
        completed.append("downstream_milling_10pct")
        write({"state": "complete", "completed": completed, "pid": os.getpid()})
    except BaseException as error:
        write({"state": "failed", "completed": completed, "error": repr(error),
               "traceback": traceback.format_exc(), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
