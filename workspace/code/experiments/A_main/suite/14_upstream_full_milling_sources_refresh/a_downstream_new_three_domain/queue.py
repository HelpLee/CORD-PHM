"""Persistently wait for refreshed three-domain upstream, then run A-main."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "three_domain"
STATUS = HERE / "queue_status.json"
STAGES = (
    ("milling", HERE / "milling", HERE / "milling/run_transfer_matrix.py"),
    ("battery", HERE / "battery", HERE / "battery/code/run.py"),
    ("bearing", HERE / "bearing", HERE / "bearing/code/run.py"),
)


def write(value) -> None:
    temporary = STATUS.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def upstream_status() -> dict:
    path = UPSTREAM / "training/status.json"
    if not path.exists():
        return {"state": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(name: str, cwd: Path, script: Path, completed: list[str]) -> None:
    result = cwd / "results.json"
    if result.exists() and json.loads(result.read_text(encoding="utf-8")).get("complete"):
        completed.append(name)
        return
    write({"state": "running", "stage": name, "completed": completed, "pid": os.getpid()})
    with (cwd / "stdout.log").open("a", encoding="utf-8", buffering=1) as out, \
         (cwd / "stderr.log").open("a", encoding="utf-8", buffering=1) as err:
        subprocess.run([sys.executable, "-u", str(script)], cwd=cwd,
                       stdout=out, stderr=err, check=True)
    completed.append(name)


def main() -> None:
    try:
        while True:
            observed = upstream_status()
            if observed.get("state") == "complete":
                break
            if observed.get("state") == "failed":
                raise RuntimeError(f"Three-domain upstream failed: {observed}")
            write({"state": "waiting_for_upstream", "observed": observed, "pid": os.getpid()})
            time.sleep(30)
        completed: list[str] = []
        for name, cwd, script in STAGES:
            run_stage(name, cwd, script, completed)
        write({"state": "complete", "completed": completed, "pid": os.getpid()})
    except BaseException as error:
        write({"state": "failed", "error": repr(error),
               "traceback": traceback.format_exc(), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
