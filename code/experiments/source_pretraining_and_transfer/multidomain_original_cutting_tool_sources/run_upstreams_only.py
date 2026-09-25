"""Run the two upstream pretraining jobs sequentially on a workstation or cluster.

The downstream jobs are intentionally not started here.  Each trainer owns its
checkpoint and can be resumed with ``--resume`` after an interruption.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
JOBS = ("milling_only", "three_domain")


def run(job: str, resume: bool) -> None:
    job_dir = HERE / job
    script = job_dir / "code" / "train_joint.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [sys.executable, "-u", str(script)]
    # The first cluster submission has no checkpoint. Start fresh in that
    # case, even though the Slurm wrapper always requests resume.
    has_checkpoint = (job_dir / "training" / "last.pt").is_file()
    effective_resume = resume and has_checkpoint
    if effective_resume:
        cmd.append("--resume")
    log = job_dir / "upstream_cluster.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {job} {'resume' if effective_resume else 'fresh'} ===\n")
        stream.flush()
        subprocess.run(cmd, cwd=str(job_dir), env=os.environ.copy(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="resume each job when its checkpoint exists")
    parser.add_argument("--job", choices=JOBS, action="append",
                        help="run only selected job(s), in the declared order")
    args = parser.parse_args()
    jobs = tuple(args.job) if args.job else JOBS
    for job in jobs:
        run(job, args.resume)


if __name__ == "__main__":
    main()
