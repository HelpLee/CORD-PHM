"""Map one Slurm array index to exactly one baseline cell."""
from __future__ import annotations

import argparse
import subprocess
import sys

from run_all import DOMAINS, FRACTIONS, SEEDS, HERE

GROUPS = {
    "sequence": ("native_cnn_lstm_attention", "native_tcn",
                 "controlled_cnn_lstm_attention"),
    "transformer": ("native_patchtst", "native_itransformer",
                    "controlled_patchtst", "controlled_itransformer"),
    "moment": ("native_moment_frozen", "native_moment_full",
               "controlled_moment_frozen", "controlled_moment_full"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    cells = [(model, domain, fraction, seed)
             for model in GROUPS[args.group] for domain in DOMAINS
             for fraction in FRACTIONS for seed in SEEDS]
    model, domain, fraction, seed = cells[args.index]
    command = [sys.executable, "-u", str(HERE / "run.py"),
               "--model", model, "--domain", domain,
               "--fraction", str(fraction), "--seed", str(seed),
               "--gpu", "0"]
    print(f"BASELINE {args.group} {args.index}: {model} {domain} "
          f"{fraction} seed={seed}", flush=True)
    subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
