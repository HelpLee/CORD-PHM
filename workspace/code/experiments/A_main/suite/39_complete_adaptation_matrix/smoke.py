import subprocess
import sys
from config import DOMAINS, HERE


def run(*args):
    subprocess.run([sys.executable, "-u", str(HERE / "downstream.py"), *args],
                   cwd=HERE, check=True)


for domain in DOMAINS:
    for mode in ("frozen", "partial"):
        run("--model", "e37_joint", "--mode", mode, "--domain", domain,
            "--fraction", ".1", "--smoke-train")
print("E39_SMOKE_PASSED", flush=True)

