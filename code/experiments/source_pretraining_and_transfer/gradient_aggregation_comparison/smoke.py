import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
def run(*args):
    subprocess.run([sys.executable, "-u", *args], cwd=HERE, check=True)

run("train.py", "--arm", "joint_pcgrad", "--smoke")
for domain in ("bearing", "battery", "milling"):
    run("downstream.py", "--model", "smoke", "--domain", domain, "--fraction", "0.1", "--verify-only")
print("SMOKE_ALL_PASSED", flush=True)
