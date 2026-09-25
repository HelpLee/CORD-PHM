"""Run the most complex arm for one update and verify its checkpoint."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
subprocess.run([sys.executable, "-u", "train.py", "--arm", "joint_obsprivate_cagrad", "--smoke", "--resume"], cwd=here, check=True)
subprocess.run([sys.executable, "-u", "downstream.py", "--model", "smoke_joint_obsprivate_cagrad",
                "--domain", "battery", "--fraction", ".1", "--verify-only"], cwd=here, check=True)
print("SMOKE_OK")
