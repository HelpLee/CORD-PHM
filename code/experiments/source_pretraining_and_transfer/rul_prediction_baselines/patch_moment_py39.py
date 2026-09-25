"""Scope a Python 3.9 annotation shim to this experiment's MOMENT copy.

MOMENT 0.1.4 declares Python >=3.10 because two source files evaluate PEP
604 unions in annotations. Deferred annotation evaluation makes these files
importable under the existing cluster Python 3.9 robot environment. No shared
site-packages file is modified.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent / ".deps" / "momentfm"
for relative in ("models/moment.py", "utils/forecasting_metrics.py"):
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    line = "from __future__ import annotations\n"
    if not source.startswith(line):
        path.write_text(line + source, encoding="utf-8")
        print(f"Patched {path}")
