"""Bundled raw-data loaders for the bearing HealthToken datasets.

This is intentionally part of ``preprocess_health_tokens``. Each public
dataset module declares only its dataset key and delegates to these shared
readers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict

import numpy as np


@dataclass
class Segment:
    file_path: Path
    segment_id: str
    sampling_rate: float
    metadata: Dict[str, object] = field(default_factory=dict)


def _root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "data_phm" / "raw").exists():
            return parent
    raise FileNotFoundError("Cannot locate data_phm/raw")


def _key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def _mat(path: Path):
    from scipy.io import loadmat
    return {k: v for k, v in loadmat(str(path), simplify_cells=True).items() if not k.startswith("__")}


def _txt(path: Path, delimiter=None, skip_header=0):
    a = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32, skip_header=skip_header, invalid_raise=True)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    return a[np.isfinite(a).any(axis=1)]


def build_segments(dataset: str):
    root = _root() / "data_phm" / "raw" / "Bearing"
    if dataset == "ims":
        raw = root / "IMS"
        files = sorted((p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() != ".pdf"), key=_key)
        return [Segment(p, f"{p.parent.name}__{p.name}", 20480.0, {"group_id": p.parent.name, "unit_id": p.parent.name, "test": p.parent.name, "run_id": p.parent.name, "condition_id": p.parent.name, "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)), "source_split": "raw_root"}) for p in files]
    if dataset == "xjtu":
        raw = root / "XJTU"; conditions = {"35Hz12kN": (35., 12.), "37.5Hz11kN": (37.5, 11.), "40Hz10kN": (40., 10.)}; out = []
        for cond, (hz, load) in conditions.items():
            for p in sorted((raw / cond).glob("Bearing*_*/[0-9]*.csv"), key=_key):
                b = p.parent.name; out.append(Segment(p, f"{cond}__{b}__{p.stem}", 25600., {"group_id": f"{cond}/{b}", "unit_id": b, "bearing": b, "run_id": b, "condition_id": cond, "condition": cond, "rotation_hz": hz, "load_kN": load, "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)), "source_split": "raw_root"}))
        return out
    if dataset == "femto":
        raw = root / "FEMTO" / "Learning_set"; out = []
        for p in sorted(raw.glob("Bearing*_*/acc_*.csv"), key=_key):
            b = p.parent.name; condition = f"Condition{b[7]}"; idx = int(re.findall(r"\d+", p.stem)[-1]); out.append(Segment(p, f"femto__{b}__{p.stem}", 25600., {"group_id": b, "unit_id": b, "bearing": b, "run_id": b, "condition_id": condition, "source_split": "learning_set", "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)), "file_index": idx, "snapshot_index": idx}))
        return out
    if dataset == "hust":
        raw = root / "HUST"; out = []
        for p in sorted((raw / "raw data").glob("*.xls"), key=_key):
            hz = float(re.search(r"(\d+)\s*[Hh][Zz]", p.stem).group(1)); out.append(Segment(p, f"hust__{p.stem}", 25600., {"group_id": f"{int(hz)}Hz", "unit_id": p.stem, "run_id": p.stem, "condition_id": p.stem.replace("_", "__"), "rotation_hz": hz, "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)), "source_split": "raw_root"}))
        return out
    specs = {"kaist": ("KAIST", {".csv"}, 25600.), "ferrara": ("Ferrara", {".mat"}, 25600.), "unsw": ("UNSW", {".mat"}, 51200.), "paderborn": ("Paderborn", {".mat"}, 64000.), "cwru": ("CWRU", {".mat"}, 12000.), "mfpt": ("MFPT", {".mat"}, 48828.), "seu": ("SEU", {".csv"}, 12000.)}
    dirname, exts, fs = specs[dataset]; raw = root / dirname
    files = [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    # The reference NPZ used ordinary path ordering for these two MAT trees.
    # Preserve it because it defines deterministic sample order and scaling.
    files = sorted(files) if dataset == "paderborn" else sorted(files, key=_key)
    if dataset == "mfpt": files = sorted(files, key=lambda p: (p.name.lower().startswith("baseline"), _key(p)))
    out = []
    for p in files:
        parent, grand = p.parent.name, p.parent.parent.name
        if dataset == "kaist": meta, sid = {"group_id": "run_to_failure", "unit_id": "run_to_failure", "run_id": "run_to_failure", "condition_id": "Vibration_Bearing_RuntoFailure", "file_id": p.stem, "filename": p.name, "source_relpath": p.name, "source_split": "raw_root"}, f"kaist__{p.stem}"
        elif dataset == "ferrara": meta, sid = {"group_id": "", "unit_id": parent, "run_id": parent, "condition_id": "", "rotation_hz": 40., "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)), "source_split": "raw_root"}, f"ferrara__{parent}__{p.stem}"
        elif dataset == "unsw":
            match = re.search(r"(\d+(?:\.\d+)?)\s*Hz", f"{p.stem} {parent} {grand}", flags=re.I)
            meta, sid = {"group_id": f"{grand}__{parent}", "unit_id": grand, "run_id": grand, "condition_id": parent, "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)), "source_split": "raw_root"}, f"unsw__{grand}__{parent}__{p.stem}"
            if match:
                meta["rotation_hz"] = float(match.group(1))
            elif parent.lower() == "multiple speeds":
                speed_suffix = re.search(r"_(\d+(?:\.\d+)?)$", p.stem)
                if speed_suffix:
                    meta["rotation_hz"] = float(speed_suffix.group(1))
        elif dataset == "paderborn": meta, sid = {"group_id": parent, "unit_id": parent, "run_id": parent, "condition_id": parent, "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)), "source_split": "raw_root"}, f"paderborn__{parent}__{p.stem}"
        elif dataset == "cwru": meta, sid = {"group_id": f"{grand}/{parent}", "unit_id": f"{grand}/{parent}", "run_id": f"{grand}/{parent}/{p.name}", "condition_id": f"{grand}/{parent}", "file_id": p.stem, "filename": p.name, "source_relpath": str(p.relative_to(raw)).replace('\\', '/'), "source_split": "raw_root"}, f"cwru__{grand}__{parent}__{p.name}"
        elif dataset == "mfpt": meta, sid = {"group_id": ".", "unit_id": ".", "run_id": ".", "condition_id": ".", "file_id": p.stem, "filename": p.name, "source_relpath": p.name, "source_split": "raw_root"}, f"mfpt__.__{p.stem}"
        else: meta, sid = {"group_id": p.stem, "unit_id": p.stem, "run_id": p.stem, "condition_id": "bearingset", "file_id": p.stem, "filename": p.name, "source_relpath": p.name, "source_split": "raw_root"}, f"seu__{p.stem}"
        if dataset == "unsw":
            if parent != "6Hz":
                continue  # Other speeds are sparse/repeated measurements of these same four tools.
            meta.update(group_id=grand, unit_id=grand, run_id=grand,
                        rotation_hz=6.0, snapshot_index=int(p.stem.split('_')[1]))
        elif dataset == "ferrara":
            meta.update(group_id=parent, unit_id=parent, run_id=parent,
                        condition_id=parent, snapshot_index=int(p.stem.split('_')[-1]))
        elif dataset == "kaist":
            from datetime import datetime
            stamp = datetime.strptime(p.stem.removeprefix('LogFile_'), '%Y-%m-%d-%H-%M-%S')
            meta['snapshot_index'] = int((stamp - datetime(2022, 6, 20, 17, 0, 31)).total_seconds())
        out.append(Segment(p, sid, fs, meta))
    return out


def read_signal(dataset: str, segment: Segment):
    p = segment.file_path
    if dataset == "ims": return _txt(p).T, 20480.
    if dataset == "xjtu": return _txt(p, delimiter=",", skip_header=1)[:, :2].T, 25600.
    if dataset == "femto":
        rows = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            v = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line)]
            if len(v) >= 6: rows.append(v)
        return np.asarray(rows, dtype=np.float32)[:, 4:6].T, 25600.
    if dataset == "hust": return _txt(p, delimiter="\t", skip_header=22)[:, 2:5].T, 25600.
    if dataset == "kaist":
        import pandas as pd
        # Keep missing rows in their native acquisition positions.
        return pd.read_csv(p,header=None,dtype=np.float32).to_numpy()[:,:2].T,25600.
    if dataset == "seu":
        # SEU distributes comma- and tab-delimited files under the same .csv
        # extension.  Detect the delimiter from the header/data prefix instead
        # of assuming that the extension implies comma-separated content.
        prefix_lines = []
        with p.open("r", encoding="utf-8", errors="ignore") as stream:
            for _ in range(20):
                line = stream.readline()
                if not line:
                    break
                prefix_lines.append(line)
        prefix = "".join(prefix_lines)
        delimiter = "\t" if prefix.count("\t") > prefix.count(",") else ","
        return _txt(p, delimiter=delimiter, skip_header=16)[:, :8].T, 12000.
    m = _mat(p)
    if dataset == "cwru":
        # Match the established CWRU NPZ contract: retain every drive-end and
        # fan-end trace present in the MAT file, tolerate files that contain
        # only one sensor, and align multi-record files to their common length.
        # Baseline-acceleration and scalar RPM fields are intentionally excluded.
        traces = [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for key, value in m.items()
            if key.endswith(("_DE_time", "_FE_time"))
        ]
        if not traces:
            raise ValueError("No CWRU drive-end or fan-end time trace found.")
        common_length = min(trace.size for trace in traces)
        return np.stack([trace[:common_length] for trace in traces]), 12000.
    if dataset == "unsw": return np.stack([np.asarray(m['accH'],dtype=np.float32).reshape(-1),np.asarray(m['accV'],dtype=np.float32).reshape(-1)]), float(m['Fs'])
    if dataset == "mfpt":
        bearing = m["bearing"]
        return np.asarray(bearing["gs"], dtype=np.float32).reshape(1, -1), float(bearing.get("sr", 48828.0))
    if dataset == "paderborn":
        root = next(v for v in m.values() if isinstance(v,dict) and 'Y' in v); return np.asarray(next(y['Data'] for y in root['Y'] if y.get('Name') == 'vibration_1'),dtype=np.float32).reshape(1,-1), 64000.
    if dataset == "ferrara": return np.asarray(m['y'],dtype=np.float32).reshape(1,-1), float(m['Fs'])
    raise ValueError(f"Unsupported bearing dataset: {dataset}")
