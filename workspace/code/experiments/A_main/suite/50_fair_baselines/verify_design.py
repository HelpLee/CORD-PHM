"""Read-only design audit for the two-family baseline protocol."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from run import PREPARED
from run_all import DOMAINS, FRACTIONS, MODELS, SEEDS, HERE
from native_data import RAW, BEARING_ROWS


def main():
    configs = json.loads((HERE / "model_configs.json").read_text(encoding="utf-8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    assert set(configs) == set(MODELS)
    assert tuple(protocol["fractions"]) == FRACTIONS
    assert tuple(protocol["seeds"]) == SEEDS
    detail = {}
    for domain in DOMAINS:
        manifest = json.loads((PREPARED / domain / "manifest.json").read_text())
        raw = next((RAW / domain).glob("*.npz"))
        raw_meta = json.loads(raw.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(PREPARED / domain / "snapshots.npz") as canonical:
            size = len(canonical["y"])
            y = canonical["y"]
        with np.load(raw) as source:
            if domain == "bearing":
                with np.load(BEARING_ROWS) as subset:
                    selected = subset["source_rows"]
                assert len(selected) == size
                assert selected.max() < len(source["x_raw_local"])
            else:
                assert len(source["x_raw_local"]) == size
                assert np.allclose(source["y_rul_norm"], y, atol=1e-6)
            channels = int(source["x_raw_local"].shape[1])
            assert source["x_raw_local"].shape[2:] == (64, 26)
        for fraction in FRACTIONS:
            with np.load(PREPARED / domain / f"split_{int(fraction * 100)}.npz") as split:
                train = set(split["train_sequence"][:, -1])
                validation = set(split["validation_sequence"][:, -1])
                test = set(split["test_sequence"][:, -1])
                assert train.isdisjoint(validation)
                assert train.isdisjoint(test)
                assert validation.isdisjoint(test)
        detail[domain] = {
            "snapshots": size, "raw_channels": channels,
            "raw_signal": raw_meta["raw_signal_name"],
            "raw_sha256": raw_meta["output_sha256"],
            "split_hashes": {key: value["sha256"]
                             for key, value in manifest["splits"].items()},
        }
    report = {"passed": True, "jobs": len(MODELS) * len(DOMAINS) *
              len(FRACTIONS) * len(SEEDS), "models": MODELS,
              "domains": detail, "legacy_result_reuse": False}
    (HERE / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
