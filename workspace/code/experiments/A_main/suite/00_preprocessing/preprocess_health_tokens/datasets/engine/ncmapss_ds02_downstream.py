"""Registry adapter for the N-CMAPSS DS02 flight-level artifact."""
import json
from pathlib import Path

from preprocess_health_tokens.prepare_ncmapss_downstream import CODE, build

DATASET_NAME = "ncmapss_ds02_downstream"
RAW_PATH = CODE / "data_phm" / "raw" / "Engine" / "N-CMAPSS_DS02-006.h5"


def process_health_tokens(cfg, *, out_root: Path, limit_segments=None):
    if limit_segments is not None:
        raise ValueError("N-CMAPSS canonical reproduction never truncates flights.")
    output = Path(out_root) / cfg.output_name
    build(RAW_PATH, output)
    return json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
