"""Registry adapter for the complete historical PHM2010 builder."""
from pathlib import Path

from preprocess_health_tokens.prepare_phm2010_downstream import build

DATASET_NAME = "phm2010_milling_downstream"
from preprocess_health_tokens.suite_paths import data_code_root
CODE = data_code_root()
RAW_ROOT = CODE / "data_phm" / "raw" / "Milling" / "phm2010"


def process_health_tokens(cfg, *, out_root: Path, limit_segments=None):
    return build(RAW_ROOT, Path(out_root) / cfg.output_name, limit_cuts=limit_segments)
