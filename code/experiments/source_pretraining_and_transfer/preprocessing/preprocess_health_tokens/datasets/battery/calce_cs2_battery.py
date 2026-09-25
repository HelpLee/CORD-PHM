"""Legacy compatibility shim; not registered as a new upstream dataset.

The new battery registry uses CALCE CS2 only through
``calce_cs2_downstream_4cells``. These aliases keep the untouched baseline
extractors importable without changing their old cycle-trajectory behavior.
"""

from preprocess_health_tokens.datasets._archive_battery_cycle_trajectory_20260824.calce_cs2_battery import (  # noqa: F401
    DATASET_NAME,
    MAX_TOKENS,
    OUT_ROOT,
    PROJECT_ROOT,
    RAW_ROOT,
    SEQ_STRIDE,
    build_trajectories,
    load_calce_cell_trajectory,
    parse_date_from_stem,
    process_calce_dataset,
    reference_capacity_from_rows,
    sorted_cell_dirs,
    sorted_txt_logs,
    sorted_workbooks,
)
