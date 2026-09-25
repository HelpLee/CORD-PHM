"""Prepare the shared five-source, all-channel Milling runtime on cluster."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "gradnorm_multidomain_pretraining"
sys.path.insert(0, str(SOURCE))
import prepare_runtime as implementation  # noqa: E402

implementation.HERE = HERE
implementation.RUNTIME = HERE / "runtime/milling"
implementation.DATASETS = (
    "luh_milling", "matwi_milling", "nonastreda_milling",
    "qit_cemc_milling", "hmotp_milling",
)

if __name__ == "__main__":
    implementation.main()
