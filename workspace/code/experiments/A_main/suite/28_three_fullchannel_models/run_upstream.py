"""Train one of the three requested all-channel upstream models."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
SOURCE15 = SUITE / "15_three_domain_original4_milling_all_channels"
PORTABLE_GL = SUITE / "19_gradnorm_downstream_cluster" / "bearing_code" / "global_local_data.py"
FOUR = ("luh_milling", "matwi_milling", "nonastreda_milling", "qit_cemc_milling")
FIVE = FOUR + ("hmotp_milling",)
SPECS = {
    "three4": ("three_domain", FOUR),
    "three5": ("three_domain", FIVE),
    "single5": ("milling_only", FIVE),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(SPECS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_kind, milling = SPECS[args.model]
    code = SOURCE15 / source_kind / "code"
    sys.path.insert(0, str(code))

    # The cluster checkout intentionally contains the processed NPZ/runtime caches
    # but not the ignored raw-preprocessing package.  Experiment 19 provides the
    # audited cluster-safe version of global_local_data: it is byte-identical to
    # the source implementation apart from replacing the three raw-data helpers
    # with fail-fast stubs.  All global feature caches are prepared and verified
    # before these jobs start, so raw acquisition files must never be read here.
    importlib.import_module("data")
    portable_spec = importlib.util.spec_from_file_location("global_local_data", PORTABLE_GL)
    portable_gl = importlib.util.module_from_spec(portable_spec)
    sys.modules["global_local_data"] = portable_gl
    portable_spec.loader.exec_module(portable_gl)

    spec = importlib.util.spec_from_file_location(f"exp28_{args.model}_trainer", code / "train_joint.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    # The immutable runtime audits are the protocol source of truth.  Some
    # older cached Bearing runtimes were generated with the historical split
    # ordering while the current Store.split() implementation orders the two
    # groups differently.  Reuse the audited train/validation units so the
    # checkpoint protocol remains identical and the trainer's consistency
    # assertion still verifies the resolved values.
    audit_paths = {
        "bearing": (HERE / "../01_shared_dependencies/bearing_runtime/"
                    "attention65_upstream_seed42/upstream_splits_scalers.json").resolve(),
        "milling": (HERE / "runtime/milling/upstream_splits_scalers.json").resolve(),
    }
    audited = {}
    for domain, path in audit_paths.items():
        records = __import__("json").loads(path.read_text(encoding="utf-8"))
        for record in records:
            audited[(domain, record["dataset"])] = (
                record["train"], record.get("val", record.get("validation", []))
            )
    original_split = trainer.gl.GlobalLocalStore.split

    def audited_split(store):
        dataset = store.name[:-8] if store.domain == "bearing" and store.name.endswith("_bearing") else store.name
        expected = audited.get((store.domain, dataset))
        return expected if expected is not None else original_split(store)

    trainer.gl.GlobalLocalStore.split = audited_split

    base = HERE / "models" / args.model
    trainer.BASE = base
    trainer.OUT = base / "training"
    trainer.EXPERIMENT = HERE
    trainer.A_SUITE = SUITE
    trainer.SHARED = SUITE / "01_shared_dependencies"
    trainer.MILLING_RUNTIME = HERE / "runtime/milling"
    trainer.MILLING = milling
    original_write = trainer.write

    def write(path, value):
        if path.name == "protocol.json" and isinstance(value, dict):
            value = dict(value)
            value["sources"] = dict(value.get("sources", {}), milling=list(milling))
            value["experiment_28_model"] = args.model
            value["only_protocol_change"] = (
                "All available channels are used. three4 uses the historical four Milling sources; "
                "three5 adds HMoTP; single5 uses the same five Milling sources without Bearing/Battery SSL."
            )
        original_write(path, value)

    trainer.write = write
    old_argv = sys.argv
    try:
        sys.argv = [str(code / "train_joint.py")] + (["--resume"] if args.resume else [])
        trainer.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
