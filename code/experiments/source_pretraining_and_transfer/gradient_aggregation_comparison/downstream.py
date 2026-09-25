"""Reuse the tested exp28 downstream protocol, with explicit arm provenance."""
import argparse
import importlib.util
import importlib
import json
import sys
import traceback
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", choices=("bearing", "battery", "milling"), required=True)
    parser.add_argument("--fraction", type=float, choices=(.1, .2, 1.), required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.model.startswith("single_") and args.model.split("_")[1] != args.domain:
        raise ValueError("Single-domain controls must use their own target domain")
    spec = importlib.util.spec_from_file_location("down28", HERE.parent / "full_channel_source_models/run_downstream.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    d.HERE = HERE
    package = HERE / ("smoke_downstream" if args.verify_only else "downstream") / args.model / args.domain / f"p{round(100*args.fraction)}"
    package.mkdir(parents=True, exist_ok=True)
    if (package / "results.json").exists() and not args.verify_only:
        previous = json.loads((package / "results.json").read_text())
        if previous.get("complete") and {row["seed"] for row in previous["rows"]} == set(range(42, 47)):
            d.upstream(args.model, args.domain, package)  # still verify hash
            print("ALREADY_COMPLETE", package, flush=True)
            return
    try:
        checkpoint = d.upstream(args.model, args.domain, package)
        if args.domain == "milling":
            if args.verify_only:
                d.SEEDS = ()  # validates conversion, strict state load and 7ch data; no training
            d.run_milling(args.model, args.fraction, checkpoint, package)
        else:
            if args.verify_only:
                sys.path[:0] = [str(d.WORKSPACE), str(d.WORKSPACE / "data_phm"),
                    str(d.SUITE / "shared_domain_components" / f"{args.domain}_code"),
                    str(d.EXP19 / f"{args.domain}_code"), str(d.EXP19)]
                module = importlib.import_module("bearing_adapter_study" if args.domain == "bearing" else "adapter_downstream")
                original = module.main
                def verify(package):
                    sys.argv = [sys.argv[0], "--verify-only"]
                    original(package)
                module.main = verify
            d.run_bearing_or_battery(args.domain, args.model, args.fraction, checkpoint, package)
        if args.verify_only:
            d.write(package / "status.json", {"state": "verified_only", "not_a_result": True})
        else:
            result = json.loads((package / "results.json").read_text())
            assert result.get("complete") and len(result["rows"]) == 5
            assert {row["seed"] for row in result["rows"]} == set(range(42, 47))
        print("DOWNSTREAM_OK", args.model, args.domain, args.fraction, flush=True)
    except BaseException as error:
        d.write(package / "status.json", {"state": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    main()
