"""Complete frozen/partial transfer cells while inheriting the audited ADAPTATION_CONTROLS protocol."""
import argparse
import importlib.util
import sys
from pathlib import Path

from config import DOMAINS, ADAPTATION_CONTROLS, HERE, checkpoint


def load_module(path, source):
    spec = importlib.util.spec_from_file_location("cord_adaptation_downstream", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("multi_domain", "single_domain"), required=True)
    parser.add_argument("--mode", choices=("frozen", "partial"), required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--fraction", choices=(.1, .2, 1.), type=float, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke-train", action="store_true")
    args = parser.parse_args()

    source_path = ADAPTATION_CONTROLS / "downstream.py"
    source = source_path.read_text(encoding="utf-8")
    old_guard = "    if args.mode!='full' and (args.domain!='bearing' or args.fraction!=.1):raise ValueError('Bearing10 adaptations only')\n"
    if source.count(old_guard) != 1:
        raise RuntimeError("ADAPTATION_CONTROLS guard drift")
    source = source.replace(old_guard, "")
    source = source.replace("choices=('full','partial','l2sp1','l2sp2')",
                            "choices=('full','frozen','partial','l2sp1','l2sp2')")
    old_effective = "effective_arm='scratch' if scratch else 'partial_finetune' if args.mode=='partial' else 'full_finetune'"
    new_effective = "effective_arm='scratch' if scratch else 'partial_finetune' if args.mode=='partial' else 'frozen_probe' if args.mode=='frozen' else 'full_finetune'"
    if source.count(old_effective) != 1:
        raise RuntimeError("ADAPTATION_CONTROLS effective-arm drift")
    source = source.replace(old_effective, new_effective)
    # ADAPTATION_CONTROLS places ``phase`` in its output path.  ADAPTATION_MATRIX has no phase dimension, so
    # remove that component while preserving every other inherited setting.
    old_package = "package=HERE/('smoke_downstream' if args.verify_only or args.smoke_train else 'downstream')/args.phase/args.model/args.mode/args.domain/f'p{round(args.fraction*100)}'"
    new_package = "package=HERE/('smoke_downstream' if args.verify_only or args.smoke_train else 'downstream')/args.model/args.mode/args.domain/f'p{round(args.fraction*100)}'"
    if source.count(old_package) != 1:
        raise RuntimeError("ADAPTATION_CONTROLS package-path drift")
    source = source.replace(old_package, new_package)
    module = load_module(source_path, source)
    module.HERE = HERE
    module.source_for = lambda model, domain: checkpoint(model, domain)

    # Patch the E28 source read by ADAPTATION_CONTROLS so Milling uses the requested transfer arm.
    original_read_text = Path.read_text
    def read_text(path, *a, **kw):
        text = original_read_text(path, *a, **kw)
        if path.name == "run_downstream.py":
            marker = 'down.ARMS = ("full_finetune",)'
            if text.count(marker) != 1:
                raise RuntimeError("E28 Milling arm drift")
            selected = "partial_finetune" if args.mode == "partial" else "frozen_probe"
            text = text.replace(marker, f'down.ARMS = ("{selected}",)')
        return text
    Path.read_text = read_text
    try:
        # ``phase`` is required by ADAPTATION_CONTROLS's parser but is no longer part of ADAPTATION_MATRIX's
        # output path after the guarded source transformation above.
        sys.argv = [sys.argv[0], "--phase", "baseline", "--model", args.model,
                    "--mode", args.mode, "--domain", args.domain,
                    "--fraction", str(args.fraction)]
        if args.verify_only:
            sys.argv.append("--verify-only")
        if args.smoke_train:
            sys.argv.append("--smoke-train")
        module.main()
    finally:
        Path.read_text = original_read_text


if __name__ == "__main__":
    main()
