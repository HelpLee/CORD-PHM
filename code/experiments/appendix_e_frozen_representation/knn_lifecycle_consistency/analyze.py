"""E4: layer-wise kNN lifecycle consistency of frozen representations."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"


def load_e3_module():
    path = ROOT / "layerwise_health_information" / "analyze.py"
    spec = importlib.util.spec_from_file_location("e3_layerwise_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load E3 representation extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def knn_lifecycle_error(representations: np.ndarray, lifecycle: np.ndarray, k: int):
    """Leave-self-out Euclidean kNN lifecycle MAE and per-query values."""
    representations = np.asarray(representations, dtype=np.float64)
    lifecycle = np.asarray(lifecycle, dtype=np.float64)
    if not 1 <= k < len(representations):
        raise ValueError(f"k must be in [1, {len(representations) - 1}], got {k}")
    squared_norm = np.einsum("nd,nd->n", representations, representations)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * representations @ representations.T
    np.maximum(squared, 0.0, out=squared)
    np.fill_diagonal(squared, np.inf)
    # Stable full sorting makes the result deterministic even if distances tie.
    neighbors = np.argsort(squared, axis=1, kind="stable")[:, :k]
    errors = np.abs(lifecycle[:, None] - lifecycle[neighbors]).mean(axis=1)
    return float(errors.mean()), errors, neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    e3 = load_e3_module()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required for E4 figures") from error

    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    metadata = []
    rows = []

    for domain in e3.DOMAINS:
        results[domain] = {"scores": {}}
        for variant in e3.VARIANTS:
            representations, info = e3.extract_layers(domain, variant, device, args.batch_size)
            metadata.append(info)
            snapshots = len(representations["final"])
            lifecycle = np.arange(snapshots, dtype=np.float64) / (snapshots - 1)
            results[domain].update({"unit": info["unit"], "snapshots": snapshots})
            results[domain]["scores"][variant] = {}
            for layer in e3.LAYERS:
                score, query_errors, neighbors = knn_lifecycle_error(
                    representations[layer], lifecycle, args.k)
                results[domain]["scores"][variant][layer] = score
                rows.append((domain, variant, layer, snapshots, args.k, score,
                             float(np.std(query_errors, ddof=1)),
                             float(np.median(query_errors))))
                np.savez_compressed(
                    OUT / f"{domain}_{variant}_{layer}_retrieval.npz",
                    lifecycle=lifecycle.astype(np.float32),
                    neighbor_indices=neighbors.astype(np.int32),
                    query_lifecycle_mae=query_errors.astype(np.float32),
                )

    with (OUT / "layerwise_knn_health_consistency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("domain", "variant", "layer", "snapshots", "k",
                         "mean_lifecycle_mae", "query_error_sample_std", "median_lifecycle_mae"))
        writer.writerows(rows)

    report = {
        "experiment": "E4_layerwise_knn_health_consistency",
        "encoder_training": False,
        "probe_training": False,
        "self_neighbor_excluded": True,
        "k": args.k,
        "distance": "Euclidean distance in each frozen layer representation",
        "metric": "mean absolute normalized-lifecycle difference to k nearest neighbors",
        "lower_is_better": True,
        "representation_definition": "identical to E3",
        "results": results,
        "sources": metadata,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    colors = {"single_domain": "#7a7a7a", "three_domain_adapter": "#1f5aa6"}
    display = {"single_domain": "Single-domain", "three_domain_adapter": "Three-domain + Adapter"}
    figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), sharey=True, constrained_layout=True)
    positions = np.arange(len(e3.LAYERS))
    all_scores = [results[domain]["scores"][variant][layer]
                  for domain in e3.DOMAINS for variant in e3.VARIANTS for layer in e3.LAYERS]
    upper = np.ceil((max(all_scores) + .01) / .02) * .02
    for axis, domain in zip(axes, e3.DOMAINS):
        for variant in e3.VARIANTS:
            values = [results[domain]["scores"][variant][layer] for layer in e3.LAYERS]
            axis.plot(positions, values, marker="o", linewidth=2.0, markersize=6,
                      color=colors[variant], label=display[variant])
        axis.set_xticks(positions, e3.LABELS)
        axis.set_ylim(0.0, upper)
        axis.set_title(f"{domain.capitalize()} ({results[domain]['snapshots']} snapshots)")
        axis.set_xlabel("Frozen encoder depth")
        axis.grid(axis="y", alpha=.22)
    axes[0].set_ylabel(f"{args.k}-NN lifecycle MAE (lower is better)")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle(f"E4. Layer-wise {args.k}-NN health consistency")
    figure.savefig(OUT / "layerwise_knn_health_consistency.png", dpi=240)
    plt.close(figure)

    lines = [
        "| Domain | Layer | Single-domain | Three-domain + Adapter | Improvement |",
        "|---|---|---:|---:|---:|",
    ]
    for domain in e3.DOMAINS:
        for layer, label in zip(e3.LAYERS, e3.LABELS):
            single = results[domain]["scores"]["single_domain"][layer]
            three = results[domain]["scores"]["three_domain_adapter"][layer]
            lines.append(f"| {domain.title()} | {label} | {single:.4f} | {three:.4f} | {single - three:+.4f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
