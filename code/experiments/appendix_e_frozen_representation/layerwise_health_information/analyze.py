"""E3: layer-wise health-information geometry in frozen representations."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
DOMAINS = ("bearing", "battery", "milling")
VARIANTS = ("single_domain", "three_domain_adapter")
LAYERS = ("stem", "block1", "block2", "final")
LABELS = ("Stem", "Block 1", "Block 2", "Final")

sys.path.insert(0, str(ROOT))
import extract_representations as extraction  # noqa: E402


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks with deterministic tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    return float(np.dot(left_rank, right_rank) / denominator) if denominator else float("nan")


def pairwise_distances(representations: np.ndarray) -> np.ndarray:
    """All upper-triangle Euclidean distances without an NxNxD allocation."""
    representations = np.asarray(representations, dtype=np.float64)
    squared_norm = np.einsum("nd,nd->n", representations, representations)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * representations @ representations.T
    np.maximum(squared, 0.0, out=squared)
    upper = np.triu_indices(len(representations), k=1)
    return np.sqrt(squared[upper])


def snapshot_readout(tokens: torch.Tensor, local_valid: torch.Tensor) -> torch.Tensor:
    """Match the encoder's Global + valid-Local-mean snapshot aggregation."""
    global_hidden = tokens[:, 0]
    local_hidden = tokens[:, 1:]
    weights = local_valid.to(local_hidden.dtype)[..., None]
    pooled = (local_hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
    return torch.cat((global_hidden, pooled), dim=-1)


def extract_layers(domain: str, variant: str, device: torch.device, batch_size: int) -> tuple[dict, dict]:
    encoder, checkpoint = extraction.load_encoder(domain, variant, device)
    arrays, _, order, _, unit = extraction.inputs(domain)
    permutation = np.argsort(order, kind="stable")
    arrays = tuple(np.asarray(array)[permutation] for array in arrays)
    order = np.asarray(order)[permutation]
    captured: dict[str, torch.Tensor] = {}

    def capture_stem(module, args):
        captured["stem"] = args[0].detach()

    def capture_layer(name):
        def hook(module, args, output):
            captured[name] = output.detach()
        return hook

    handles = [
        encoder.transformer.layers[0].register_forward_pre_hook(capture_stem),
        encoder.transformer.layers[0].register_forward_hook(capture_layer("block1")),
        encoder.transformer.layers[1].register_forward_hook(capture_layer("block2")),
    ]
    collected = {layer: [] for layer in LAYERS}
    try:
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                batch = tuple(torch.as_tensor(array[start:start + batch_size], device=device) for array in arrays)
                captured.clear()
                final, _ = encoder(domain, *batch)
                if tuple(captured) != ("stem", "block1", "block2"):
                    raise RuntimeError(f"Incomplete hooks for {domain}/{variant}: {tuple(captured)}")
                channel_valid = batch[2].bool()
                token_valid = batch[3].bool()
                local_valid = (token_valid & channel_valid[..., None]).any(1)
                for layer in LAYERS[:-1]:
                    readout = snapshot_readout(captured[layer], local_valid)
                    collected[layer].append(readout.float().cpu().numpy())
                collected["final"].append(final.float().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    representations = {layer: np.concatenate(parts) for layer, parts in collected.items()}
    expected = {"stem": 192, "block1": 192, "block2": 192, "final": 96}
    for layer, values in representations.items():
        if values.shape != (len(order), expected[layer]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {domain}/{variant}/{layer} representation: {values.shape}")
    metadata = {
        "domain": domain,
        "variant": variant,
        "unit": str(unit),
        "snapshots": int(len(order)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": extraction.sha256(checkpoint),
    }
    return representations, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(args.device)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required for E3 figures") from error

    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    rows = []
    metadata = []

    for domain in DOMAINS:
        # Lifecycle is defined only by position within this held-out device.
        domain_payload = {}
        for variant in VARIANTS:
            representations, info = extract_layers(domain, variant, device, args.batch_size)
            domain_payload[variant] = representations
            metadata.append(info)
        snapshots = len(domain_payload[VARIANTS[0]]["final"])
        lifecycle = np.arange(snapshots, dtype=np.float64) / (snapshots - 1)
        lifecycle_distance = pairwise_distances(lifecycle[:, None])
        results[domain] = {"snapshots": snapshots, "unit": metadata[-1]["unit"], "scores": {}}
        for variant in VARIANTS:
            results[domain]["scores"][variant] = {}
            for layer in LAYERS:
                representation_distance = pairwise_distances(domain_payload[variant][layer])
                score = spearman(representation_distance, lifecycle_distance)
                results[domain]["scores"][variant][layer] = score
                rows.append((domain, variant, layer, snapshots,
                             snapshots * (snapshots - 1) // 2, score))

    with (OUT / "layerwise_health_information_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("domain", "variant", "layer", "snapshots", "snapshot_pairs", "rho_health"))
        writer.writerows(rows)

    report = {
        "experiment": "layerwise_health_information",
        "encoder_training": False,
        "variants": list(VARIANTS),
        "layers": list(LAYERS),
        "intermediate_readout": "concat(Global token, mean of valid Local tokens)",
        "final_readout": "96D frozen snapshot embedding e_t",
        "lifecycle": "r_t = t/(T-1) within each held-out device",
        "metric": "Spearman correlation between all pairwise representation distances and lifecycle distances",
        "results": results,
        "sources": metadata,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    colors = {"single_domain": "#7a7a7a", "three_domain_adapter": "#1f5aa6"}
    display = {"single_domain": "Single-domain", "three_domain_adapter": "Three-domain + Adapter"}
    figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), sharey=True, constrained_layout=True)
    positions = np.arange(len(LAYERS))
    all_scores = [results[domain]["scores"][variant][layer]
                  for domain in DOMAINS for variant in VARIANTS for layer in LAYERS]
    lower = max(-1.0, np.floor((min(all_scores) - .03) / .05) * .05)
    upper = min(1.0, np.ceil((max(all_scores) + .03) / .05) * .05)
    for axis, domain in zip(axes, DOMAINS):
        for variant in VARIANTS:
            values = [results[domain]["scores"][variant][layer] for layer in LAYERS]
            axis.plot(positions, values, marker="o", linewidth=2.0, markersize=6,
                      color=colors[variant], label=display[variant])
        axis.axhline(0.0, color="#b5b5b5", linewidth=.8)
        axis.set_xticks(positions, LABELS)
        axis.set_ylim(lower, upper)
        axis.set_title(f"{domain.capitalize()} ({results[domain]['snapshots']} snapshots)")
        axis.set_xlabel("Frozen encoder depth")
        axis.grid(axis="y", alpha=.22)
    axes[0].set_ylabel(r"Health-structure score  $\rho_{health}$")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle("E3. Layer-wise health information gain")
    figure.savefig(OUT / "layerwise_health_information_gain.png", dpi=240)
    plt.close(figure)

    lines = [
        "| Domain | Layer | Single-domain | Three-domain + Adapter | Gain |",
        "|---|---|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        for layer, label in zip(LAYERS, LABELS):
            single = results[domain]["scores"]["single_domain"][layer]
            three = results[domain]["scores"]["three_domain_adapter"][layer]
            lines.append(f"| {domain.title()} | {label} | {single:.4f} | {three:.4f} | {three - single:+.4f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
