"""E2: leave-one-domain-out transfer of a frozen linear degradation axis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
DOMAINS = ("bearing", "battery", "milling")
TARGETS = (
    ("milling", ("bearing", "battery")),
    ("battery", ("bearing", "milling")),
    ("bearing", ("battery", "milling")),
)


def load_domain(domain: str) -> dict:
    """Load and chronologically order the frozen three-domain+Adapter embedding."""
    path = ROOT / "outputs" / domain / "three_domain_adapter.npz"
    with np.load(path, allow_pickle=False) as payload:
        embedding = np.asarray(payload["embedding"], dtype=np.float64)
        order = np.asarray(payload["order"])
        unit = np.asarray(payload["unit"])
    if embedding.ndim != 2 or embedding.shape[1] != 96:
        raise ValueError(f"Unexpected {domain} embedding shape: {embedding.shape}")
    if order.ndim != 1 or len(order) != len(embedding):
        raise ValueError(f"Invalid chronological order for {domain}: {order.shape}")
    permutation = np.argsort(order, kind="stable")
    embedding = embedding[permutation]
    order = order[permutation]
    if len(embedding) < 2:
        raise ValueError(f"{domain} requires at least two snapshots")
    lifecycle = np.arange(len(embedding), dtype=np.float64) / (len(embedding) - 1)
    return {
        "embedding": embedding,
        "order": order,
        "unit": str(unit.reshape(-1)[0]) if unit.size else "unknown",
        "lifecycle": lifecycle,
        "source": str(path),
    }


def fit_health_direction(payload: dict, sources: tuple[str, str]):
    """Fit one shared linear health direction with equal total weight per domain."""
    designs, targets, weights = [], [], []
    for domain in sources:
        embedding = payload[domain]["embedding"]
        lifecycle = payload[domain]["lifecycle"]
        designs.append(np.column_stack((embedding, np.ones(len(embedding)))))
        targets.append(1.0 - lifecycle)
        weights.append(np.full(len(embedding), 1.0 / len(embedding)))
    design = np.concatenate(designs)
    target = np.concatenate(targets)
    weight = np.sqrt(np.concatenate(weights))
    beta, _, rank, singular_values = np.linalg.lstsq(
        design * weight[:, None], target * weight, rcond=None)
    return beta[:-1], float(beta[-1]), int(rank), singular_values


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Dependency-free average ranks, including deterministic tie handling."""
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
    left_rank, right_rank = average_ranks(left), average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    return float(np.dot(left_rank, right_rank) / denominator) if denominator else float("nan")


def health_concordance(scores: np.ndarray) -> float:
    """Fraction of earlier/later pairs satisfying score_early > score_late."""
    concordant = ties = total = 0
    for index in range(len(scores) - 1):
        differences = scores[index] - scores[index + 1:]
        concordant += int(np.count_nonzero(differences > 0))
        ties += int(np.count_nonzero(differences == 0))
        total += len(differences)
    return float((concordant + 0.5 * ties) / total)


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required for E2 figures") from error

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {domain: load_domain(domain) for domain in DOMAINS}
    results, directions = {}, {}
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), constrained_layout=True)

    for axis, (target_domain, source_domains) in zip(axes, TARGETS):
        weight, bias, rank, singular_values = fit_health_direction(payload, source_domains)
        target = payload[target_domain]
        score = target["embedding"] @ weight + bias
        rho = spearman(score, target["lifecycle"])
        concordance = health_concordance(score)
        key = f"{source_domains[0]}_{source_domains[1]}_to_{target_domain}"
        directions[f"{key}_weight"] = weight
        directions[f"{key}_bias"] = np.asarray(bias)
        results[key] = {
            "sources": list(source_domains),
            "target": target_domain,
            "target_unit": target["unit"],
            "source_samples": {domain: int(len(payload[domain]["embedding"])) for domain in source_domains},
            "target_samples": int(len(score)),
            "spearman_score_vs_lifecycle": rho,
            "early_to_late_direction_agreement": -rho,
            "health_concordance_index": concordance,
            "linear_system_rank": rank,
            "smallest_singular_value": float(singular_values[-1]),
        }

        with (OUT / f"{key}_target_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("target_domain", "target_unit", "snapshot_index",
                             "normalized_lifecycle", "frozen_health_score"))
            for index, (position, value) in enumerate(zip(target["lifecycle"], score)):
                writer.writerow((target_domain, target["unit"], index, float(position), float(value)))

        axis.plot(target["lifecycle"], score, color="#2457a6", linewidth=1.15, alpha=.78)
        axis.scatter(target["lifecycle"], score, c=target["lifecycle"], cmap="viridis",
                     s=14, edgecolors="none", zorder=3)
        source_label = "+".join(domain.capitalize() for domain in source_domains)
        axis.set_title(f"{source_label} -> {target_domain.capitalize()}")
        axis.set_xlabel("Target normalized lifecycle  t/(T-1)")
        axis.set_ylabel("Frozen linear health score")
        axis.grid(alpha=.18, linewidth=.7)
        axis.text(.04, .06, f"Spearman rho = {rho:.3f}\nHealth C-index = {concordance:.3f}",
                  transform=axis.transAxes, fontsize=10,
                  bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "alpha": .88,
                        "edgecolor": "#aaaaaa"})

    figure.suptitle("E2. Cross-domain degradation-axis transfer\n"
                    "Frozen three-domain + Adapter encoder; high score = healthy")
    figure.savefig(OUT / "cross_domain_degradation_axis_transfer.png", dpi=240)
    plt.close(figure)
    np.savez(OUT / "fitted_source_domain_directions.npz", **directions)

    summary = {
        "experiment": "E2_cross_domain_degradation_axis_transfer",
        "encoder": "frozen three-domain + domain Adapter",
        "representation": "96D snapshot projector output e_t",
        "fit": "equal-domain-weighted linear least squares with intercept",
        "source_target": "health position = 1 - t/(T-1)",
        "orientation": "high score = healthy; low score = degraded",
        "target_labels_used_for_fit_or_orientation": False,
        "target_chronology_used_for_evaluation_only": True,
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "| Frozen sources -> zero-shot target | Target snapshots | Spearman rho(score, lifecycle) | Direction agreement (-rho) | Health C-index |",
        "|---|---:|---:|---:|---:|",
    ]
    for target_domain, source_domains in TARGETS:
        key = f"{source_domains[0]}_{source_domains[1]}_to_{target_domain}"
        result = results[key]
        label = f"{'+'.join(item.title() for item in source_domains)} -> {target_domain.title()}"
        lines.append(f"| {label} | {result['target_samples']} | "
                     f"{result['spearman_score_vs_lifecycle']:.4f} | "
                     f"{result['early_to_late_direction_agreement']:.4f} | "
                     f"{result['health_concordance_index']:.4f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
