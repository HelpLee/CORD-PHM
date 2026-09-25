"""Generate the registered C1/C2 figures and exact five-seed tables."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / "source_pretraining_and_transfer"
OUT = HERE / "outputs"
DOMAINS = ("bearing", "battery", "milling")
FRACTIONS = (.1, .2, 1.0)
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("rmse", "mae", "r2")
PACKAGES = {
    "bearing": ("40_downstream_bearing_scratch", "41_downstream_bearing_three_domain_adapter"),
    "battery": ("30_downstream_battery_scratch", "31_downstream_battery_three_domain_adapter"),
    "milling": ("22_downstream_milling_scratch", "20_downstream_milling_three_domain_adapter"),
}
LABELS = {"scratch": "Scratch", "frozen_probe": "Frozen",
          "partial_finetune": "Partial FT", "full_finetune": "Full FT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(domain: str):
    scratch_name, transfer_name = PACKAGES[domain]
    rows, sources = [], []
    for treatment, package_name in (("scratch", scratch_name), ("transfer", transfer_name)):
        package = SUITE / package_name
        result_path, protocol_path = package / "results.json", package / "protocol.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["complete"] is True
        selected = [row for row in payload["rows"]
                    if (treatment == "scratch" and row["arm"] == "scratch")
                    or (treatment == "transfer" and row["arm"] != "scratch")]
        assert len(selected) == (15 if treatment == "scratch" else 45)
        rows.extend(selected)
        sources.append(dict(package=package_name, result=str(result_path),
                            result_sha256=sha256(result_path), protocol=str(protocol_path),
                            protocol_sha256=sha256(protocol_path)))
    return rows, sources


def aggregate(rows):
    grouped = defaultdict(list)
    seeds = defaultdict(set)
    for row in rows:
        assert row["seed"] in SEEDS and row["fraction"] in FRACTIONS
        for metric in METRICS:
            key = (row["fraction"], row["arm"], metric)
            grouped[key].append(float(row["metrics"][metric]))
            seeds[key].add(row["seed"])
    output = []
    for (fraction, arm, metric), values in sorted(grouped.items()):
        assert len(values) == 5 and seeds[(fraction, arm, metric)] == set(SEEDS)
        output.append(dict(fraction=fraction, arm=arm, metric=metric, n=5,
                           mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                           values=values))
    return output


def lookup(summary, fraction, arm, metric):
    return next(row for row in summary if row["fraction"] == fraction
                and row["arm"] == arm and row["metric"] == metric)


def plot_grid(all_summary, arms, title, destination):
    import matplotlib.pyplot as plt
    x = np.arange(len(FRACTIONS))
    styles = {
        "scratch": dict(color="#4c4c4c", marker="o", linestyle="--"),
        "frozen_probe": dict(color="#0072B2", marker="s", linestyle="-"),
        "partial_finetune": dict(color="#E69F00", marker="^", linestyle="-"),
        "full_finetune": dict(color="#009E73", marker="D", linestyle="-"),
    }
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 10.2), constrained_layout=True)
    for row_index, domain in enumerate(DOMAINS):
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            for arm in arms:
                records = [lookup(all_summary[domain], fraction, arm, metric)
                           for fraction in FRACTIONS]
                mean = np.asarray([record["mean"] for record in records])
                std = np.asarray([record["std"] for record in records])
                axis.errorbar(x, mean, yerr=std, capsize=3, linewidth=1.8,
                              markersize=5.5, label=LABELS[arm], **styles[arm])
            axis.set_xticks(x, ("10%", "20%", "100%"))
            axis.set_xlabel("Labeled training data")
            axis.set_ylabel({"rmse": "RMSE ↓", "mae": "MAE ↓", "r2": "R² ↑"}[metric])
            axis.grid(axis="y", alpha=.22, linewidth=.7)
            if row_index == 0:
                axis.set_title(metric.upper().replace("R2", "R²"))
            if column_index == 0:
                axis.text(-.28, .5, domain.capitalize(), rotation=90,
                          transform=axis.transAxes, va="center", ha="center", fontweight="bold")
            if row_index == 0 and column_index == 2:
                axis.legend(frameon=False, loc="best")
    fig.suptitle(title, fontsize=14)
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path, all_summary, allowed_arms):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("domain", "fraction", "arm", "metric", "n", "mean", "std"))
        for domain in DOMAINS:
            for row in all_summary[domain]:
                if row["arm"] in allowed_arms:
                    writer.writerow((domain, row["fraction"], row["arm"], row["metric"],
                                     row["n"], row["mean"], row["std"]))


def markdown_table(all_summary, arms, heading):
    lines = [f"# {heading}", ""]
    for domain in DOMAINS:
        lines += [f"## {domain.capitalize()}", "",
                  "| Labels | Strategy | RMSE | MAE | R² |",
                  "|---:|---|---:|---:|---:|"]
        for fraction in FRACTIONS:
            for arm in arms:
                values = {metric: lookup(all_summary[domain], fraction, arm, metric)
                          for metric in METRICS}
                def fmt(metric):
                    return f'{values[metric]["mean"]:.4f} ± {values[metric]["std"]:.4f}'
                lines.append(f"| {int(fraction * 100)}% | {LABELS[arm]} | "
                             f"{fmt('rmse')} | {fmt('mae')} | {fmt('r2')} |")
        lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_summary, provenance = {}, {}
    for domain in DOMAINS:
        rows, sources = load_rows(domain)
        all_summary[domain] = aggregate(rows)
        provenance[domain] = sources
    c1_arms = ("scratch", "frozen_probe")
    c2_arms = ("frozen_probe", "partial_finetune", "full_finetune")
    plot_grid(all_summary, c1_arms,
              "C1  Label efficiency: Scratch vs Three-domain pretrained Frozen Probe",
              OUT / "C1_label_efficiency.png")
    plot_grid(all_summary, c2_arms,
              "C2  Adaptation strategy for the Three-domain pretrained model",
              OUT / "C2_adaptation_strategy.png")
    write_csv(OUT / "C1_label_efficiency.csv", all_summary, c1_arms)
    write_csv(OUT / "C2_adaptation_strategy.csv", all_summary, c2_arms)
    (OUT / "C1_label_efficiency.md").write_text(
        markdown_table(all_summary, c1_arms, "C1 Label Efficiency"), encoding="utf-8")
    (OUT / "C2_adaptation_strategy.md").write_text(
        markdown_table(all_summary, c2_arms, "C2 Adaptation Strategy"), encoding="utf-8")
    compact = {domain: [{k: v for k, v in row.items() if k != "values"}
                        for row in summary] for domain, summary in all_summary.items()}
    (OUT / "summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(dict(
        audit="all expected cells contain exactly seeds 42--46",
        source_files=provenance), indent=2), encoding="utf-8")
    print(json.dumps(dict(state="complete", domains=list(DOMAINS),
                          figures=["C1_label_efficiency.png", "C2_adaptation_strategy.png"],
                          result_tables=["C1_label_efficiency.csv", "C2_adaptation_strategy.csv"]), indent=2))


if __name__ == "__main__":
    main()
