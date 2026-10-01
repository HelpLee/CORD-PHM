<div align="center">

# CORD-PHM

### Cross-System Degradation Representation Learning for Prognostics

**Learning reusable degradation representations across heterogeneous physical systems**

<br>

[**Overview**](#overview) &nbsp;·&nbsp;
[**Results**](#results-at-a-glance) &nbsp;·&nbsp;
[**Reproduce**](#quick-reproduction) &nbsp;·&nbsp;
[**Paper ↔ Code**](PAPER_CODE_MAP.md)

<br>

<img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Source_Systems-3-2563EB" alt="3 source systems">
<img src="https://img.shields.io/badge/Downstream_Seeds-5-7C3AED" alt="5 downstream seeds">
<img src="https://img.shields.io/badge/Review-Double--blind-4B5563" alt="Double-blind">

<br><br>

<sub>Different observations · Different physics · Reusable degradation structure</sub>

</div>

---

<table>
<tr>
<td align="center" width="25%">
<b>3 source systems</b><br>
<sub>Bearing · Battery · Cutting Tool</sub>
</td>
<td align="center" width="25%">
<b>1 shared encoder</b><br>
<sub>Transformer backbone</sub>
</td>
<td align="center" width="25%">
<b>3 adaptation modes</b><br>
<sub>Frozen · Partial FT · Full FT</sub>
</td>
<td align="center" width="25%">
<b>1 unseen system type</b><br>
<sub>N-CMAPSS Engine</sub>
</td>
</tr>
</table>

> [!NOTE]
> **CORD keeps heterogeneous observation interfaces system-specific while sharing the representation backbone.**  
> The goal is not to force different physical systems into one handcrafted health coordinate, but to learn degradation structure that can be reused across devices, datasets, and system types.

## Overview

CORD studies a simple question:

> **Can degradation knowledge learned from heterogeneous physical systems be reused for prognostics without assuming identical sensors, observation spaces, or physical failure mechanisms?**

The selected model jointly pretrains **one shared encoder** on bearing, battery, and cutting-tool systems. Observation interfaces and reconstruction decoders remain system-specific, while reusable encoder parameters are shared across systems.

<table>
<tr>
<td align="center" width="22%">
<b>Bearing</b><br>
<sub>vibration observations</sub>
</td>
<td align="center" width="4%">→</td>
<td align="center" width="22%">
<b>System-specific<br>interfaces</b>
</td>
<td align="center" width="4%">→</td>
<td align="center" width="22%">
<b>CORD<br>shared encoder</b>
</td>
<td align="center" width="4%">→</td>
<td align="center" width="22%">
<b>RUL<br>adaptation</b>
</td>
</tr>

<tr>
<td align="center">
<b>Battery</b><br>
<sub>electrochemical observations</sub>
</td>
<td align="center">→</td>
<td align="center">
<b>System-specific<br>interfaces</b>
</td>
<td align="center">→</td>
<td align="center">
<b>Reusable degradation<br>representation</b>
</td>
<td align="center">→</td>
<td align="center">
<b>Frozen / Partial / Full</b>
</td>
</tr>

<tr>
<td align="center">
<b>Cutting Tool</b><br>
<sub>machining observations</sub>
</td>
<td align="center">→</td>
<td align="center">
<b>System-specific<br>interfaces</b>
</td>
<td align="center">→</td>
<td align="center">
<b>Shared Transformer<br>backbone</b>
</td>
<td align="center">→</td>
<td align="center">
<b>Unseen devices<br>and systems</b>
</td>
</tr>
</table>

### What is shared, and what stays private?

| System-specific | Shared across systems |
|:--|:--|
| Observation interface | Transformer backbone |
| Channel handling | Encoder parameters |
| Reconstruction decoder | Representation space |
| New-system calibration interface | Cross-system source pretraining |

---

## Results at a Glance

The complete represented-system benchmark evaluates **CORD vs. matched single-domain pretraining** under identical downstream protocols.

The table below shows the particularly diagnostic **Frozen + 10% labels** setting: the encoder is fixed, so improvements come from the pretrained representation rather than downstream encoder optimization.

<div align="center">

| Target system | Single-domain RMSE ↓ | **CORD RMSE ↓** | **Relative reduction** | Single-domain R² ↑ | **CORD R² ↑** |
|:--|--:|--:|--:|--:|--:|
| Bearing | 0.1229 ± 0.0158 | **0.1087 ± 0.0137** | **11.5%** | 0.8256 | **0.8635** |
| Battery | 0.0739 ± 0.0052 | **0.0704 ± 0.0025** | **4.7%** | 0.9339 | **0.9402** |
| Cutting Tool | 0.1245 ± 0.0145 | **0.0864 ± 0.0053** | **30.6%** | 0.7793 | **0.8948** |

</div>

<sub>Mean ± standard deviation over downstream seeds 42–46. Full Frozen / Partial FT / Full FT results at 10%, 20%, and 100% labels are retained in the adaptation matrix.</sub>

<br>

<table>
<tr>
<td align="center" width="33%">
<b>Represented systems</b><br><br>
Bearing · Battery · Cutting Tool<br>
<sub>10% / 20% / 100% labels</sub>
</td>
<td align="center" width="33%">
<b>Adaptation</b><br><br>
Frozen · Partial FT · Full FT<br>
<sub>matched downstream protocols</sub>
</td>
<td align="center" width="33%">
<b>New system type</b><br><br>
N-CMAPSS Engine<br>
<sub>excluded from source pretraining</sub>
</td>
</tr>
</table>

**Complete represented-system report**

```text
code/experiments/source_pretraining_and_transfer/
└── included_type_adaptation_matrix/
    └── reports/
        └── adaptation_matrix.json
```

---

## Evaluation

CORD separates two generalization settings.

### I. Represented-system adaptation

Bearing, battery, and cutting-tool targets are evaluated using:

- **Frozen** — train only the system-specific RUL readout;
- **Partial FT** — update the readout, final Transformer block, FinalNorm, and system adapter;
- **Full FT** — update the complete encoder and downstream head.

All three modes are evaluated at **10%, 20%, and 100% labels** with seeds **42–46**.

### II. Pretraining-excluded system adaptation

N-CMAPSS engine data are excluded from source pretraining.

The engine experiment introduces a new engine-specific interface while loading and freezing the pretrained shared components during interface calibration.

```text
code/experiments/appendix_d_engine_adaptation/
```

---

## Representation Analysis

CORD is evaluated not only by downstream error, but also by inspecting the organization of its frozen representations.

<table>
<tr>
<td width="50%" valign="top">

### Representation geometry

- health trajectories
- layer-wise lifecycle structure
- k-NN lifecycle consistency
- degradation-axis transfer

</td>
<td width="50%" valign="top">

### Transfer diagnostics

- cross-unit retrieval
- cutting-tool health probe
- paired downstream gains
- pretraining-mechanism analysis

</td>
</tr>
</table>

The analyses are intentionally conservative: they test whether health information becomes more reusable, **not** whether all physical systems share one universal degradation axis.

```text
code/experiments/appendix_e_frozen_representation/
```

<details>
<summary><b>Show the seven retained representation analyses</b></summary>

<br>

1. **Health trajectory** — visualizes each held-out device's frozen 96D trajectory.
2. **Degradation-axis transfer** — fits a health direction on two domains and applies it to the third.
3. **Layer-wise health information** — compares representation distance with within-device lifecycle distance.
4. **k-NN lifecycle consistency** — evaluates local health consistency in frozen representation space.
5. **Cutting-tool health probe** — tests transferable health information with a source-only linear probe.
6. **Cross-unit retrieval** — connects representation geometry with paired downstream transfer gains.
7. **Pretraining mechanism** — separates self-supervised pretraining effects from additional-source-domain effects.

</details>

---

## Quick Reproduction

> [!TIP]
> The main represented-system experiment can be reached in three steps.

```bash
# 1. Verify all bundled files
python validate_package.py

# 2. Stage the externally stored processed datasets
python prepare_external_data.py /path/to/data_phm

# 3. Reproduce the main adaptation matrix
bash run_included_type_adaptation.sh
```

Or run the matrix explicitly:

```bash
cd code/experiments/source_pretraining_and_transfer/included_type_adaptation_matrix

python summarize.py
python submit.py
```

`submit.py` schedules only missing cells and reuses retained completed cells where available.

---

## Paper ↔ Code

One of the goals of this repository is to make every major manuscript claim traceable to its implementation and retained evidence.

| Manuscript evidence | Repository entry point |
|:--|:--|
| Method / Appendix A — observation construction | `code/preprocess_health_tokens/` |
| Appendix B — architecture and optimization | `code/experiments/source_pretraining_and_transfer/` |
| Main Tables 2–3 / Appendix C — adaptation matrix | `included_type_adaptation_matrix/` |
| Appendix C — external RUL baselines | `rul_prediction_baselines/` |
| Main Figure 3 / Appendix D — engine adaptation | `code/experiments/appendix_d_engine_adaptation/` |
| Main Figures 4–5 / Appendix E — representation analysis | `code/experiments/appendix_e_frozen_representation/` |
| Main Table 5 / Appendix F — structured vs. raw input | `code/preprocess_health_tokens/raw_patch_ablation/` |
| Main Table 6 — ISM-only vs. ISM+IDM | `transfer_mechanism_comparison/` |
| Appendix G — efficiency and deployment | `code/experiments/appendix_g_efficiency/` |

<div align="center">

### **[Open the complete Paper ↔ Code Map →](PAPER_CODE_MAP.md)**

</div>

---

## Baselines & Ablations

<table>
<tr>
<td width="50%" valign="top">

### RUL baselines

- MLP
- Random Forest
- XGBoost
- TCN
- PatchTST
- iTransformer
- MOMENT

`rul_prediction_baselines/`

</td>
<td width="50%" valign="top">

### Ablations

- Structured vs. Raw-Resampled inputs
- ISM-only vs. ISM+IDM
- gradient aggregation controls
- single-domain vs. multi-domain pretraining

`raw_patch_ablation/`  
`transfer_mechanism_comparison/`

</td>
</tr>
</table>

---

<details>
<summary><b>Architecture & optimization details</b></summary>

<br>

The selected CORD source model jointly trains one encoder on bearing, battery, and cutting-tool source datasets.

- System-specific observation interfaces and decoders remain private.
- The Transformer backbone is shared.
- Bearing ISM coefficient: `0.3`
- Battery ISM coefficient: `0.3`
- Cutting-tool ISM coefficient: `1.0`
- IDM gradients remain full-strength for all three systems.
- Shared gradients are aggregated with CAGrad (`alpha=0.4`, `rescale=1`).
- A minimum Euclidean correction enforces the selected bearing-gradient directional floor.
- Source learning rate: `1e-4`
- Updates per system type per epoch: `20`
- Batch size: `32`
- Microbatch size: `8`
- Maximum epochs: `2000`
- Validation patience: `30`
- Minimum improvement: `1e-4`

Selected implementation:

```text
code/experiments/source_pretraining_and_transfer/
└── selected_bearing_floor_cagrad/
    ├── config.py
    ├── methods.py
    ├── train.py
    ├── downstream.py
    └── models/
```

</details>

<details>
<summary><b>Environment</b></summary>

<br>

The exact environment snapshot is retained at:

```text
code/experiments/source_pretraining_and_transfer/
└── shared_domain_components/
    └── requirements-lock.txt
```

Key packages include:

| Package | Version |
|:--|:--|
| PyTorch | 2.5.1 |
| NumPy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| SciPy | 1.16.3 |
| XGBoost | 3.2.0 |

The repository also retains the Slurm launchers used for the original experiments. Cluster-specific partitions and module settings may need to be adapted to another environment.

</details>

<details>
<summary><b>Repository structure</b></summary>

<br>

```text
CORD-PHM/
│
├── README.md
├── PAPER_CODE_MAP.md
│
├── validate_package.py
├── prepare_external_data.py
├── run_included_type_adaptation.sh
│
├── external_data_manifest.json
├── file_checksums.json
├── results_availability.json
├── package_metadata.json
│
└── code/
    ├── preprocess_health_tokens/
    │   ├── datasets/
    │   ├── raw_patch_ablation/
    │   └── build_downstream_raw_patches.py
    │
    └── experiments/
        ├── source_pretraining_and_transfer/
        │   ├── preprocessing/
        │   ├── shared_domain_components/
        │   ├── component_gradient_routing/
        │   ├── conditional_gradient_routing/
        │   ├── selected_bearing_floor_cagrad/
        │   ├── adaptation_protocol_controls/
        │   ├── included_type_adaptation_matrix/
        │   ├── rul_prediction_baselines/
        │   └── transfer_mechanism_comparison/
        │
        ├── appendix_c_included_type_generalization/
        ├── appendix_d_engine_adaptation/
        ├── appendix_e_frozen_representation/
        └── appendix_g_efficiency/
```

The current directory structure is intentionally retained because experiment configurations, checkpoints, result records, and integrity manifests depend on these relative paths.

</details>

<details>
<summary><b>Checkpoints and retained evidence</b></summary>

<br>

### Selected multi-domain encoder

```text
code/experiments/source_pretraining_and_transfer/
└── selected_bearing_floor_cagrad/
    └── models/
        └── joint_bearingfloor_cagrad/
            └── training/
                └── encoder.pt
```

### Evidence availability

| Component | Status |
|:--|:--|
| Main represented-system matrix / Appendix C | Complete |
| N-CMAPSS engine adaptation / Appendix D | Code and protocol |
| Frozen representation analysis / Appendix E | Code and retained analysis records |
| Structured-vs-raw ablation / Appendix F | Code and processed metadata |
| Efficiency / Appendix G | Code and protocol |

Machine-readable inventory:

```text
results_availability.json
```

</details>

<details>
<summary><b>Data & integrity</b></summary>

<br>

Dataset arrays are **not redistributed**.

Required processed files are identified by relative path, byte count, and SHA256 in:

```text
external_data_manifest.json
```

The reproducibility package additionally retains:

| File | Purpose |
|:--|:--|
| `PAPER_CODE_MAP.md` | manuscript-to-code mapping |
| `external_data_manifest.json` | external processed-data identities |
| `file_checksums.json` | bundled-file integrity |
| `results_availability.json` | evidence availability |
| `package_metadata.json` | anonymized package metadata |
| `validate_package.py` | package validation |
| `prepare_external_data.py` | external-data staging |

The repository deliberately excludes raw datasets, processed `.npz` arrays, generated NumPy caches, transient scheduler logs, and Python bytecode.

</details>

---

## Double-Blind Review

> [!IMPORTANT]
> This repository is prepared for double-blind review. Author names, affiliations, workstation paths, cluster accounts, development-repository history, raw datasets, and scheduler logs are omitted from the reproducibility package.

Neutral placeholders such as `/path/to/CORD` and `/home/anonymous` must be adapted to the local environment.

---

## Citation

Citation information will be added after the double-blind review period.

## License

Licensing information will be added with the public release.

---

<div align="center">

<sub><b>CORD-PHM</b> · Cross-system degradation representation learning for prognostics</sub>

<br><br>

[Back to top ↑](#cord-phm)

</div>
