# Battery discharge-cycle snapshot preprocessing

This is the new battery pipeline. The complete legacy tree is preserved at
`code/_archive_preprocess_health_tokens_cycle_trajectory_20260824`, and the old
processed NPZ files are preserved under
`code/data_phm/processed_health_tokens/_archive_battery_cycle_trajectory_20260824`.

## Contract

- One sample is one complete **discharge segment**, from discharge start to cutoff.
- Only discharge `V(t), I(t), T(t), Q(t)` is retained. Charge samples are not mixed in.
- `u = Q(t) / Q_end` maps each discharge to `[0, 1]`.
- Native samples are not resampled to a fixed point count.
- Local windows use width `5%` capacity and stride `2.5%` capacity (39 windows for a complete cycle).
- A snapshot must contain at least 30 valid windows out of the theoretical 39.
  Cycles below this common coverage threshold are rejected for every battery dataset.
- Each window has the exact 26 features defined by `BATTERY_LOCAL_FEATURE_NAMES` in
  `common/battery_snapshot_utils.py`.
- Output is `[N, 1, 64, 26]`; real windows are marked by `token_mask`.
- Slots 0--38 keep fixed capacity positions even when an individual window is
  invalid; an invalid interior window is masked in place and is never compacted
  into another physical position. Slots 39--63 are structural padding.
- Missing/non-computable individual features are zero and marked false in `feature_mask`.
  In particular, a dataset without temperature does not fabricate `0 degC`; dimensions
  17--22 are zero only as masked placeholders.

## Upstream eligibility among the ten candidates

| Candidate | New upstream | Reason |
|---|---:|---|
| NASA Battery | yes | Explicit discharge records with native time, voltage, current and temperature; Q can be integrated. |
| Oxford Battery Degradation | yes | Native C1dc time/voltage/capacity/temperature curves; current is derived physically as `-dQ/dt`. |
| Stanford / FastCharge | no | The local copy exposes author-interpolated cycle curves and no physical time vector, so it violates native-sample/local-duration requirements. |
| SNL Battery | no | The available `Cell.Cyc` representation used by the legacy code is charge/summary oriented; a complete native discharge trace cannot be identified reliably. |
| XJTU Battery | yes | Per-cycle raw time/voltage/current/capacity/temperature arrays; source capacity resets locate discharge boundaries, while monotone `Q_d(t)` is reintegrated from native time/current; capacity-only fallbacks are rejected. |
| ISU-ILCC | yes | `Cycling_json.zip` has per-cycle `QV_discharge` time/voltage/current/capacity curves. The configured DoD endpoint is treated as that experiment's discharge cutoff; sparse RPT fallbacks are rejected. |
| HUST Battery | yes | Explicit cycle DataFrames and discharge status with time/voltage/current; the first-to-last discharge-status interval is retained, Q is integrated, and temperature is feature-masked. |
| RWTH Drive-Cycle Aging | yes | Continuous time/voltage/current/temperature; complete discharges are segmented from charged voltage to confirmed cutoff and Q is integrated. |
| SDU Battery | yes | Primary- and second-life per-cycle time/voltage/current/discharge-capacity arrays; the complete negative-current interval is retained and temperature is feature-masked when absent. |
| MICH_EXP / Michigan Fast Formation | yes | In each aging-test cycle number, the contiguous negative-current run with the largest discharge-capacity gain is selected. This excludes diagnostic pulse trains, intervening charges, and day-long rests from the main discharge snapshot. |

CALCE CS2 is not upstream. Only `CS2_35`, `CS2_36`, `CS2_37`, and `CS2_38`
are registered as downstream. CALCE CX2 is not registered anywhere in the new pipeline.
The unregistered `calce_cs2_battery.py` file is only a compatibility shim for
the pre-existing XGBoost/CNN-LSTM baseline extractors; `--battery-upstream`
cannot select it.

## Commands

Generate all selected upstream datasets:

```bash
python code/preprocess_health_tokens/main.py --battery-upstream
```

Generate only the CALCE CS2 downstream four-cell file:

```bash
python code/preprocess_health_tokens/main.py --cs2-downstream
```

The downstream output keeps its original filename:
`calce_cs2_downstream_battery_health_tokens.npz`.

`--limit-segments 1` limits source files/cells for adapter debugging. It does not
change the tokenizer contract. Full preprocessing was intentionally not run while
creating this code.

During generation, every battery dataset reports live cumulative counts in the
progress line as `scanned` (examined cycles), `kept` (valid snapshots), and
`skipped` (rejected cycles). Each rejection warning includes the same totals,
and a final per-dataset summary is printed before the NPZ is saved.

Bounded read-only adapter QA (does not write NPZ files):

```bash
python code/preprocess_health_tokens/audit_battery_adapters.py --sample-cycles 30 --limit-files 1
```
