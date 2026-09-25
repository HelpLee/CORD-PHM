# Two-family RUL baseline study

The previous local baseline code and results are preserved at
`../_archive/06_fair_baselines_legacy_20260923`. None of its metrics is
silently imported into this study. Canonical splits and working model code are
reused; scores are regenerated under the new input definitions.

## Questions and inputs

| Family | Question | Input |
|---|---|---|
| Standard/native | How well does a model perform on the actual RUL task with a sensible model-specific input? | Raw sensor amplitudes from `data_phm/processed_raw_patches`, derived directly from `data_phm/raw`. Trees use waveform statistics; neural models consume raw temporal views. |
| Controlled | Does TRACE benefit from its representation/pretraining rather than merely the HealthToken preprocessing? | The exact downstream HealthToken local `[C,64,26]` plus global values and channel/token/history masks available to TRACE. No raw inputs. |

The canonical device split, targets, 10/20/100% training-label budgets,
validation selection, five seeds (42--46), and RMSE/MAE/R² metrics are shared.
The **standard group is a practical performance comparison, not a
pretraining-only causal ablation**. The controlled group fixes input
information more tightly. Both groups use their own scalar RUL prediction
head; neither borrows TRACE's domain-specific downstream head.

## Raw data alignment

| Domain | Raw channels | Alignment and allowed observables |
|---|---:|---|
| Bearing | 2 | XJTU raw vibration; canonical post-FPT 568 rows selected by the source-row mapping |
| Battery | 1 | CALCE raw voltage; 3464 exact rows, plus observed cycle capacity and duration |
| Milling | 7 | PHM2010 raw force/vibration/AE channels; 822 exact rows |

`native_data.py` verifies row counts and available raw RUL labels before
training. Only canonical **training** rows fit the robust scaler; validation
and held-out devices do not. No wear measurement, normalized RUL label, or
test-derived statistic is used as a feature. The raw-patch files are
parameter-free resampled sensor windows, not handcrafted 26-feature
HealthTokens. They retain the existing 64-window sample alignment so the
label/split protocol remains identical.
Only observed raw windows are packed in their original order, then resampled
to a fixed waveform length. This prevents the 55--57 padded bearing slots
from being mistaken for real sensor readings. For MOMENT, raw amplitudes
remain unscaled before its built-in normalization; other native models use
training-source robust scaling.

## Models

The native group comprises XGBoost, Random Forest, CNN-LSTM-Attention, TCN,
PatchTST, iTransformer, and MOMENT frozen/full. Tree features are
waveform-level location, spread, energy and quantile summaries plus observed
whole-snapshot values and lifecycle trend. CNN-LSTM/TCN convolve over raw
sensor time, then model lifecycle history. PatchTST applies per-channel
instance normalization, shared non-overlapping/overlapping temporal patches
and a shared Transformer; iTransformer embeds each physical channel's whole
time series as a variate token. MOMENT receives a 512-point waveform-history
view and its official pretrained encoder, with a new RUL head.

The controlled group comprises MLP, TCN, CNN-LSTM-Attention, PatchTST,
iTransformer, and MOMENT frozen/full. They use exactly the same HealthToken
input tensors; only model/training rules change. MOMENT uses external
pretraining and must always be reported separately from scratch models.

PatchTST's channel-independent patching follows the
[ICLR 2023 paper](https://arxiv.org/abs/2211.14730) and
[official implementation](https://github.com/yuqinie98/PatchTST).
iTransformer's variate-token attention follows the
[ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf)
and [official implementation](https://github.com/thuml/iTransformer).
MOMENT's 512-point, patch-based, pretrained encoder follows its
[ICML 2024 paper](https://openreview.net/pdf?id=FVvf69a5rx) and
[official code](https://github.com/moment-timeseries-foundation-model/moment).
The original papers mostly study forecasting or general representations;
scalar RUL readouts here are task adaptations, **not exact reproductions of
their forecasting heads**.

## Reproduce

The package intentionally does not duplicate NPZs or pretrained weights.
On this machine, `run.py` reuses the canonical prepared splits from
`experiments/A_main/suite/50_fair_baselines/prepared` and the staged official
MOMENT weight cache. On another machine, stage `data_phm` and run:

```powershell
python prepare_splits.py
python verify_design.py
python run_all.py --stage-only
python run_all.py --models native_xgboost native_random_forest
python run_all.py --models controlled_mlp controlled_tcn --gpu 0
python aggregate.py
python compare_trace.py
```

`run_all.py` without `--models` schedules the complete 675-cell grid.
It resumes completed cells. Model configurations are fixed in
`model_configs.json`; the stage manifest records every intended job.
`results_summary.json` reports missing cells explicitly.
`compare_trace.py` joins completed rows with TRACE
Scratch/Single/Three-domain and available partial/frozen results without
promoting incomplete rows to five-seed results.
