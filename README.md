# Conv-AE anomaly detection on SKAB (edge subsystem)

Work in progress, built iteration by iteration.

- **Iteration 1:** data pipeline, reference-only normalization, windowing, z-score baseline, F1/FAR/MAR harness.
- **Iteration 2:** offline Conv1D autoencoder trained on anomaly-free reference windows, scored by the same harness.
- **Iteration 3:** TFLite conversion (FP32 and full-integer INT8), compared against the FP32 Keras model.
- **Iteration 4:** Raspberry Pi batch benchmark (latency, memory, CPU, thermals, desktop-vs-device agreement).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml]"                              # ml = TensorFlow (Python 3.10-3.12 recommended)
git clone --depth 1 https://github.com/waico/SKAB.git data/raw/SKAB
pytest -q
python scripts/evaluate.py --config configs/config.yaml     # baseline
python scripts/train.py    --config configs/conv_ae.yaml    # Conv-AE
python scripts/convert_tflite.py --config configs/conv_ae.yaml   # TFLite + FP32/INT8 comparison
python scripts/investigate_quantization.py                       # optional: alternative strategies
```

Scorecards: `reports/metrics/scorecard_<model>.json`. Trained artifacts: `models/conv_ae_v1/`
(`model.keras`, `artifact.json`, `training_history.json`, `scores.npz`).

## Evaluation protocol (this repo)

- 33 of 34 labeled SKAB files; `other/2.csv` is excluded (first anomaly at row 104, no valid reference).
- Reference = first 400 rows of each file. Normalizer fitted on those rows only, per file.
- Conv-AE: rows 0-299 train, rows 300-399 validation (early stopping + threshold). One global model.
- Threshold = 99th percentile of anomaly-free validation reconstruction errors. No test labels are used for any decision.
- Test = windows ending at row >= 400; label taken at window end; point-wise metrics pooled across files.
- `f1 = 2TP/(2TP+FP+FN)`, `far = FP/(FP+TN)`, `mar = FN/(FN+TP)`, all fractions in [0,1] (SKAB reports FAR/MAR in %).

## Differences from the published SKAB Conv-AE benchmark (core/Conv_AE.py + notebook)

| | SKAB benchmark | this repo |
|---|---|---|
| train/test split | first 400 rows train | same |
| models | one per file | one global model |
| window | 60 | 32 |
| architecture | strided Conv1D + Conv1DTranspose, dropout | Conv1D + pooling + dense latent + upsampling |
| score | sum over sensors of mean abs error | mean squared error |
| threshold | 0.999 quantile of train residuals x 4/3 | 0.99 quantile of held-out validation errors |
| decision rule | flag only after a run of consecutive anomalous windows | single-window threshold |

Numbers from the two setups are therefore **not directly comparable**.

## Hyperparameter classes

- hot-swappable: `detector.threshold`
- model-artifact dependent: `windowing.window_size`, `model.latent_dim`, `model.filter_count`, `model.kernel_size`, weights

## Model compression (Iteration 3)

`scripts/convert_tflite.py` writes `detector_fp32.tflite`, `detector_int8.tflite`,
`detector_int8_wide.tflite`, `detector.tflite` (the selected variant), `tflite_meta.json`
(per-variant threshold and I/O quantization) and `reports/metrics/compression_report.json`.

- Each variant gets **its own threshold** from its own anomaly-free validation errors (quantization shifts the error scale).
- INT8 is full-integer with int8 input/output. Standard calibration uses 500 anomaly-free training windows.
- Acceptance criteria were fixed before measuring: |delta| <= 0.03 for F1, FAR and MAR, and ROC-AUC drop <= 0.02 versus FP32 Keras.
- Finding: with reference-only z-scoring, test inputs reach |z| ~ 250 (sensors with tiny reference std), while anomaly-free data stays within |z| ~ 5. Standard INT8 calibration clips and saturates on such inputs and fails the criteria. The **wide** variant adds copies of the calibration windows scaled by U(1, 4) and passes. It was introduced after looking at test metrics of the standard model (two widths tried), so its numbers carry mild selection bias.
- Hot-swappable vs artifact-dependent: the INT8 input scale/zero-point and all weights belong to the model artifact; only the threshold is runtime-adjustable.

## Raspberry Pi benchmark (Iteration 4)

```bash
python scripts/export_pi_bundle.py --out pi_bundle     # desktop: bundle with .tflite, windows, desktop reference scores
# copy pi_bundle/ to the Pi (64-bit OS, python3, numpy, ai-edge-litert or tflite-runtime), then on the Pi:
python3 benchmark_pi.py                                # writes pi_benchmark_report.json
```

- The bundle is self-contained and numpy-only on the Pi (no TensorFlow, pandas or SKAB).
- Each variant runs in its own subprocess (per-variant peak RSS). Latency is measured from Python:
  `invoke` = `interpreter.invoke()` only; `end_to_end` = quantize + invoke + dequantize + error.
- Desktop-vs-Pi agreement criteria (fixed before measuring): >= 99.9% identical decisions at the variant's threshold,
  |delta| <= 0.002 for pooled F1/FAR/MAR, and (FP32 only) max relative score diff <= 1e-4.
- The report records the device model, OS/kernel, runtime package versions, XNNPACK presence in the runtime log,
  temperature/frequency/throttle flags and a sustained-run drift check. `is_raspberry_pi` is false on any other machine.
- Benchmark results from a non-Pi machine must not be reported as Pi results.
