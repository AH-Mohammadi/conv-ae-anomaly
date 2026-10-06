# Conv-AE anomaly detection on SKAB (edge subsystem)

Work in progress, built iteration by iteration.

- **Iteration 1:** data pipeline, reference-only normalization, windowing, z-score baseline, F1/FAR/MAR harness.
- **Iteration 2:** offline Conv1D autoencoder trained on anomaly-free reference windows, scored by the same harness.
- **Iteration 3:** TFLite conversion (FP32 and full-integer INT8), compared against the FP32 Keras model.
- **Iteration 4:** Raspberry Pi batch benchmark (latency, memory, CPU, thermals, desktop-vs-device agreement). Recommendation: deploy `fp32_tflite` (see report).
- **Iteration 5:** streaming inference (ring buffer, startup handling, real-time replay, stability checks).
- **Iteration 6:** score calibration ([0,1]) and a versioned structured output record.

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

## Streaming inference (Iteration 5)

```bash
python scripts/replay_stream.py --speed 0                        # as fast as possible, all files once
python scripts/replay_stream.py --files valve1/0.csv --speed 1    # one file, real time
python scripts/replay_stream.py --loops 10 --speed 0              # soak test
```

- `StreamingDetector` (`stream.py`) takes samples one at a time: the first `reference_rows` fit the
  per-episode normalizer (no score emitted), then every sample is windowed through a `RingBuffer`
  (`ringbuffer.py`, no reallocation after construction) and scored. One `StreamingDetector` per SKAB
  file (a fresh reference/normalizer per "episode"), matching the offline per-file protocol exactly.
- **Verified exact parity with the offline harness**: streamed one-file-at-a-time, the detector
  reproduces the SAME scores, predictions and labels as `evaluate.py`/`compress.py` to float tolerance
  (test, and confirmed on real SKAB: 23,421 pooled points, F1 0.7884/FAR 0.3650/MAR 0.1482, identical
  streamed vs offline).
- `replay.py` paces samples using SKAB's real recorded timestamps (mostly 1s, some 2s, a few large gaps
  from concatenated recording sessions), scaled by `--speed`; `0` runs as fast as possible. It tracks
  deadline overruns (processing slower than the inter-arrival gap) and periodic RSS for a growth check.
- 10-loop soak test (330 episodes, 234,210 scored windows, sandbox): 0 missed windows, 0 deadline
  overruns, RSS 495.6 -> 501.3 MB (sandbox runs full TensorFlow; a Pi deployment with ai-edge-litert
  would start much smaller). The deployed variant is `fp32_tflite`, per the Iteration 4 Pi results.
- Known simplification: offline/streaming parity is exact for `stride=1` (used everywhere in this
  project); for `stride>1` the streaming stride phase is anchored to the start of scoring rather than
  to sample 0, which is documented in `stream.py` rather than engineered away.

## Score calibration and structured output (Iteration 6)

```bash
python scripts/convert_tflite.py --config configs/conv_ae.yaml   # now also fits+saves calibration per variant
python scripts/replay_stream.py --speed 0                         # now emits stream_output_record_v1 JSONL
```

- **Calibration** (`calibration.py`): `score = sigmoid(log((error+eps)/(threshold+eps)) / scale)`, with
  `scale` fit as the standard deviation of that log-ratio over anomaly-free VALIDATION errors only (no
  test data). `score(threshold) == 0.5` exactly, so `score > 0.5` reproduces the raw `error > threshold`
  decision with no change to F1/FAR/MAR (verified bit-for-bit on the real dataset: 23,421/23,421 records
  agree). Log space was used because test-region errors reach roughly 1000x the threshold on some files
  (the out-of-distribution-input finding from Iteration 3); a linear scale saturates almost every such
  point to 1.0 and loses the gradation a severity score should carry.
- Each TFLite variant gets its own calibration (like its own threshold), stored in `tflite_meta.json`
  and `compression_report.json` under `score_calibration`.
- **Structured output** (`records.py`): every streamed detection is a `stream_output_record_v1` JSON
  object: `schema_version, timestamp, window_id, score (calibrated), raw_error, threshold, is_anomaly,
  model_version, config_version, processing_ns, stage, file`. This is the project's "Knowledge" payload
  for a future Coordinator (not implemented here).
- `model_version` names the deployed artifact including which TFLite variant (e.g.
  `conv_ae_v1:fp32_tflite`) -- quantization changes the weights/op set, so it is artifact-dependent, same
  classification as window size or filter count. `config_version` is a short fingerprint of the currently
  active runtime-adjustable parameters (today: just the threshold); it is the hook Iteration 7's runtime
  reconfiguration will use to mark records as produced under a new configuration, without a new
  model_version.
