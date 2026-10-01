# Conv-AE anomaly detection on SKAB (edge subsystem)

Work in progress, built iteration by iteration.

- **Iteration 1:** data pipeline, reference-only normalization, windowing, z-score baseline, F1/FAR/MAR harness.
- **Iteration 2:** offline Conv1D autoencoder trained on anomaly-free reference windows, scored by the same harness.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml]"                              # ml = TensorFlow (Python 3.10-3.12 recommended)
git clone --depth 1 https://github.com/waico/SKAB.git data/raw/SKAB
pytest -q
python scripts/evaluate.py --config configs/config.yaml     # baseline
python scripts/train.py    --config configs/conv_ae.yaml    # Conv-AE
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
