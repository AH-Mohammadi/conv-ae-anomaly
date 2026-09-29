# Conv-AE anomaly detection on SKAB (edge subsystem)

Work in progress, built iteration by iteration.

**Iteration 1:** data pipeline + evaluation harness + z-score baseline.
Scorecards are written to `reports/metrics/`. F1/FAR/MAR definitions are in
`src/anomaly_detector/metrics.py`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
git clone --depth 1 https://github.com/waico/SKAB.git data/raw/SKAB
pytest -q
python scripts/evaluate.py --config configs/config.yaml
```
