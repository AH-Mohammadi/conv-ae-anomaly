"""Benchmark the TFLite variants on this machine (intended: Raspberry Pi).

From the repository:  python scripts/benchmark_pi.py --bundle pi_bundle
Inside an exported bundle the generated benchmark_pi.py does the same with the bundle as default.
"""
from anomaly_detector.pibench_cli import main

if __name__ == "__main__":
    main()
