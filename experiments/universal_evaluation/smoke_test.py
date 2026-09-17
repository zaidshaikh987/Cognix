"""
Cognix Universal Benchmark — Quick smoke test.

Runs 1 seed, 50 samples, all 3 graph modes, NORMAL scenario.
Verifies metrics compute, outputs save, no NaN/Inf, latency instruments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from run_universal_benchmark import main

if __name__ == "__main__":
    main(smoke=True)
