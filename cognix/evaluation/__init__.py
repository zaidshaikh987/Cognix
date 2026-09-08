"""
COGNIX Evaluation Framework

Provides reproducible benchmarking, ablation running, and statistical validation.
"""
from cognix.evaluation.benchmark import BenchmarkConfig, ExperimentResult
from cognix.evaluation.runner import BenchmarkRunner
from cognix.evaluation.reporting import JSONReporter, CSVReporter, BenchmarkReporter

__all__ = [
    "BenchmarkConfig",
    "ExperimentResult",
    "BenchmarkRunner",
    "BenchmarkReporter",
    "JSONReporter",
    "CSVReporter"
]
