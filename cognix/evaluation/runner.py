from typing import List, Callable, Dict, Any
from datetime import datetime, timezone
import subprocess

from cognix.evaluation.benchmark import BenchmarkConfig, ExperimentResult
from cognix.evaluation.reporting import BenchmarkReporter

class BenchmarkRunner:
    """
    Executes a benchmark configuration across multiple seeds and reports results.
    """
    
    def __init__(self, reporters: List[BenchmarkReporter]):
        self.reporters = reporters
        
    def _get_git_hash(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            return "unknown"
            
    def run(self, config: BenchmarkConfig, evaluation_fn: Callable[[int, BenchmarkConfig], Dict[str, Any]]) -> List[ExperimentResult]:
        """
        evaluation_fn should take (seed, config) and return a dictionary with all the core metrics:
        {
            "accuracy": float,
            "brier_score": float,
            "nll": float,
            "ece": float,
            "mean_epistemic": float,
            "mean_aleatoric": float,
            "empirical_coverage": float,
            "mean_set_size": float,
            "escalation_rate": float,
            "latency_ms": float,
            "failure_conditions": List[str],
            "metadata": Dict[str, Any]
        }
        """
        git_hash = self._get_git_hash()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        results = []
        for seed in config.seeds:
            print(f"[{timestamp}] Running {config.experiment_id} - {config.scenario_name} (Seed: {seed})")
            
            # Execute the actual experiment simulation/evaluation for this seed
            metrics = evaluation_fn(seed, config)
            
            result = ExperimentResult(
                experiment_id=config.experiment_id,
                git_hash=git_hash,
                timestamp=timestamp,
                seed=seed,
                scenario_name=config.scenario_name,
                graph_type=config.graph_type,
                fusion_type=config.fusion_type,
                calibration_type=config.calibration_type,
                accuracy=metrics.get("accuracy", 0.0),
                brier_score=metrics.get("brier_score", 0.0),
                nll=metrics.get("nll", 0.0),
                ece=metrics.get("ece", 0.0),
                mean_epistemic=metrics.get("mean_epistemic", 0.0),
                mean_aleatoric=metrics.get("mean_aleatoric", 0.0),
                empirical_coverage=metrics.get("empirical_coverage", 0.0),
                mean_set_size=metrics.get("mean_set_size", 0.0),
                escalation_rate=metrics.get("escalation_rate", 0.0),
                latency_ms=metrics.get("latency_ms", 0.0),
                failure_conditions=metrics.get("failure_conditions", []),
                metadata=metrics.get("metadata", {})
            )
            results.append(result)
            
        # Report results
        for reporter in self.reporters:
            reporter.report(results)
            
        return results
