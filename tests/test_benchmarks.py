import pytest
import os
import json
from cognix.evaluation.benchmark import BenchmarkConfig, ExperimentResult
from cognix.evaluation.runner import BenchmarkRunner
from cognix.evaluation.reporting import JSONReporter, CSVReporter

def mock_evaluation_fn(seed: int, config: BenchmarkConfig):
    return {
        "accuracy": 0.85 + (seed * 0.01),
        "ece": 0.05,
        "mean_epistemic": 0.1,
        "mean_aleatoric": 0.2,
        "latency_ms": 150.0,
        "metadata": {"test_seed": seed}
    }

def test_benchmark_runner_and_reporting(tmp_path):
    """Test that the benchmark runner correctly aggregates seeds and reporters write files."""
    json_path = os.path.join(tmp_path, "results.json")
    csv_path = os.path.join(tmp_path, "results.csv")
    
    reporters = [
        JSONReporter(json_path),
        CSVReporter(csv_path)
    ]
    
    runner = BenchmarkRunner(reporters=reporters)
    
    config = BenchmarkConfig(
        experiment_id="test_exp_01",
        scenario_name="test_scenario",
        seeds=[1, 2, 3],
        num_samples=100,
        num_agents=3,
        graph_type="EpistemicGAT",
        fusion_type="EpistemicWeighted",
        calibration_type="Conformal"
    )
    
    results = runner.run(config, mock_evaluation_fn)
    
    assert len(results) == 3
    assert results[0].seed == 1
    assert results[0].accuracy == 0.86  # 0.85 + 0.01 * 1
    
    # Check JSON
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
        assert len(data) == 3
        assert data[0]["accuracy"] == 0.86
        
    # Check CSV
    assert os.path.exists(csv_path)
    with open(csv_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 4  # 1 header + 3 rows
        assert "accuracy" in lines[0]
        assert "0.86" in lines[1]
