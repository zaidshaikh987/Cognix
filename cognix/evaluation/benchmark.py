from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

@dataclass
class BenchmarkConfig:
    experiment_id: str
    scenario_name: str
    seeds: List[int]
    num_samples: int
    num_agents: int
    graph_type: str
    fusion_type: str
    calibration_type: str
    dataset: str = "synthetic"
    noise_level: float = 0.0
    ood_severity: float = 0.0
    missing_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ExperimentResult:
    experiment_id: str
    git_hash: Optional[str]
    timestamp: str
    seed: int
    scenario_name: str
    
    # Configurations used
    graph_type: str
    fusion_type: str
    calibration_type: str
    
    # Core Metrics
    accuracy: float
    brier_score: float
    nll: float
    ece: float
    
    # Uncertainty & Calibration
    mean_epistemic: float
    mean_aleatoric: float
    empirical_coverage: float
    mean_set_size: float
    
    # System
    escalation_rate: float
    latency_ms: float
    
    # Metadata & specific failure logs
    failure_conditions: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
