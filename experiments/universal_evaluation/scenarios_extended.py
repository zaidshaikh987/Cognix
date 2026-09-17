"""
Extended scenario machinery for the Cognix universal benchmark.

Adds:
  - Noise severity sweep (multiple sigma levels)
  - Agent-failure sweep (0, 1, 2 failed agents)
  - Conflicting agent scenario (label flipping)
  - Multi-failure scenario (noise + missing agents combined)

Wraps the existing DataGenerator without modifying it.
"""
import numpy as np
from typing import Tuple, List, Dict, Any

# Reuse existing DataGenerator
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognix.evaluation.scenarios import DataGenerator


# Noise severity levels for robustness sweep
NOISE_SEVERITY_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

# Agent failure counts for multi-agent resilience sweep
AGENT_FAILURE_COUNTS = [0, 1, 2]


class ExtendedDataGenerator:
    """
    Extends the base DataGenerator with additional conditions for the universal benchmark.
    All conditions are defined upfront and use fixed seeds.
    """

    def __init__(self, n_samples: int = 200, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        self.gen = DataGenerator(n_samples=n_samples, seed=seed)

    def generate_base(self, num_features: int = 4):
        return self.gen.generate_base(num_features=num_features)

    def apply_condition(self, X, y, condition: str, **kwargs):
        return self.gen.apply_condition(X, y, condition, **kwargs)

    def generate_conflicting(
        self, num_features: int = 4, flip_fraction: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Conflicting agents scenario: corrupts feature columns to produce
        contradictory signals. Implemented by flipping 50% of the labels
        that agents observe, which creates systematic disagreement.

        Returns (X, y, conflict_labels) where conflict_labels=1 means
        that sample has conflicting agent predictions.
        """
        rng = np.random.default_rng(self.seed + 9999)
        X, y = self.gen.generate_base(num_features=num_features)
        # Flip a subset of samples in feature column 0 (sign inversion)
        flip_idx = rng.choice(len(X), size=int(flip_fraction * len(X)), replace=False)
        X_out = X.copy()
        # Flip the sign of the primary feature for these samples
        X_out[flip_idx, 0] = -X_out[flip_idx, 0]
        conflict_labels = np.zeros(len(X))
        conflict_labels[flip_idx] = 1.0
        return X_out, y, conflict_labels

    def generate_multi_failure(
        self, num_features: int = 4, noise_level: float = 3.0,
        missing_agents: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multi-failure: HIGH_NOISE + MISSING_AGENT combined.
        """
        if missing_agents is None:
            missing_agents = [1, 2]
        X, y = self.gen.generate_base(num_features=num_features)
        rng = np.random.default_rng(self.seed + 777)
        # Apply noise
        X = X.copy()
        X[:, 1] += rng.normal(0, noise_level, len(X))
        # Apply missing agent feature zeroing
        for idx in missing_agents:
            if idx < X.shape[1]:
                X[:, idx] = 0.0
        ood_labels = np.zeros(len(X))  # not OOD, just degraded
        return X, y, ood_labels


def get_scenario_config(
    scenario_name: str,
    num_agents: int = 4,
    severity_idx: int = 0,
    failed_count: int = 0,
) -> Dict[str, Any]:
    """
    Return a configuration dict for the requested scenario.

    Parameters
    ----------
    scenario_name : one of NORMAL, HIGH_NOISE, MISSING_AGENT, OOD_SHIFT,
                    NOISE_SWEEP, AGENT_FAILURE_SWEEP, CONFLICTING, MULTI_FAILURE
    severity_idx  : index into NOISE_SEVERITY_LEVELS (for NOISE_SWEEP)
    failed_count  : number of failed agents (for AGENT_FAILURE_SWEEP)
    """
    cfg: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "noise_level": 0.0,
        "ood_severity": 0.0,
        "missing_agents": [],
        "conflict_fraction": 0.0,
        "description": "",
    }

    if scenario_name == "NORMAL":
        cfg["description"] = "Clean in-distribution data"

    elif scenario_name == "HIGH_NOISE":
        cfg["noise_level"] = 3.0
        cfg["description"] = "High Gaussian noise on feature 1 (sigma=3.0)"

    elif scenario_name == "MISSING_AGENT":
        cfg["missing_agents"] = [1]
        cfg["description"] = "Agent 1 completely failed (healthy=False)"

    elif scenario_name == "OOD_SHIFT":
        cfg["ood_severity"] = 10.0
        cfg["description"] = "50% samples with feature 1 shifted by +10 (OOD)"

    elif scenario_name == "NOISE_SWEEP":
        level = NOISE_SEVERITY_LEVELS[severity_idx]
        cfg["noise_level"] = level
        cfg["description"] = f"Noise sweep: sigma={level}"

    elif scenario_name == "AGENT_FAILURE_SWEEP":
        missing = list(range(failed_count))  # first N agents fail
        cfg["missing_agents"] = missing
        cfg["description"] = f"Agent failure sweep: {failed_count} agents failed"

    elif scenario_name == "CONFLICTING":
        cfg["conflict_fraction"] = 0.5
        cfg["description"] = "50% of samples have sign-flipped primary feature (conflicting signals)"

    elif scenario_name == "MULTI_FAILURE":
        cfg["noise_level"] = 3.0
        cfg["missing_agents"] = [1, 2]
        cfg["description"] = "Multi-failure: HIGH_NOISE (sigma=3) + 2 missing agents"

    return cfg
