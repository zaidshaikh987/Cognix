import numpy as np
from typing import Tuple, List

class DataGenerator:
    """Generates synthetic multi-modal data for heterogeneous agents."""

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate_base(self, num_features: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """P_train(X): Standard normally distributed features."""
        X = self.rng.normal(0, 1, (self.n_samples, num_features))
        
        if num_features >= 3:
            logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] ** 2
        else:
            logits = 1.5 * X[:, 0]
            
        probs = 1 / (1 + np.exp(-logits))
        y = (self.rng.random(self.n_samples) < probs).astype(float)
        return X, y

    def apply_condition(
        self, X: np.ndarray, y: np.ndarray, condition: str, 
        noise_level: float = 3.0, ood_severity: float = 10.0,
        missing_agent_idx: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply OOD/Noise conditions to feature matrix."""
        X = X.copy()
        ood_labels = np.zeros(len(X))

        if condition == "NORMAL":
            pass
        elif condition == "HIGH_NOISE":
            X[:, 1] += self.rng.normal(0, noise_level, len(X))
        elif condition == "OOD_SHIFT":
            shift_idx = self.rng.choice(len(X), size=int(0.5 * len(X)), replace=False)
            X[shift_idx, 1] += ood_severity
            ood_labels[shift_idx] = 1.0
        elif condition == "MISSING_AGENT":
            if missing_agent_idx is None:
                missing_agent_idx = [1]
            for idx in missing_agent_idx:
                if idx < X.shape[1]:
                    X[:, idx] = 0.0

        return X, y, ood_labels
