import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConformalPredictionSet:
    prediction_set: list[int]
    confidence: float
    quantile: float
    coverage_target: float

from cognix.core.interfaces import Calibrator

class ConformalPredictor(Calibrator):
    """
    Inductive Conformal Prediction.
    Reference: Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
    Reference for inductive CP: Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). 
    Inductive Confidence Machines for Regression.
    """
    def __init__(self):
        self.cal_scores: Optional[np.ndarray] = None
        self.n_cal = 0

    def fit(self, cal_outputs: np.ndarray, cal_labels: np.ndarray):
        """
        Fit on calibration set, compute nonconformity scores.
        Nonconformity score for classification: 1 - p_hat[true_class] (for softmax outputs)
        cal_outputs: (N, C) predicted probabilities
        cal_labels: (N,) true integer labels
        """
        N = cal_outputs.shape[0]
        # nonconformity score: 1 - prob(true_class)
        scores = 1.0 - cal_outputs[np.arange(N), cal_labels]
        self.cal_scores = np.sort(scores)
        self.n_cal = N

    def predict(self, test_outputs: np.ndarray, alpha: float = 0.05) -> list[ConformalPredictionSet]:
        """
        Predict prediction sets for new examples.
        test_outputs: (M, C) predicted probabilities
        """
        if self.cal_scores is None:
            raise RuntimeError("Conformal predictor must be calibrated before prediction.")
            
        M, C = test_outputs.shape
        q_idx = int(np.ceil((self.n_cal + 1) * (1 - alpha)))
        
        # Handle edge cases for quantile index
        if q_idx > self.n_cal:
            q_val = 1.0 # Max possible score
        else:
            q_val = self.cal_scores[q_idx - 1]
            
        prediction_sets = []
        for i in range(M):
            # nonconformity score for all classes y: 1 - p_hat[y]
            scores_y = 1.0 - test_outputs[i]
            pred_set = np.where(scores_y <= q_val)[0].tolist()
            
            prediction_sets.append(ConformalPredictionSet(
                prediction_set=pred_set,
                confidence=1.0 - alpha,
                quantile=q_val,
                coverage_target=1.0 - alpha
            ))
            
        return prediction_sets

def evaluate_coverage(prediction_sets: list[ConformalPredictionSet], true_labels: np.ndarray) -> float:
    """Empirical coverage of the prediction sets."""
    covered = 0
    for i, p_set in enumerate(prediction_sets):
        if true_labels[i] in p_set.prediction_set:
            covered += 1
    return covered / len(true_labels) if len(true_labels) > 0 else 0.0

class TwoStageConformalPredictor:
    """
    TwoStageConformalPredictor: individual threshold tau_i per agent, 
    then collective threshold on aggregated output.
    """
    def __init__(self):
        self.individual_predictors = {}
        self.ensemble_predictor = ConformalPredictor()

    def calibrate(self, agent_outputs: dict[str, np.ndarray], ensemble_outputs: np.ndarray, labels: np.ndarray):
        for agent_id, outputs in agent_outputs.items():
            cp = ConformalPredictor()
            cp.calibrate(outputs, labels)
            self.individual_predictors[agent_id] = cp
            
        self.ensemble_predictor.calibrate(ensemble_outputs, labels)
