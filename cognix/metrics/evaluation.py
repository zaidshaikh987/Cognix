"""
COGNIX Metrics & Evaluation Module.

Contains mathematically rigorous, centralized metric calculations 
for evaluating calibration, accuracy, and out-of-distribution detection.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import time

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def calculate_ece(predictions: np.ndarray, ground_truths: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE) for binary classification.
    
    ECE = sum_{m=1}^M (|B_m| / n) * |acc(B_m) - conf(B_m)|
    """
    if len(predictions) == 0:
        return 0.0
        
    bin_boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    
    for i in range(bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i+1]
        
        mask = (predictions >= lower) & (predictions <= (upper if i == bins-1 else upper - 1e-8))
        if np.any(mask):
            bin_preds = predictions[mask]
            bin_truths = ground_truths[mask]
            
            confidences = np.maximum(bin_preds, 1.0 - bin_preds)
            accuracies = (np.round(bin_preds) == bin_truths).astype(float)
            
            avg_conf = np.mean(confidences)
            avg_acc = np.mean(accuracies)
            
            weight = len(bin_preds) / len(predictions)
            ece += weight * np.abs(avg_acc - avg_conf)
            
    return float(ece)

def accuracy(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    """Calculate binary accuracy."""
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(np.round(predictions) == ground_truths))

def brier_score(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    """Calculate Brier Score (Mean Squared Error for probabilities)."""
    if len(predictions) == 0:
        return 0.0
    if _SKLEARN_AVAILABLE:
        return float(brier_score_loss(ground_truths, predictions))
    return float(np.mean((predictions - ground_truths) ** 2))

def coverage(predictions: np.ndarray, thresholds: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate empirical coverage for conformal prediction sets.
    Assuming thresholds define a prediction set [lower, upper].
    For binary classification with scores p, if threshold tau is given,
    we cover if p >= tau (class 1) or 1-p >= tau (class 0).
    """
    pass # Defined properly in conformal calibrator, here for completeness

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Receiver Operating Characteristic curve."""
    if len(np.unique(labels)) < 2 or not _SKLEARN_AVAILABLE:
        return float('nan')
    return float(roc_auc_score(labels, scores))

def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve."""
    if len(np.unique(labels)) < 2 or not _SKLEARN_AVAILABLE:
        return float('nan')
    return float(average_precision_score(labels, scores))

def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """False Positive Rate at 95% True Positive Rate."""
    if len(np.unique(labels)) < 2 or not _SKLEARN_AVAILABLE:
        return float('nan')
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    if idx < len(fpr):
        return float(fpr[idx])
    return float('nan')

def communication_reduction(messages_sent: int, n_agents: int, directed: bool = True) -> float:
    """
    Calculate communication reduction ratio.
    Max possible messages = N*(N-1) if directed, N*(N-1)/2 if undirected.
    """
    if n_agents <= 1:
        return 0.0
    max_msgs = n_agents * (n_agents - 1)
    if not directed:
        max_msgs /= 2.0
    
    if max_msgs == 0:
        return 0.0
    return float(1.0 - (messages_sent / max_msgs))


class LatencyTracker:
    """Tracks latency percentiles across repeated executions."""
    def __init__(self):
        self.records: Dict[str, List[float]] = {}
        
    def record(self, stage: str, elapsed_ms: float):
        if stage not in self.records:
            self.records[stage] = []
        self.records[stage].append(elapsed_ms)
        
    def get_percentiles(self, stage: str) -> Dict[str, float]:
        if stage not in self.records or not self.records[stage]:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        data = np.array(self.records[stage])
        return {
            "mean": float(np.mean(data)),
            "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99))
        }
    
    def get_all(self) -> Dict[str, Dict[str, float]]:
        return {stage: self.get_percentiles(stage) for stage in self.records}
