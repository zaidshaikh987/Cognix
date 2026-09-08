"""
Distribution Shift Adaptation & Off-Policy Conformal Prediction.
"""
import numpy as np

class DriftDetector:
    """
    Detects distribution shift using a sliding window Kolmogorov-Smirnov (KS) test approximation.
    """
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = []
        self.current_window = []
        
    def update_reference(self, data: float):
        self.reference_data.append(data)
        
    def check_drift(self, new_data: float) -> bool:
        """
        Returns True if drift is detected.
        """
        self.current_window.append(new_data)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
            
        if len(self.reference_data) < self.window_size or len(self.current_window) < self.window_size:
            return False
            
        try:
            from scipy.stats import ks_2samp
            stat, p_value = ks_2samp(self.reference_data[-self.window_size:], self.current_window)
            return p_value < self.threshold
        except ImportError:
            # Fallback heuristic if scipy not available
            ref_mean = np.mean(self.reference_data[-self.window_size:])
            curr_mean = np.mean(self.current_window)
            return abs(ref_mean - curr_mean) > 0.5

class OffPolicyConformal:
    """
    Off-Policy Conformal Prediction (MA-COPP approximation).
    Reweights calibration scores based on policy shift ratios.
    """
    def __init__(self):
        self.cal_scores = []
        self.weights = []
        
    def calibrate(self, scores: np.ndarray, likelihood_ratios: np.ndarray):
        """
        scores: Nonconformity scores
        likelihood_ratios: pi_eval(a|s) / pi_behavior(a|s)
        """
        self.cal_scores = np.array(scores)
        self.weights = np.array(likelihood_ratios)
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
    def get_quantile(self, alpha: float) -> float:
        """
        Computes weighted quantile.
        """
        if len(self.cal_scores) == 0:
            return 1.0
            
        # Sort scores and corresponding weights
        idx = np.argsort(self.cal_scores)
        sorted_scores = self.cal_scores[idx]
        sorted_weights = self.weights[idx]
        
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Find index where cumulative weight exceeds (1 - alpha)
        target = 1.0 - alpha
        for i, cw in enumerate(cumulative_weights):
            if cw >= target:
                return sorted_scores[i]
                
        return sorted_scores[-1]
