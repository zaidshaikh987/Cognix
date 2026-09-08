"""
Gaussian Mixture Models (GMM) for Density Estimation and UQ.
"""
import numpy as np

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    pass

class GMMUncertainty:
    """
    Fits a GMM to training features to act as a density estimator.
    Low density -> High Epistemic Uncertainty (OOD).
    """
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.is_fitted = False
        
    def fit(self, features: np.ndarray):
        """Fits the GMM on training features."""
        self.gmm.fit(features)
        self.is_fitted = True
        
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """
        Returns log-likelihood of each sample.
        Lower log-likelihood means higher uncertainty.
        """
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before scoring.")
        return self.gmm.score_samples(features)
        
    def predict_uncertainty(self, features: np.ndarray) -> np.ndarray:
        """
        Converts log-likelihood to a normalized uncertainty score [0, 1].
        """
        log_probs = self.score_samples(features)
        # Invert and normalize (heuristic mapping)
        unc = -log_probs
        unc = np.clip((unc - np.min(unc)) / (np.max(unc) - np.min(unc) + 1e-8), 0, 1)
        return unc
