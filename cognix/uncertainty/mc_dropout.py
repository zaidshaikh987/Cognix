from typing import Any
import numpy as np

from cognix.core.interfaces import UncertaintyEstimator
from cognix.core.types import UncertaintyResult

class MCDropout(UncertaintyEstimator):
    """
    Monte Carlo Dropout Uncertainty Estimator.
    
    Uncertainty decomposition (Bernoulli predictive-variance decomposition):
        p_bar = mean(p_t)  for t = 1..T
        U_epistemic = mean((p_t - p_bar)^2)  [variance of MC samples]
        U_aleatoric = mean(p_t * (1 - p_t))  [expected Bernoulli variance]
        U_total     = U_epistemic + U_aleatoric
    """
    
    def __init__(self, T: int = 30):
        self.T = T
        
    def estimate(self, model: Any, observation: np.ndarray) -> UncertaintyResult:
        import torch
        model.train()  # Activate dropout
        
        preds = []
        with torch.no_grad():
            x_t = torch.FloatTensor(observation)
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
                
            for _ in range(self.T):
                out = model(x_t).item()
                preds.append(out)
                
        preds = np.array(preds)
        
        p_bar = np.mean(preds)
        epistemic = np.var(preds)
        aleatoric = np.mean(preds * (1 - preds))
        
        return UncertaintyResult(
            prediction=float(p_bar),
            epistemic=float(epistemic),
            aleatoric=float(aleatoric),
            total=float(epistemic + aleatoric)
        )
