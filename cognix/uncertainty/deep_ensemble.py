from typing import Any
import numpy as np

from cognix.core.interfaces import UncertaintyEstimator
from cognix.core.types import UncertaintyResult

class DeepEnsemble(UncertaintyEstimator):
    """
    Deep Ensemble Uncertainty Estimator (Stub).
    
    Expects an ensemble of models to compute prediction variance.
    """
    
    def __init__(self, num_models: int = 5):
        self.num_models = num_models
        
    def estimate(self, model_ensemble: list[Any], observation: np.ndarray) -> UncertaintyResult:
        import torch
        preds = []
        with torch.no_grad():
            x_t = torch.FloatTensor(observation)
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            
            for model in model_ensemble:
                out = model(x_t).item()
                preds.append(out)
                
        preds = np.array(preds)
        p_bar = float(np.mean(preds))
        epistemic = float(np.var(preds))
        
        return UncertaintyResult(
            prediction=p_bar,
            epistemic=epistemic,
            aleatoric=0.0,
            total=epistemic
        )
