import numpy as np
from typing import Any, List, Callable
from .base import UncertaintyEstimator, UncertaintyEstimate

class DeepEnsemble(UncertaintyEstimator):
    """
    Deep Ensembles for uncertainty estimation.
    
    Reference:
    Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). 
    Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.
    """
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type

    def estimate(self, models: List[Callable], inputs: Any) -> UncertaintyEstimate:
        """
        Run inference over a list of models (ensemble members) to estimate uncertainty.
        
        Args:
            models: List of callable models.
            inputs: Model inputs.
        """
        if not models:
            raise ValueError("Ensemble must contain at least one model.")
            
        if len(models) == 1:
            import warnings
            warnings.warn("DeepEnsemble called with M=1 model. Returning zero epistemic uncertainty.")
        
        samples = []
        for model in models:
            out = model(inputs)
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            samples.append(out)
            
        samples = np.array(samples)  # Shape (M, ...)
        
        if len(models) == 1:
            return UncertaintyEstimate(
                aleatoric=0.0, epistemic=0.0, total=0.0, 
                raw_samples=samples, method="deep_ensemble"
            )

        if self.task_type == "classification":
            # Assuming output is probabilities
            samples = np.clip(samples, 1e-9, 1 - 1e-9)
            p_mean = np.mean(samples, axis=0)
            
            # Total uncertainty = H[p_mean]
            total_unc = -np.sum(p_mean * np.log(p_mean), axis=-1)
            
            # Aleatoric uncertainty = mean of individual entropies E[H[p_m]]
            entropies = -np.sum(samples * np.log(samples), axis=-1)
            aleatoric_unc = np.mean(entropies, axis=0)
            
            # Epistemic = Total - Aleatoric (Jensen-Shannon divergence)
            epistemic_unc = total_unc - aleatoric_unc
            
            total_val = float(np.mean(total_unc)) if total_unc.ndim > 0 else float(total_unc)
            alea_val = float(np.mean(aleatoric_unc)) if aleatoric_unc.ndim > 0 else float(aleatoric_unc)
            epi_val = float(np.mean(epistemic_unc)) if epistemic_unc.ndim > 0 else float(epistemic_unc)
            
        else:
            # Regression: samples represent predictions
            var_pred = np.var(samples, axis=0)
            epi_val = float(np.mean(var_pred))
            alea_val = 0.0 # Standard deep ensemble without variance head
            total_val = epi_val

        return UncertaintyEstimate(
            aleatoric=alea_val,
            epistemic=max(0.0, epi_val),
            total=max(0.0, total_val),
            raw_samples=samples,
            method="deep_ensemble"
        )
