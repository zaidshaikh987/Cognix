import numpy as np
from typing import Any, Callable
from .base import UncertaintyEstimator, UncertaintyEstimate

class MonteCarloDropout(UncertaintyEstimator):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Reference: 
    Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: 
    Representing Model Uncertainty in Deep Learning. ICML 2016.
    """
    def __init__(self, num_passes: int = 50, task_type: str = "classification"):
        self.num_passes = num_passes
        self.task_type = task_type

    def _enable_dropout_pytorch(self, model: Any):
        """Helper to ensure dropout is active during inference for PyTorch models."""
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                # Ensure model is in train mode or specifically enable dropout layers
                model.train()
        except ImportError:
            pass

    def estimate(self, model: Callable, inputs: Any) -> UncertaintyEstimate:
        """
        Run T forward passes with dropout active to estimate uncertainty.
        Assumes the model is a callable that applies dropout natively when requested.
        """
        self._enable_dropout_pytorch(model)
        
        samples = []
        for _ in range(self.num_passes):
            out = model(inputs)
            # If tensor, convert to numpy
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            samples.append(out)
            
        samples = np.array(samples)  # Shape (T, ...)
        
        if len(samples) == 1:
            # Degenerate case
            return UncertaintyEstimate(
                aleatoric=0.0, epistemic=0.0, total=0.0, 
                raw_samples=samples, method="mc_dropout"
            )

        # Compute uncertainty depending on task type
        if self.task_type == "classification":
            # samples shape (T, num_classes), assuming probabilities (softmax applied)
            # Clip for numerical stability
            samples = np.clip(samples, 1e-9, 1 - 1e-9)
            
            p_mean = np.mean(samples, axis=0)
            
            # Total uncertainty: Entropy of expected probabilities H[E[p]]
            total_unc = -np.sum(p_mean * np.log(p_mean), axis=-1)
            
            # Aleatoric uncertainty: Expected entropy E[H[p]]
            entropies = -np.sum(samples * np.log(samples), axis=-1)
            aleatoric_unc = np.mean(entropies, axis=0)
            
            # Epistemic uncertainty: Total - Aleatoric (Mutual Information)
            epistemic_unc = total_unc - aleatoric_unc
            
            # Mean over batch if batch > 1
            total_val = float(np.mean(total_unc)) if total_unc.ndim > 0 else float(total_unc)
            alea_val = float(np.mean(aleatoric_unc)) if aleatoric_unc.ndim > 0 else float(aleatoric_unc)
            epi_val = float(np.mean(epistemic_unc)) if epistemic_unc.ndim > 0 else float(epistemic_unc)
            
        else:
            # Regression: samples shape (T,) or (T, features)
            var_total = np.var(samples, axis=0)
            # In standard MC Dropout regression (without explicit variance output),
            # all variance is treated as epistemic. Aleatoric must be learned explicitly.
            # Assuming homoscedastic aleatoric noise = 0 for simplicity if not provided.
            alea_val = 0.0
            epi_val = float(np.mean(var_total))
            total_val = epi_val

        # Handle NaNs
        if np.isnan(total_val):
            total_val, alea_val, epi_val = 0.0, 0.0, 0.0

        return UncertaintyEstimate(
            aleatoric=alea_val,
            epistemic=epi_val,
            total=total_val,
            raw_samples=samples,
            method="mc_dropout"
        )
