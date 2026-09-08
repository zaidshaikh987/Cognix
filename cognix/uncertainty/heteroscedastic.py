"""
Heteroscedastic Loss for Aleatoric Uncertainty.
Reference: Kendall & Gal (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
"""
try:
    import torch
    import torch.nn as nn
except ImportError:
    pass

class HeteroscedasticLoss(nn.Module):
    """
    Computes regression loss where the model predicts both the mean and the variance (aleatoric uncertainty).
    L = 0.5 * exp(-log_var) * (y - mu)^2 + 0.5 * log_var
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_mu: torch.Tensor, pred_log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_mu: (N, D) predicted mean
            pred_log_var: (N, D) predicted log variance (log(sigma^2))
            target: (N, D) ground truth
        """
        # Exponentiate log_var for stable division
        inv_var = torch.exp(-pred_log_var)
        
        # Loss computation
        sq_error = (target - pred_mu)**2
        loss = 0.5 * inv_var * sq_error + 0.5 * pred_log_var
        
        return loss.mean()
