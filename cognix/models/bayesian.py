"""
Bayes by Backprop (BBB) / Variational Inference implementation.
Reference: Blundell et al. (2015). Weight Uncertainty in Neural Networks.
"""
import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

class BayesLinear(nn.Module):
    """
    Bayesian Linear layer with Mean-Field Gaussian Approximation and Reparameterization Trick.
    """
    def __init__(self, in_features: int, out_features: int, prior_mu: float = 0.0, prior_sigma: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior parameters
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        # Variational parameters (Weight)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Variational parameters (Bias)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize mu to match standard linear layer initialization
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        # Initialize rho such that initial sigma is small
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass using the reparameterization trick.
        w = mu + log(1 + exp(rho)) * epsilon
        """
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        if sample or self.training:
            weight_eps = torch.randn_like(weight_sigma)
            bias_eps = torch.randn_like(bias_sigma)
            
            weight = self.weight_mu + weight_sigma * weight_eps
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
        
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between variational posterior and prior.
        D_KL(q(w|theta) || p(w)) for a Gaussian prior.
        """
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # KL for weights
        kl_weight = 0.5 * (
            2 * torch.log(self.prior_sigma / weight_sigma) +
            (weight_sigma**2 + (self.weight_mu - self.prior_mu)**2) / self.prior_sigma**2 - 1
        ).sum()
        
        # KL for bias
        kl_bias = 0.5 * (
            2 * torch.log(self.prior_sigma / bias_sigma) +
            (bias_sigma**2 + (self.bias_mu - self.prior_mu)**2) / self.prior_sigma**2 - 1
        ).sum()
        
        return kl_weight + kl_bias

class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) Loss.
    Loss = NLL + kl_weight * sum(KL)
    """
    def __init__(self, criterion: nn.Module, kl_weight: float = 1.0):
        super().__init__()
        self.criterion = criterion
        self.kl_weight = kl_weight
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, model: nn.Module) -> torch.Tensor:
        nll = self.criterion(outputs, targets)
        
        kl_sum = 0.0
        for module in model.modules():
            if isinstance(module, BayesLinear):
                kl_sum += module.kl_divergence()
                
        return nll + self.kl_weight * kl_sum
