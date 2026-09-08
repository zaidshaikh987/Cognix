import numpy as np
import scipy.stats as stats
from .base import BeliefState

class BayesianBelief:
    """
    Beta-distribution belief representation.
    Reference: De Finetti (1974), Theory of Probability.
    
    Beta(alpha, beta) represents a belief over a binary/probability event.
    For multi-class, Dirichlet distribution is a natural extension.
    """
    def __init__(self, agent_id: str, alpha: float = 1.0, beta_param: float = 1.0, classes: int = 2):
        self.agent_id = agent_id
        if classes == 2:
            self.alpha = float(alpha)
            self.beta_param = float(beta_param)
            self.dirichlet_alphas = None
        else:
            self.alpha = float(alpha)
            self.beta_param = float(beta_param)
            self.dirichlet_alphas = np.ones(classes) * alpha
            
        self.classes = classes

    @classmethod
    def from_prediction(cls, agent_id: str, p: float, n_effective: float = 1.0) -> 'BayesianBelief':
        """Initialize from a probability prediction."""
        alpha = p * n_effective
        beta_param = (1 - p) * n_effective
        return cls(agent_id, alpha, beta_param, classes=2)

    @classmethod
    def from_multiclass_prediction(cls, agent_id: str, probs: np.ndarray, n_effective: float = 1.0) -> 'BayesianBelief':
        """Initialize from a multiclass prediction using Dirichlet."""
        bb = cls(agent_id, classes=len(probs))
        bb.dirichlet_alphas = probs * n_effective
        return bb

    def update(self, observation: int):
        """
        Bayesian update: posterior Beta(alpha + successes, beta + failures)
        or Dirichlet posterior.
        observation: index of the true class
        """
        if self.classes == 2:
            if observation == 1:
                self.alpha += 1.0
            elif observation == 0:
                self.beta_param += 1.0
            else:
                raise ValueError("Observation must be 0 or 1 for binary classification")
        else:
            self.dirichlet_alphas[observation] += 1.0

    @property
    def posterior_mean(self) -> np.ndarray:
        if self.classes == 2:
            p = self.alpha / (self.alpha + self.beta_param)
            return np.array([1 - p, p])
        else:
            return self.dirichlet_alphas / np.sum(self.dirichlet_alphas)

    @property
    def posterior_variance(self) -> np.ndarray:
        if self.classes == 2:
            a, b = self.alpha, self.beta_param
            var = (a * b) / ((a + b)**2 * (a + b + 1))
            return np.array([var, var])
        else:
            a0 = np.sum(self.dirichlet_alphas)
            return self.dirichlet_alphas * (a0 - self.dirichlet_alphas) / (a0**2 * (a0 + 1))

    def credible_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Returns the (alpha/2, 1-alpha/2) credible interval for the positive class."""
        if self.classes == 2:
            lower = stats.beta.ppf(alpha / 2, self.alpha, self.beta_param)
            upper = stats.beta.ppf(1 - alpha / 2, self.alpha, self.beta_param)
            return float(lower), float(upper)
        else:
            raise NotImplementedError("Credible interval not implemented for multiclass")

    def to_belief_state(self) -> BeliefState:
        return BeliefState(
            agent_id=self.agent_id,
            belief=self.posterior_mean,
            alpha=self.alpha if self.classes == 2 else np.mean(self.dirichlet_alphas),
            beta_param=self.beta_param if self.classes == 2 else np.mean(self.dirichlet_alphas),
            confidence=1.0 - np.mean(self.posterior_variance)
        )
