import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class UncertaintyEstimate:
    """
    Represents an uncertainty estimate for a prediction.
    Separates aleatoric (data) uncertainty from epistemic (model) uncertainty.
    """
    aleatoric: float
    epistemic: float
    total: float
    raw_samples: Optional[np.ndarray] = None
    method: str = "unknown"

# Type alias for a collection of uncertainty estimates across multiple agents
UncertaintyBundle = Dict[str, UncertaintyEstimate]

class UncertaintyEstimator(abc.ABC):
    """
    Abstract base class for uncertainty estimation methods in the COGNIX framework.
    """
    
    @abc.abstractmethod
    def estimate(self, model: Any, inputs: Any) -> UncertaintyEstimate:
        """
        Estimate uncertainty for the given model and inputs.
        
        Args:
            model: The agent or underlying model.
            inputs: The inputs to run inference on.
            
        Returns:
            UncertaintyEstimate: The computed uncertainty metrics.
        """
        pass
