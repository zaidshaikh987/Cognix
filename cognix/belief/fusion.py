import numpy as np
from typing import Dict
from cognix.core.interfaces import BeliefFuser
from cognix.core.types import FusionResult

class EpistemicWeightedFusion(BeliefFuser):
    """
    PROPOSED COGNIX MECHANISM: Epistemic Weighted Fusion
    w_j = reliability_j / (sigma_e_j + eps)
    then normalize: w_j = w_j / sum(w_j)
    """
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        
    def fuse(
        self, 
        predictions: Dict[str, float], 
        uncertainties: Dict[str, float], 
        reliabilities: Dict[str, float]
    ) -> FusionResult:
        
        if not predictions:
            raise ValueError("Empty predictions dictionary provided for fusion.")
            
        agents = list(predictions.keys())
        weights_list = []
        
        for agent_id in agents:
            rel = reliabilities.get(agent_id, 1.0) if reliabilities else 1.0
            unc = uncertainties.get(agent_id, 1.0) if uncertainties else 1.0
            weights_list.append(rel / (unc + self.eps))
            
        weights_arr = np.array(weights_list)
        weights_arr /= (np.sum(weights_arr) + self.eps)
        
        fused_prob = 0.0
        weight_dict = {}
        for i, agent_id in enumerate(agents):
            fused_prob += weights_arr[i] * predictions[agent_id]
            weight_dict[agent_id] = float(weights_arr[i])
            
        return FusionResult(
            probability=float(fused_prob),
            weights=weight_dict
        )

class AverageFusion(BeliefFuser):
    """
    Standard unweighted average fusion baseline.
    """
    def fuse(
        self, 
        predictions: Dict[str, float], 
        uncertainties: Dict[str, float], 
        reliabilities: Dict[str, float]
    ) -> FusionResult:
        
        if not predictions:
            raise ValueError("Empty predictions dictionary provided for fusion.")
            
        agents = list(predictions.keys())
        w = 1.0 / len(agents)
        
        fused_prob = sum(predictions.values()) * w
        weight_dict = {a: w for a in agents}
        
        return FusionResult(
            probability=float(fused_prob),
            weights=weight_dict
        )
