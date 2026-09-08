"""
Epistemic Shapley Attribution.

PROPOSED COGNIX MECHANISM (requires experimental validation):
  phi_i = sigma_e(coalition WITH agent i) - sigma_e(coalition WITHOUT agent i)

This is analogous to standard Shapley values but applied to epistemic uncertainty reduction
instead of model output. It approximates the marginal contribution of each agent to
collective epistemic uncertainty.

Note: Standard Shapley values (Shapley, 1953) are an established game-theoretic concept.
The specific application to epistemic uncertainty in multi-agent systems is a proposed mechanism.
"""

import numpy as np
import random
from typing import Callable, Any

class EpistemicShapley:
    def compute(self, agent_ids: list[str], uncertainty_fn: Callable[[list[str]], float], num_samples: int = 100) -> dict[str, float]:
        """
        Use Monte Carlo approximation of Shapley values.
        """
        shapley_values = {agent: 0.0 for agent in agent_ids}
        n = len(agent_ids)
        
        if n == 0:
            return shapley_values
            
        for _ in range(num_samples):
            permutation = list(agent_ids)
            random.shuffle(permutation)
            
            coalition = []
            prev_uncertainty = uncertainty_fn(coalition)
            
            for agent in permutation:
                coalition.append(agent)
                curr_uncertainty = uncertainty_fn(coalition)
                # Contribution is reduction in uncertainty (or change)
                # phi_i = sigma_e(coalition WITH agent i) - sigma_e(coalition WITHOUT agent i)
                marginal_contribution = curr_uncertainty - prev_uncertainty
                shapley_values[agent] += marginal_contribution
                prev_uncertainty = curr_uncertainty
                
        # Average over samples
        for agent in agent_ids:
            shapley_values[agent] /= num_samples
            
        return shapley_values

    def interpret(self, shapley_values: dict[str, float]) -> dict[str, Any]:
        if not shapley_values:
            return {}
            
        ranking = sorted(shapley_values.items(), key=lambda x: x[1])
        top_contributor = ranking[0][0] if ranking else None
        
        summary = f"Top contributor to uncertainty reduction: {top_contributor}"
        
        return {
            'top_contributor': top_contributor,
            'ranking': ranking,
            'summary': summary
        }
