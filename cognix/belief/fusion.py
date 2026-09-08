import numpy as np
from typing import Optional
from .base import BeliefState, BeliefFuser, FusionStrategy

class CognixBeliefFuser(BeliefFuser):
    """
    Fuses multiple agent beliefs into a single BeliefState.
    """
    def fuse(
        self,
        beliefs: list[BeliefState],
        strategy: FusionStrategy,
        epistemic_uncertainties: Optional[dict[str, float]] = None,
        reliabilities: Optional[dict[str, float]] = None,
        eps: float = 1e-6
    ) -> BeliefState:
        if not beliefs:
            raise ValueError("Empty beliefs list provided for fusion.")

        n_classes = len(beliefs[0].belief)
        agent_ids = [b.agent_id for b in beliefs]
        
        if strategy == FusionStrategy.UNIFORM:
            weights = np.ones(len(beliefs)) / len(beliefs)
        
        elif strategy == FusionStrategy.MAJORITY:
            predictions = [np.argmax(b.belief) for b in beliefs]
            counts = np.bincount(predictions, minlength=n_classes)
            majority_class = np.argmax(counts)
            fused_belief = np.zeros(n_classes)
            fused_belief[majority_class] = 1.0
            return BeliefState(
                agent_id="fused_majority",
                belief=fused_belief,
                alpha=0.0,
                beta_param=0.0,
                confidence=counts[majority_class] / len(beliefs)
            )
            
        elif strategy == FusionStrategy.CONFIDENCE:
            weights = np.array([b.confidence for b in beliefs])
            weights /= (np.sum(weights) + eps)
            
        elif strategy == FusionStrategy.RELIABILITY:
            if not reliabilities:
                weights = np.ones(len(beliefs)) / len(beliefs)
            else:
                weights = np.array([reliabilities.get(b.agent_id, 1.0) for b in beliefs])
                weights /= (np.sum(weights) + eps)
                
        elif strategy == FusionStrategy.EPISTEMIC_WEIGHTED:
            """
            PROPOSED COGNIX MECHANISM: Epistemic Weighted Fusion
            w_j = reliability_j / (sigma_e_j + eps)
            then normalize: w_j = w_j / sum(w_j)
            Note: This is a proposed mechanism. Experimental evaluation required to assess benefits.
            """
            weights = []
            for b in beliefs:
                rel = reliabilities.get(b.agent_id, 1.0) if reliabilities else 1.0
                unc = epistemic_uncertainties.get(b.agent_id, 1.0) if epistemic_uncertainties else 1.0
                weights.append(rel / (unc + eps))
            weights = np.array(weights)
            weights /= (np.sum(weights) + eps)
            
        elif strategy == FusionStrategy.BAYESIAN:
            # Proper Bayesian product-of-experts for Beta distribution
            fused_alpha = sum(b.alpha for b in beliefs)
            fused_beta = sum(b.beta_param for b in beliefs)
            fused_belief_val = fused_alpha / (fused_alpha + fused_beta)
            fused_belief = np.array([1 - fused_belief_val, fused_belief_val]) if n_classes == 2 else np.ones(n_classes) / n_classes # Simplification for multiclass
            return BeliefState(
                agent_id="fused_bayesian",
                belief=fused_belief,
                alpha=fused_alpha,
                beta_param=fused_beta,
                confidence=1.0 - (fused_alpha * fused_beta) / ((fused_alpha + fused_beta)**2 * (fused_alpha + fused_beta + 1))
            )
            
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

        # Weighted average
        fused_prob = np.zeros(n_classes)
        fused_confidence = 0.0
        
        for i, b in enumerate(beliefs):
            fused_prob += weights[i] * b.belief
            fused_confidence += weights[i] * b.confidence
            
        return BeliefState(
            agent_id="fused_" + strategy.name.lower(),
            belief=fused_prob,
            alpha=1.0,
            beta_param=1.0,
            confidence=fused_confidence
        )
