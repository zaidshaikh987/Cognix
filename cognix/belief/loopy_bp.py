"""
Loopy Belief Propagation (LBP) for agent graphs.
"""
import numpy as np
from typing import Dict, List, Tuple
from cognix.belief.base import BeliefState

class LoopyBeliefPropagation:
    """
    Implements Loopy Belief Propagation for a graph of agents.
    Reference: Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems.
    """
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 1e-4):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence D_KL(P || Q)"""
        eps = 1e-10
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    def run(self, beliefs: Dict[str, BeliefState], adjacency: Dict[str, List[str]], compatibilities: Dict[Tuple[str, str], np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Run LBP over the agent graph.
        
        Args:
            beliefs: Dictionary mapping agent_id to initial BeliefState.
            adjacency: Dictionary mapping agent_id to a list of neighbor agent_ids.
            compatibilities: Dictionary mapping (agent_i, agent_j) to a compatibility matrix (n_classes, n_classes).
                             If None, assumes identity matrix (neighbor should have same class).
                             
        Returns:
            Tuple of:
            - final_beliefs: Dictionary mapping agent_id to final probability distribution (np.ndarray).
            - kl_history: Dictionary mapping agent_id to the KL divergence at the final step.
        """
        if not beliefs:
            return {}, {}
            
        agent_ids = list(beliefs.keys())
        # Infer n_classes from the first belief state
        n_classes = len(beliefs[agent_ids[0]].belief)
        
        # Initialize messages: msg[(i, j)] is message from i to j
        messages: Dict[Tuple[str, str], np.ndarray] = {}
        for i in agent_ids:
            for j in adjacency.get(i, []):
                messages[(i, j)] = np.ones(n_classes) / n_classes

        # Initialize node potentials (initial beliefs)
        node_potentials = {aid: np.copy(b.belief) for aid, b in beliefs.items()}
        
        # Default compatibility matrix: Identity (agents tend to agree)
        if compatibilities is None:
            compatibilities = {}
            for i in agent_ids:
                for j in adjacency.get(i, []):
                    # Slight smoothing to avoid zero probabilities
                    mat = np.eye(n_classes) * 0.9 + np.ones((n_classes, n_classes)) * (0.1 / n_classes)
                    compatibilities[(i, j)] = mat
                    
        current_beliefs = {aid: np.copy(b.belief) for aid, b in beliefs.items()}
        kl_history = {aid: 0.0 for aid in agent_ids}
        
        damping = 0.5  # Damped Message Passing to prevent oscillation
        
        for iteration in range(self.max_iterations):
            new_messages = {}
            max_kl = 0.0
            
            # 1. Compute new messages
            for i in agent_ids:
                for j in adjacency.get(i, []):
                    # Product of incoming messages to i, excluding j
                    incoming_prod = np.ones(n_classes)
                    for k in adjacency.get(i, []):
                        if k != j:
                            incoming_prod *= messages[(k, i)]
                            
                    belief_i = node_potentials[i] * incoming_prod
                    belief_i /= (np.sum(belief_i) + 1e-10)
                    
                    comp_matrix = compatibilities[(i, j)]
                    msg_ij = comp_matrix.T @ belief_i
                    msg_ij /= (np.sum(msg_ij) + 1e-10)
                    
                    # Apply damping
                    new_messages[(i, j)] = damping * messages[(i, j)] + (1 - damping) * msg_ij
                    
            messages = new_messages
            
            # 2. Update beliefs and check convergence
            new_beliefs = {}
            for i in agent_ids:
                incoming_prod = np.ones(n_classes)
                for k in adjacency.get(i, []):
                    incoming_prod *= messages[(k, i)]
                    
                belief_i = node_potentials[i] * incoming_prod
                belief_i /= (np.sum(belief_i) + 1e-10)
                new_beliefs[i] = belief_i
                
                kl = self._kl_divergence(belief_i, current_beliefs[i])
                kl_history[i] = kl
                max_kl = max(max_kl, kl)
                
            current_beliefs = new_beliefs
            
            if max_kl < self.convergence_threshold:
                break
                
        # Compute Bethe Free Energy (Simplified Approximation)
        # F_Bethe = U - H (Energy - Entropy)
        bethe_free_energy = 0.0
        for i in agent_ids:
            entropy = -np.sum(current_beliefs[i] * np.log(current_beliefs[i] + 1e-10))
            bethe_free_energy -= entropy
            
        return current_beliefs, kl_history
