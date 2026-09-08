"""
Epistemic-Aware Graph Attention Network (GAT).

PROPOSED COGNIX MECHANISM (requires experimental validation, not established prior art):
The proposed modification initializes attention weights based on epistemic uncertainty:
  w_initial(j -> i) = 1 / (1 + sigma_e_j)
where sigma_e_j is the epistemic uncertainty of the sending agent j.

These initial weights are used to bias (not replace) the standard GAT attention learning:
  e_ij_epistemic = e_ij_standard * w_initial(j -> i)
This acts as an uncertainty-based attention prior.
"""

import numpy as np
from cognix.graph.gat import GATLayer

class EpistemicGATLayer(GATLayer):
    def forward_epistemic(self, H: np.ndarray, A: np.ndarray, epistemic_prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = H.shape[0]
        Wh = H @ self.W
        
        a_input = np.zeros((N, N, 2 * self.out_features))
        for i in range(N):
            for j in range(N):
                a_input[i, j, :] = np.concatenate([Wh[i], Wh[j]])
                
        e = (a_input @ self.a).squeeze(-1)
        e = np.where(e > 0, e, self.alpha * e)
        
        # PROPOSED COGNIX MECHANISM: Bias standard GAT attention with epistemic prior
        e_epistemic = e * epistemic_prior
        
        mask = np.where(A > 0, 1, -np.inf)
        e_epistemic = e_epistemic + mask
        
        e_max = np.max(e_epistemic, axis=1, keepdims=True)
        exp_e = np.exp(e_epistemic - e_max)
        attention_weights = exp_e / np.sum(exp_e, axis=1, keepdims=True)
        
        H_prime = np.maximum(0, attention_weights @ Wh)
        return H_prime, attention_weights

class EpistemicGAT:
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        self.layers = []
        if num_layers == 1:
            self.layers.append(EpistemicGATLayer(input_dim, output_dim))
        else:
            self.layers.append(EpistemicGATLayer(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(EpistemicGATLayer(hidden_dim, hidden_dim))
            self.layers.append(EpistemicGATLayer(hidden_dim, output_dim))

    def compute_epistemic_weights(self, uncertainties: dict[str, float], agent_order: list[str]) -> np.ndarray:
        """
        Compute epistemic prior weights matrix.
        w_initial(j -> i) = 1 / (1 + sigma_e_j)
        """
        N = len(agent_order)
        W_prior = np.zeros((N, N))
        for j_idx, j_agent in enumerate(agent_order):
            sigma_e_j = uncertainties.get(j_agent, 0.0)
            weight = 1.0 / (1.0 + sigma_e_j)
            W_prior[:, j_idx] = weight
        return W_prior

    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray, epistemic_uncertainties: dict[str, float], agent_order: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Forward pass with epistemic uncertainty.
        """
        epistemic_prior = self.compute_epistemic_weights(epistemic_uncertainties, agent_order)
        H = node_features
        attention_weights_list = []
        
        for layer in self.layers:
            H, attn = layer.forward_epistemic(H, adjacency_matrix, epistemic_prior)
            attention_weights_list.append(attn)
            
        return H, attention_weights_list

    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        KL-divergence tracking between attention distributions across iterations.
        """
        epsilon = 1e-10
        p_safe = np.clip(p, epsilon, 1.0)
        q_safe = np.clip(q, epsilon, 1.0)
        return np.sum(p_safe * np.log(p_safe / q_safe))
