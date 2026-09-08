"""
Standard Graph Attention Network (GAT) implementation using NumPy.
Reference: Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. ICLR 2018.

Note: This is a BASELINE method.
"""

import numpy as np

class GATLayer:
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        limit_W = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit_W, limit_W, (in_features, out_features))
        
        limit_a = np.sqrt(6 / (2 * out_features + 1))
        self.a = np.random.uniform(-limit_a, limit_a, (2 * out_features, 1))

    def forward(self, H: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = H.shape[0]
        # Linear transformation
        Wh = H @ self.W # (N, out_features)
        
        # Calculate attention scores
        a_input = np.zeros((N, N, 2 * self.out_features))
        for i in range(N):
            for j in range(N):
                a_input[i, j, :] = np.concatenate([Wh[i], Wh[j]])
                
        e = (a_input @ self.a).squeeze(-1) # (N, N)
        
        # LeakyReLU
        e = np.where(e > 0, e, self.alpha * e)
        
        # Masked attention
        mask = np.where(A > 0, 1, -np.inf)
        e = e + mask
        
        # Softmax over j
        e_max = np.max(e, axis=1, keepdims=True)
        exp_e = np.exp(e - e_max)
        attention_weights = exp_e / np.sum(exp_e, axis=1, keepdims=True)
        
        # H' = sigma(sum(alpha * W * H))
        H_prime = np.maximum(0, attention_weights @ Wh) # ReLU
        
        return H_prime, attention_weights

class GATNetwork:
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        self.layers = []
        if num_layers == 1:
            self.layers.append(GATLayer(input_dim, output_dim))
        else:
            self.layers.append(GATLayer(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GATLayer(hidden_dim, hidden_dim))
            self.layers.append(GATLayer(hidden_dim, output_dim))

    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        H = node_features
        attention_weights_list = []
        for layer in self.layers:
            H, attn = layer.forward(H, adjacency_matrix)
            attention_weights_list.append(attn)
        return H, attention_weights_list
