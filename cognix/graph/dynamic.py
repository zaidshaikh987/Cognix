"""
Dynamic Graph Construction for COGNIX graph modules.
"""
import numpy as np

class DynamicGraphBuilder:
    """
    Constructs adjacency matrices dynamically based on agent feature similarity.
    """
    def __init__(self, k_neighbors: int = 3, threshold: float = 0.5):
        self.k_neighbors = k_neighbors
        self.threshold = threshold

    def build_knn_graph(self, features: np.ndarray) -> np.ndarray:
        """
        Builds a K-Nearest Neighbors adjacency matrix using Cosine Similarity.
        Args:
            features: (N, D) array of agent features.
        Returns:
            adjacency_matrix: (N, N) binary array.
        """
        N = features.shape[0]
        if N <= 1:
            return np.ones((N, N))
            
        # Normalize features for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_features = features / norms
        
        # Cosine similarity matrix
        sim_matrix = norm_features @ norm_features.T
        
        adjacency = np.zeros((N, N))
        for i in range(N):
            # Sort indices descending, excluding self
            indices = np.argsort(sim_matrix[i])[::-1]
            indices = [idx for idx in indices if idx != i]
            
            # Connect to top K
            top_k = indices[:self.k_neighbors]
            for j in top_k:
                if sim_matrix[i, j] >= self.threshold:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0 # Ensure undirected for standard GCN
                    
            # Self-loop
            adjacency[i, i] = 1.0
            
        return adjacency
