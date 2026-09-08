import numpy as np
from cognix.core.interfaces import GraphRefinement
from cognix.core.types import GraphResult

class NoGraph(GraphRefinement):
    """
    An identity GraphRefinement plugin.
    
    Returns the initial node predictions (feature 0) unchanged.
    Used for ablations and architectures that do not require spatial/graph reasoning.
    """
    
    def forward(
        self,
        node_features: np.ndarray,
        adjacency_matrix: np.ndarray,
        epistemic_uncertainties: dict[str, float],
        agent_order: list[str]
    ) -> GraphResult:
        
        # Identity mapping: node_features[:, 0] is the base prediction
        N = node_features.shape[0]
        # Return in shape (N, 1) so it mimics H_prime[:, 0]
        H_prime = node_features[:, 0].reshape(N, 1)
        
        # Attention is identity
        attn = np.eye(N, dtype=np.float32)
        
        return GraphResult(
            node_outputs=H_prime,
            attention=[attn],
            metadata={"type": "NoGraph"}
        )
