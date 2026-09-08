import pytest
import numpy as np
from cognix.engine.decision_engine import DecisionEngine
from cognix.core.interfaces import GraphRefinement
from cognix.core.types import GraphResult

class NoGraph(GraphRefinement):
    """Null-object pattern for graph refinement. Passes data through unchanged."""
    def forward(self, node_features, adjacency, epistemic_uncertainties, agent_order):
        return GraphResult(
            node_outputs=node_features,
            attention=None,
            metadata={"type": "no_graph"}
        )

class StandardGATMock(GraphRefinement):
    """Mock standard GAT that doesn't use epistemic uncertainties."""
    def forward(self, node_features, adjacency, epistemic_uncertainties, agent_order):
        # Simply scales the features
        import torch
        outputs = node_features * 1.5
        attention = [torch.eye(node_features.shape[0])]
        return GraphResult(
            node_outputs=outputs,
            attention=attention,
            metadata={"type": "standard_gat"}
        )

class EpistemicGATMock(GraphRefinement):
    """Mock epistemic GAT that utilizes epistemic uncertainty as a prior."""
    def forward(self, node_features, adjacency, epistemic_uncertainties, agent_order):
        import torch
        # Scale features and apply a fake epistemic prior weight
        prior_weights = torch.tensor([epistemic_uncertainties.get(a, 0.1) for a in agent_order]).unsqueeze(1)
        outputs = node_features * prior_weights
        attention = [torch.ones((node_features.shape[0], node_features.shape[0])) / node_features.shape[0]]
        return GraphResult(
            node_outputs=outputs,
            attention=attention,
            metadata={"type": "epistemic_gat"}
        )

def test_graph_substitutability():
    """
    Test behavioral substitutability: changing the GraphRefinement plugin 
    must not require modifying the DecisionEngine, and the engine continues execution.
    """
    import torch
    
    # Mock inputs
    node_features = torch.ones((3, 4))
    adjacency = torch.ones((3, 3))
    epistemic_uncertainties = {"agent_1": 0.2, "agent_2": 0.5, "agent_3": 0.8}
    agent_order = ["agent_1", "agent_2", "agent_3"]
    
    graphs = [
        NoGraph(),
        StandardGATMock(),
        EpistemicGATMock()
    ]
    
    for graph in graphs:
        # Dependency Injection of the GraphRefinement strategy
        engine = DecisionEngine(graph=graph)
        
        # The engine must accept the injected plugin and produce valid GraphResults internally
        # Here we manually verify the graph plugin contract directly
        result = engine.graph.forward(node_features, adjacency, epistemic_uncertainties, agent_order)
        
        assert isinstance(result, GraphResult)
        assert result.node_outputs is not None
        assert "type" in result.metadata
