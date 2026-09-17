# Getting Started

This guide will walk you through setting up COGNIX and running your first multi-agent decision pipeline.

## Installation

COGNIX is distributed as a standard Python package. It requires Python 3.10+.

```bash
pip install cognix
```

Optional features (such as the analytics dashboard and explainability adapters) can be installed using extras:

```bash
pip install "cognix[dashboard]"
pip install "cognix[explainability]"
```

## Your First COGNIX Pipeline

The central component of COGNIX is the `CognixPipeline`, which takes predictions from multiple agents, applies graph reasoning, fuses their beliefs, calibrates them, and outputs a final `DecisionResult`.

Below is a complete, runnable minimal example.

### 1. Import Dependencies

```python
import numpy as np
from cognix import (
    CognixPipeline,
    StandardAgent,
    MonteCarloDropout,
    EpistemicGAT,
    EpistemicWeightedFusion,
    ConformalPredictor
)
```

### 2. Define an Agent

Agents in COGNIX must implement the `StandardAgent` interface. We define a simple dummy agent that produces a prediction and uses Monte Carlo Dropout for uncertainty estimation.

```python
class DemoCameraAgent(StandardAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        # Use Monte Carlo Dropout for uncertainty quantification
        self.uq_method = MonteCarloDropout(n_samples=20)
        
    def predict(self, x: np.ndarray) -> float:
        # A mock classification probability
        return 0.85
        
    def estimate_uncertainty(self, x: np.ndarray):
        # Returns an object with `.epistemic`, `.aleatoric`, and `.total`
        return self.uq_method.estimate(self, x)

agents = [DemoCameraAgent("camera_front"), DemoCameraAgent("camera_rear")]
```

### 3. Build the Pipeline

Construct the `CognixPipeline` by providing the core reasoning components. We use `EpistemicGAT` for graph reasoning, `EpistemicWeightedFusion` for belief fusion, and `ConformalPredictor` for calibration.

```python
pipeline = CognixPipeline(
    gnn=EpistemicGAT(num_layers=2, input_dim=3, hidden_dim=8, output_dim=4),
    belief_fuser=EpistemicWeightedFusion(),
    calibrator=ConformalPredictor(),
    mode="production"  # In production, missing agent inputs are gracefully handled
)
```

### 4. Run the Pipeline

Pass your agents, input data, and adjacency matrix (if your agents have specific physical or logical connections) to the pipeline.

```python
# Mock sensor data for the agents
data = np.random.rand(1, 3)

# Execute the decision workflow
result = pipeline.run(agents, data, adjacency={})
```

### 5. Inspect the Decision Result

The `DecisionResult` object contains everything the system concluded, including confidence, total uncertainty, risk level, and whether human escalation was required.

```python
print(f"Decision: {result.decision.name}")
print(f"Final Confidence: {result.confidence:.3f}")
print(f"Total Uncertainty: {result.total_uncertainty:.3f}")
print(f"Risk Level: {result.risk_level.name}")
print(f"Escalation Triggered: {result.escalation_required}")
print(f"Explainability / Shapley Values: {result.agent_contributions}")
```

## Next Steps

- Learn about the core architecture in [Concepts & Architecture](concepts.md).
- Dive deep into [Uncertainty Estimation](uncertainty.md).
- Understand how [EpistemicGAT](graph-reasoning.md) routes trust dynamically.
