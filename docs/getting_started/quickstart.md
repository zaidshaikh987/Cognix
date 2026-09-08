# Quickstart Guide

Get up and running with COGNIX in under 5 minutes.

## 1. Installation

```bash
pip install cognix
```

## 2. Your First Decision Cycle

```python
from cognix.core import Belief
from cognix.fusion import EpistemicWeightedFusion
from cognix.decision import RuleBasedDecisionMaker, DecisionEngine

# 1. Create dummy beliefs from two sensors
cam_belief = Belief(
    prediction={"pedestrian": 0.8, "empty": 0.2},
    epistemic_uncertainty=0.1,  # Highly certain
    aleatoric_uncertainty=0.1,
    agent_id="Camera"
)

lidar_belief = Belief(
    prediction={"pedestrian": 0.1, "empty": 0.9},
    epistemic_uncertainty=0.8,  # Highly uncertain (maybe rain/fog)
    aleatoric_uncertainty=0.2,
    agent_id="Lidar"
)

# 2. Setup the engine
fusion = EpistemicWeightedFusion()
decision = RuleBasedDecisionMaker(abstain_threshold=0.3)
engine = DecisionEngine(fusion, decision)

# 3. Execute
result = engine.decide([cam_belief, lidar_belief])

print(f"Action: {result.action}")
print(f"Confidence: {result.confidence:.2f}")
# Because Lidar had high epistemic uncertainty, the fusion heavily weights the Camera.
```

## 3. Next Steps
- Explore `examples/av_scenario.py`
- Launch the `dashboard/app.py`
