# API Reference

This page documents the public API exposed at the top level of the `cognix` package.

## Core Engine

### CognixPipeline

**Purpose**: The central orchestrator that strings together graph reasoning, fusion, calibration, and escalation.

**Import**:
```python
from cognix import CognixPipeline
```

**Constructor**:
```python
CognixPipeline(
    gnn=None,
    belief_fuser=None,
    calibrator=None,
    attribution=None,
    escalation=None,
    mode="production"
)
```

### DecisionEngine

**Purpose**: Lower-level execution engine that handles the individual execution steps of the pipeline. Typically invoked internally by `CognixPipeline`.

### DecisionResult

**Purpose**: The return object of the pipeline.

**Fields**:
- `decision`: `DecisionOutcome` (ACT, ESCALATE, ABSTAIN).
- `confidence`: `float`.
- `calibrated_confidence`: `float` or `None`.
- `total_uncertainty`: `float`.
- `epistemic_uncertainty`: `float`.
- `aleatoric_uncertainty`: `float`.
- `risk_level`: `RiskLevel` enum.
- `escalation_required`: `bool`.
- `agent_contributions`: `dict` of Shapley values.

## Graph Models

### EpistemicGAT

**Purpose**: Graph Attention Network that dynamically suppresses agents experiencing high epistemic uncertainty.

**Import**:
```python
from cognix import EpistemicGAT
```

**Constructor**:
```python
EpistemicGAT(
    num_layers=2,
    input_dim=3,
    hidden_dim=8,
    output_dim=4,
    use_epistemic_prior=True
)
```

## Belief Fusion

### EpistemicWeightedFusion

**Purpose**: Fuses agent predictions using weights inversely proportional to their epistemic uncertainty.

**Import**:
```python
from cognix import EpistemicWeightedFusion
```

## Calibration

### ConformalPredictor

**Purpose**: Applies conformal prediction to generate calibrated prediction sets.

**Import**:
```python
from cognix import ConformalPredictor
```

## Decision & Risk

### EscalationEngine

**Purpose**: Evaluates confidence, uncertainty, and conformal sets against strict thresholds to trigger human fallbacks.

**Import**:
```python
from cognix import EscalationEngine
```
