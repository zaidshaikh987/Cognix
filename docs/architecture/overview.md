# COGNIX Architecture Overview

The COGNIX architecture is built around a unidirectional data flow from raw agent predictions to a final, safe system decision.

## Pipeline Architecture

```ascii
[Agent 1: Camera]       [Agent 2: Lidar]      [Agent N...]
       │                       │                   │
       ▼                       ▼                   ▼
┌──────────────┐        ┌──────────────┐     ┌──────────────┐
│ Uncertainty  │        │ Uncertainty  │     │ Uncertainty  │
│ Estimator    │        │ Estimator    │     │ Estimator    │
└──────┬───────┘        └──────┬───────┘     └──────┬───────┘
       │ (Predict, Epistemic,  │                    │
       │  Aleatoric)           │                    │
       ▼                       ▼                    ▼
       └───────────────┬───────┘────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   Belief Fusion     │ ◄── Epistemic-Weighted Strategy
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │    Calibration      │ ◄── Ensure P(y|x) matches accuracy
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Decision Engine    │ ◄── Outputs (ACT, WAIT, ABSTAIN)
            └─────────────────────┘
```

## Core Principles
1. **Uncertainty as a First-Class Citizen**: No prediction is passed without its uncertainty profile.
2. **Epistemic vs. Aleatoric Isolation**: We explicitly isolate model ignorance (epistemic) from data noise (aleatoric).
3. **Fail-Safe Defaults**: If total uncertainty exceeds the safety threshold, the system defaults to ABSTAIN or ESCALATE.
4. **Transparent Attribution**: The `explainability` module can always trace a bad decision back to the specific agent and uncertainty type that caused it.
