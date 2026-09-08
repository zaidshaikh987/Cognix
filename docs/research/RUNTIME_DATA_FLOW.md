# COGNIX Runtime Data Flow

To ensure strict scientific integrity, COGNIX enforces a unidirectional data flow where no metrics are artificially generated. Every number displayed in the dashboard or reported in a paper is traceably computed from the step before it.

```mermaid
graph TD
    %% Ground Truth & Input
    GT[Ground Truth State] -->|Simulation/Dataset| IN[Sensor Input Data]
    IN --> A1[Agent 1: MC Dropout]
    IN --> A2[Agent 2: MC Dropout]
    IN --> A3[Agent 3: MC Dropout]
    
    %% Agent Processing
    A1 -->|Stochastic Forward Passes| P1[Point Prediction 1]
    A1 -->|Variance Calculation| U1[Epistemic UQ 1]
    
    A2 -->|Stochastic Forward Passes| P2[Point Prediction 2]
    A2 -->|Variance Calculation| U2[Epistemic UQ 2]
    
    A3 -->|Stochastic Forward Passes| P3[Point Prediction 3]
    A3 -->|Variance Calculation| U3[Epistemic UQ 3]
    
    %% Engine Pipeline
    P1 --> FUS[Epistemic-Weighted Fusion]
    P2 --> FUS
    P3 --> FUS
    
    U1 --> T1[Trust Weight 1]
    U2 --> T2[Trust Weight 2]
    U3 --> T3[Trust Weight 3]
    
    T1 --> FUS
    T2 --> FUS
    T3 --> FUS
    
    %% Output
    FUS -->|Combined Belief| DEC[Decision Engine]
    U1 -->|Uncertainty Union| DEC
    U2 --> DEC
    U3 --> DEC
    
    DEC --> OUT[DecisionTrace & Metadata]
    OUT -->|Export to results/| DASH[Research Dashboard]
```

## Guarantees
1. **No Synthetic Overrides**: Uncertainty is derived exclusively from the mathematical variance of the underlying PyTorch models (MC Dropout) or Deep Ensembles.
2. **Immutable Provenance**: The resulting `DecisionTrace` records the exact seed, run ID, and configuration parameters that produced it.
3. **Trace Playback**: The dashboard operates exclusively by reading generated `DecisionTrace` JSON logs. It no longer contains internal "demo mode" signal generators.
