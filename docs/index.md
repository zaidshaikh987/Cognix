# COGNIX

### Uncertainty-Aware Multi-Agent AI Decision Framework

COGNIX is a domain-agnostic Python framework for building multi-agent AI systems that combine predictions, uncertainty estimation, graph-based reasoning, belief fusion, calibration, and risk-aware decision making.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)

```text
                Agent A ─────┐
                Agent B ─────┤
                Agent C ─────┼──► Graph Reasoning
                Agent D ─────┘          │
                                        ▼
                              Uncertainty + Fusion
                                        │
                                        ▼
                                  Calibration
                                        │
                                        ▼
                              Risk-Aware Decision
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                           DECIDE             ESCALATE
```

> **"COGNIX turns collections of uncertain agent predictions into calibrated, risk-aware decisions."**

## Why COGNIX?

| Capability | Purpose |
|---|---|
| **Multi-agent reasoning** | Combine predictions from multiple independent agents. |
| **Uncertainty estimation** | Quantify predictive uncertainty via Monte Carlo dropout and Deep Ensembles. |
| **Epistemic reasoning** | Explicitly account for model knowledge limits when agents face novel situations. |
| **Graph reasoning** | Model relationships and dynamic trust between agents using Graph Attention Networks. |
| **Belief fusion** | Combine agent beliefs inversely weighted by their epistemic uncertainty. |
| **Calibration** | Improve probabilistic reliability using Conformal Prediction and Temperature Scaling. |
| **Conformal prediction** | Produce prediction sets with formal coverage guarantees. |
| **Escalation** | Safely route uncertain or high-risk cases to human fallback based on explicit policies. |
| **Explainability** | Analyze individual agent contributions to aggregate uncertainty using Epistemic Shapley values. |

## Core Architecture

The COGNIX pipeline follows a sequential, modular process:

1. **Predictions + Uncertainty**: Agents produce raw predictions along with separated epistemic and aleatoric uncertainty estimates.
2. **Graph Reasoning (NoGraph / GAT)**: A Graph Attention Network models agent relationships, using epistemic uncertainty as a prior to dynamically suppress untrustworthy agents.
3. **Belief Fusion**: The network outputs are fused into a single collective belief state.
4. **Calibration + Conformal**: The fused belief is calibrated, and a prediction set is generated to guarantee a target coverage rate.
5. **Decision / Risk / Escalation**: A risk assessor evaluates the total uncertainty, and an escalation engine determines if the system can safely act or if it must escalate to a human.

## Installation

Install COGNIX via pip:
```bash
pip install cognix
```

Optional dependencies for specific features:
```bash
pip install "cognix[dashboard]"
pip install "cognix[explainability]"
```

For development:
```bash
git clone https://github.com/cognix-framework/cognix.git
cd cognix
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
from cognix import CognixPipeline, EpistemicGAT, EpistemicWeightedFusion, ConformalPredictor
from cognix import StandardAgent, MonteCarloDropout

# 1. Create your agents with Uncertainty Quantification (e.g., MC Dropout)
class CameraAgent(StandardAgent):
    def __init__(self):
        super().__init__("camera_1")
        self.uq_method = MonteCarloDropout(n_samples=20)
        
    def predict(self, x): return 0.85
    def estimate_uncertainty(self, x): 
        # Returns Epistemic and Aleatoric bounds
        return self.uq_method.estimate(self, x)

agents = [CameraAgent()]

# 2. Build the decision pipeline with Graph Reasoning and Calibration
pipeline = CognixPipeline(
    gnn=EpistemicGAT(num_layers=2, input_dim=3, hidden_dim=8, output_dim=4),
    belief_fuser=EpistemicWeightedFusion(),
    calibrator=ConformalPredictor(),
    mode="production"
)

# 3. Run a decision cycle
data = np.random.rand(1, 3)
result = pipeline.run(agents, data, adjacency={})

# 4. Inspect the risk-aware result
print(f"Decision: {result.decision.name}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Total Uncertainty: {result.total_uncertainty:.4f}")
print(f"Risk Level: {result.risk_level.name}")
```

## Think of COGNIX like...

If **NumPy** provides numerical primitives, and **pandas** provides structured data primitives, **COGNIX** provides primitives for uncertainty-aware multi-agent decision workflows. It is designed to act as the standard scaffolding for deploying high-stakes ensemble AI systems.

## Graph Reasoning

COGNIX provides three core graph representation strategies for modeling multi-agent networks:

| Model | Graph reasoning | Epistemic prior | Intended role |
|---|---:|---:|---|
| **NoGraph** | No | No | Non-graph baseline for simple weighted fusion. |
| **StandardGAT** | Yes | No | Standard graph attention modelling structural relationships. |
| **EpistemicGAT** | Yes | Yes | Uncertainty-aware graph attention with monotonic suppression. |

**EpistemicGAT** dynamically suppresses attention to agents experiencing out-of-distribution (OOD) failures. It mathematically injects epistemic uncertainty directly into the attention mechanism using an additive log prior:

`e'_{ij} = e_{ij} + \log(p_{ij})` where `p_{ij} = \frac{1}{1 + \sigma_{e,j}}`

## Uncertainty

COGNIX explicitly distinguishes between:
- **Epistemic Uncertainty**: Model ignorance. Arises from out-of-distribution data (e.g., a camera facing severe glare). Can be reduced with more training data.
- **Aleatoric Uncertainty**: Data noise. Arises from inherent randomness or sensor noise (e.g., low-resolution LiDAR).

COGNIX measures these via **MonteCarloDropout** or **DeepEnsembles**. If Agent A has high epistemic uncertainty, COGNIX knows Agent A is "guessing" and can suppress its influence.

## Belief Fusion

Once agents produce graph-refined predictions, COGNIX fuses them into a single probability distribution. 

- **EpistemicWeightedFusion**: Agents with lower epistemic uncertainty receive larger relative weights in the fusion process.
- **AverageFusion**: A simple unweighted arithmetic mean baseline.
- **BayesianBelief**: Iterative Bayesian updates for continuous belief tracking.

## Calibration & Uncertainty Sets

- **TemperatureScaling**: Post-hoc probability smoothing to align predicted probabilities with empirical accuracy.
- **ConformalPredictor**: Post-hoc generation of prediction sets (e.g., predicting `{Class A, Class B}`) guaranteed to contain the true class with `1 - alpha` probability.

## Risk-Aware Decisions

The **EscalationEngine** evaluates the final calibrated belief, total uncertainty, and conformal set size. 

If the confidence is too low, epistemic uncertainty is too high, or the conformal prediction set size grows beyond a safe threshold (e.g., predicting 3 different possible actions), the pipeline outputs **`ESCALATE`**, signaling the need for a human operator or a fail-safe fallback.

*(Note: Epistemic Shapley values are calculated for explainability and post-hoc attribution, but they are explanation-only and do not trigger escalation).*

## Benchmark Snapshot

Current repository benchmark snapshot: **NORMAL** scenario.

- **Data**: Synthetic multi-modal AV data (`CarlAnomalyDataset`)
- **Agents**: 6 (Camera, LiDAR, Depth, GNSS, IMU, Seg)
- **Runs**: 20 seeds
- **Hardware**: Standard CPU execution
- **Model**: EpistemicGAT

**Metrics:**
- **Accuracy**: 74.3%
- **Expected Calibration Error (ECE)**: 0.104
- **Conformal Coverage**: 95.2% (Target: 90%)
- **Latency (P50)**: 5.8 ms
- **Throughput**: ~152 inferences/sec

*(Note: Current demonstrations use synthetic data. Real-world dataset and simulator integrations are planned extensions. Communication-efficient inference is a future extension, and theoretical communication bandwidths are currently measured in-process).*

## Interactive Analytics Dashboard

Launch the COGNIX visual analytics dashboard to inspect live traces and benchmark results:

```bash
python dashboard/app.py
```
*(Requires the `[dashboard]` extra).*

The dashboard visualizes Epistemic vs. Aleatoric uncertainty separation, latency distributions, and conformal prediction set sizes directly from the benchmark's JSON outputs.

## Project Structure

```text
cognix/
├── agents/        # Agent base classes and registry
├── attribution/   # Epistemic Shapley explainability
├── belief/        # Belief fusion and Bayesian tracking
├── calibration/   # Conformal prediction and Platt scaling
├── config/        # Pydantic configuration schemas
├── decision/      # Escalation and Risk engines
├── engine/        # The core Pipeline and Decision execution logic
├── graph/         # EpistemicGAT and baseline GNN implementations
└── uncertainty/   # MC Dropout, Deep Ensembles, and Decompositions
```

## Design Philosophy

- **Uncertainty first**: Decisions should expose uncertainty rather than hide it.
- **Modular agents**: Agents should remain replaceable and independently configurable.
- **Explicit calibration**: Confidence should be evaluated rather than assumed.
- **Risk-aware decisions**: Uncertain predictions can be routed through explicit decision policies.
- **Domain agnostic**: Core abstractions should not depend on one application domain.
- **Reproducible experiments**: Benchmarking should use controlled seeds and explicit protocols.

## Roadmap

**v0.1 (Current)**
- Core pipeline
- Agents
- Uncertainty Estimation
- Graph reasoning
- Fusion & Calibration
- Conformal prediction
- Escalation Engine
- Benchmarking (Synthetic)

**Future Extensions**
- Communication-efficient inference (Pre-transmission pruning)
- Real-world datasets (CARLA integration)
- Distributed multi-agent execution
- More scalable graph structures

## Testing

Run the test suite using pytest:

```bash
pytest tests/
```

## License

COGNIX is released under the MIT License.
