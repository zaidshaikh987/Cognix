# 🧠 COGNIX

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)]()

> A domain-agnostic, uncertainty-aware multi-agent AI decision framework.

COGNIX is a framework designed to tackle the most critical problem in Multi-Agent AI systems: **Trust under Distributional Shift**. When multiple AI agents (sensors, models, or LLMs) disagree, COGNIX uses mathematical Uncertainty Quantification (Epistemic vs. Aleatoric) to dynamically down-weight degraded agents and fuse a safe, calibrated decision.

## 🌟 Key Features

- **Domain Agnostic**: Built for Autonomous Vehicles, Medical Diagnostics, Robotics, and Finance. If it has logits, COGNIX can fuse it.
- **Epistemic-Weighted Fusion**: Decouples model uncertainty from data uncertainty using Monte Carlo Dropout and Deep Ensembles.
- **Dynamic Escalation**: Automatically triggers `REQUEST_INFORMATION` or `ESCALATE` when total uncertainty crosses safety thresholds.
- **V2V / Network Graph Ready**: Built-in support for Inter-Agent Graph Refinement (querying neighboring nodes when local sensors fail).
- **Real-Time Dashboard**: Ships with a FastAPI + WebSocket dashboard for streaming live Expected Calibration Error (ECE) and Shapley Attribution.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zaidshaikh987/Cognix.git
cd Cognix

# Install the framework
pip install -e .
```

### Basic Usage

```python
from cognix import DecisionEngine
from cognix.config.schema import CognixConfig

# Initialize the engine
config = CognixConfig()
engine = DecisionEngine(config=config, belief="epistemic_weighted")

# Agents can be PyTorch models, LLMs, or any predictive function
agents = [radar_agent, camera_agent, lidar_agent]

# The engine handles Uncertainty Quantification and Fusion internally
result = engine.decide(agents, current_state_data)

if result.decision == "ACT":
    print(f"Proceeding with confidence: {result.confidence}")
elif result.decision == "ESCALATE":
    print("Emergency handover triggered due to high Epistemic Uncertainty.")
```

## 📊 The Dashboard

COGNIX comes with a built-in research dashboard to visualize your agent swarms in real-time.

```bash
# Run a synthetic experiment
python examples/rq_synthetic_001.py

# Launch the dashboard
python dashboard/app.py
```
Navigate to `http://localhost:8000` to view the live telemetry.

## 🔬 Scientific Validation

COGNIX is rigorously tested against Out-Of-Distribution (OOD) scenarios. In our benchmark synthetic test (`RQ_SYNTHETIC_001`), Epistemic-Weighted Fusion reduced Expected Calibration Error (ECE) from **16.12%** (Uniform Fusion) down to **4.36%** when targeted agents were subjected to severe covariate shift.

See the [Research Documentation](docs/research/RQ_SYNTHETIC_001_REPORT.md) for full statistical analyses.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details on how to get started.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
