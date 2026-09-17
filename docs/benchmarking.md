# Experimental Workflow

Evaluating uncertainty-aware multi-agent systems requires rigorous stress testing beyond standard clean-data benchmarks. COGNIX includes an experimental workflow designed to measure both predictive performance and probabilistic quality across adverse conditions.

## Benchmark Scenarios

The COGNIX evaluation suite tests the architecture under several scenarios (currently using synthetic data):

- **NORMAL**: Baseline execution with clean data.
- **HIGH_NOISE**: Severe aleatoric noise injected into all agent inputs (simulating extreme weather).
- **MISSING_AGENT**: Sudden catastrophic failure (blackout) of a subset of agents (e.g., Camera fails).
- **OOD_SHIFT**: Gradual drift into out-of-distribution spaces (e.g., GPS drift).
- **CONFLICTING**: Adversarial or malfunctioning agents broadcasting highly confident but incorrect predictions.
- **MULTI_FAILURE**: Cascading failures across multiple agents simultaneously.

## Metrics

COGNIX outputs a comprehensive `all_summaries.json` capturing:

### Predictive Performance
- **Accuracy** & **Balanced Accuracy**
- **F1 Score**, Precision, Recall
- **AUROC**

### Probabilistic Quality
- **Negative Log Likelihood (NLL)**
- **Brier Score**
- **Expected Calibration Error (ECE)** & **Maximum Calibration Error (MCE)**

### Uncertainty & Conformal
- **Mean Epistemic Uncertainty**
- **Epistemic Separation** (Ability to distinguish OOD from In-Distribution)
- **Conformal Coverage** (Target vs Empirical)
- **Conformal Set Size**

### Efficiency
- **Latency (P50, P95, P99)**
- **Throughput (Inferences / Sec)**

## Current Benchmark Snapshot

*Data source: `results/universal_evaluation/summaries/all_summaries.json`*

**Current repository benchmark snapshot: NORMAL scenario.**
- **Hardware**: Standard CPU execution
- **Runs**: 20 controlled random seeds

| Metric | EpistemicGAT (Mean) |
|---|---|
| Accuracy | 74.3% |
| Balanced Accuracy | 74.8% |
| F1 Score | 0.753 |
| AUROC | 0.826 |
| Brier Score | 0.179 |
| ECE | 0.104 |
| Conformal Coverage | 95.2% |
| Conformal Set Size | 1.61 |
| Latency P50 | 8.35 ms |
| Latency P99 | 19.31 ms |
| Throughput | 112.9 inf/sec |

*Note: The current snapshot utilizes the `CarlAnomalyDataset` operating in synthetic mode. Real-world dataset integrations are planned extensions.*
