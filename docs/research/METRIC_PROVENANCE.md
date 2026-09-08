# COGNIX Metric Provenance

To satisfy rigorous research requirements, COGNIX guarantees that all reported metrics correspond to a formal mathematical definition evaluated against a tracked dataset.

## Core Metrics

### 1. Expected Calibration Error (ECE)
**Source Function**: `calculate_ece(predictions, truths, bins)` in `examples/rq_synthetic_001.py`.
**Formula**: 
$$ \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)| $$
**Description**: Measures the absolute difference between model confidence and actual accuracy across $M$ probability bins.
**Provenance Validation**: ECE is only calculated *after* the complete dataset of `DecisionTrace` outputs is generated. The dashboard displays the mathematically computed ECE from the `DecisionTrace.metrics` object.

### 2. Expected Accuracy
**Source Function**: `np.mean(np.round(preds) == truths)`
**Description**: The proportion of correct decisions vs the ground truth.
**Provenance Validation**: Like ECE, this is calculated post-run and attached to the trace metadata.

### 3. Epistemic Shapley Attribution
**Source Function**: `EpistemicShapley.compute(agents, uncertainty_fn)`
**Formula**: Standard Shapley value computation over the power set of agents, where the characteristic function $v(S)$ is the average epistemic uncertainty of subset $S$.
**Provenance Validation**: Calculated live during `DecisionEngine.decide()` and explicitly logged in `DecisionTrace.attribution`.

### 4. Communication Reduction
**Source Function**: `InformationGainRouter.forward()`
**Formula**: $1 - \frac{\text{messages\_sent}}{\text{max\_possible\_messages}}$
**Provenance Validation**: Derived directly from the active graph topology and logged in `DecisionTrace.communication`.

## Provenance Enforcement
Every metric calculation emits a `MetricProvenance` dataclass guaranteeing traceability to the specific Run ID, Git Commit, and random Seed.
