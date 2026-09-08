# COGNIX Framework — Dummy Data Audit

This document explicitly calls out every location in the codebase where hard-coded, random, synthetic, or placeholder data is used instead of genuine computation or real data.

## 1. `dashboard/app.py`

### `CognixSampleAgent.predict`
- **Line 126:** `noise = 0.05 * math.sin(t * 0.3 + hash(self.name) % 10)`
- **Line 129:** `conf = random.uniform(0.35, 0.55)`
- **Why it exists:** Provides visual movement in the UI to simulate sensor noise.
- **Affects research results?** Yes, entirely invalidates dashboard visuals as research data.
- **Replacement Strategy:** Must be replaced with an adapter that reads from a real benchmark dataset or model outputs.

### `CognixSampleAgent.estimate_uncertainty`
- **Line 143:** `noise = 0.03 * math.sin(t * 0.2 + hash(self.name) % 7)`
- **Line 146:** `epi = random.uniform(0.55, 0.75)`
- **Line 152:** `alea = epi * 0.55`
- **Why it exists:** Provides synthetic uncertainty visualization.
- **Affects research results?** Yes.
- **Replacement Strategy:** Must use actual output from `cognix/uncertainty/mc_dropout.py` or similar.

### `run_cognix_cycle`
- **Line 214:** `shapley = { ... result.agent_trust_weights.get(a.name, 0) * (result.epistemic_uncertainty or 0.2) ... }`
- **Why it exists:** Generates plausible looking Shapley attribution without running the computationally expensive `EpistemicShapley` Monte Carlo permutations.
- **Affects research results?** Yes, bypasses actual Shapley math.
- **Replacement Strategy:** Call `engine.attribution.compute()`.

## 2. `cognix/engine/pipeline.py`

### `_estimate_uncertainties`
- **Line 283:** `epistemic = max(0.0, 1.0 - confidence) * 0.5`
- **Why it exists:** Fallback heuristic when no `uncertainty_estimator` is provided to the engine.
- **Affects research results?** If hit, it breaks the core hypothesis (that epistemic uncertainty is distinct from confidence).
- **Replacement Strategy:** Should raise a warning if hit during an experiment.

### `_aggregate_uncertainty`
- **Line 326:** `unc = uncertainties.get(agent_id, {"epistemic": 0.2, "aleatoric": 0.1, "total": 0.3})`
- **Why it exists:** Hardcoded default dictionary if agent uncertainty is missing.
- **Affects research results?** Yes, contaminates aggregates.
- **Replacement Strategy:** Handle missing data explicitly (e.g. drop agent or error).

## 3. `cognix/decision/escalation.py`

### `compute_escalation_metrics`
- **Line 73:** `correct = sum(1 for p, t in zip(predictions, true_outcomes) if p == t)`
- **Why it exists:** Simplified stand-in for full selective risk/escalation performance metrics.
- **Affects research results?** Weakens evaluation.
- **Replacement Strategy:** Implement actual precision/recall per risk tier.

## Summary

The core algorithmic logic of COGNIX is mostly clean of dummy data. The *vast majority* of dummy data is concentrated in the **Dashboard presentation layer** to ensure the UI had something to render before datasets were integrated. 

To convert this framework to a rigorous research state, the dashboard must be re-wired to consume offline datasets or live inference nodes rather than `CognixSampleAgent`.
