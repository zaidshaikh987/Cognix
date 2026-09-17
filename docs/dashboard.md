# Interactive Dashboards

COGNIX provides local web applications to help researchers and developers visualize agent behavior, uncertainty separation, and benchmark performance.

## Installation

Ensure you have installed the optional dashboard dependencies:

```bash
pip install "cognix[dashboard]"
```

## Analytics Dashboard

The **Analytics Dashboard** is a static frontend powered by a local API that parses the `all_summaries.json` and raw benchmark logs. 

It provides highly detailed comparative charts (rendered via Chart.js) for:
- Model comparisons (NoGraph vs. StandardGAT vs. EpistemicGAT).
- Calibration curves and Expected Calibration Error (ECE).
- Epistemic vs. Aleatoric uncertainty scatter plots.
- P50, P95, and P99 Latency distributions.
- Statistical significance (Wilcoxon p-values) between models.

**To launch:**
```bash
python dashboard/app.py
```
Then navigate to `http://localhost:8000`.

*Note: Dashboard values are read directly from actual benchmark result files. It will display "N/A" for scenarios that have not yet been evaluated.*

## Live Interactive Simulation

For demonstrations, COGNIX also includes a "live" dashboard that runs the decision engine in a continuous loop, generating synthetic anomalies on the fly and plotting the live decision trace and dynamic graph attention.

**To launch:**
```bash
python dashboard_live/app.py
```
Then navigate to `http://localhost:8001`.
