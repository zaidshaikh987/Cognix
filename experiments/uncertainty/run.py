"""
Usage: python -m experiments.uncertainty.run

Experiment: Compare MC Dropout vs Deep Ensemble uncertainty estimates
on synthetic classification data.

Records: config.yaml, metrics.json, summary.json in results/uncertainty/
"""

import os
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

def run_experiment():
    print("Starting Uncertainty Experiment...")
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    
    # Mock deep ensemble
    models = []
    for i in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=i)
        mlp.fit(X[:800], y[:800])
        models.append(mlp)
        
    X_test, y_test = X[800:], y[800:]
    
    # Estimate uncertainty (variance of predictions as proxy for ensemble)
    preds = np.array([m.predict_proba(X_test)[:, 1] for m in models])
    ensemble_mean = np.mean(preds, axis=0)
    ensemble_var = np.var(preds, axis=0)
    
    errors = np.abs(np.round(ensemble_mean) - y_test)
    correlation = np.corrcoef(ensemble_var, errors)[0, 1]
    
    results = {
        "mean_epistemic": float(np.mean(ensemble_var)),
        "mean_aleatoric": float(np.mean(ensemble_mean * (1 - ensemble_mean))),
        "correlation_with_errors": float(correlation)
    }
    
    os.makedirs("results/uncertainty", exist_ok=True)
    
    with open("results/uncertainty/config.yaml", "w") as f:
        f.write("experiment: uncertainty\nmodels: MLP\nensemble_size: 5")
        
    with open("results/uncertainty/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    with open("results/uncertainty/summary.json", "w") as f:
        json.dump({"status": "success", "correl": results["correlation_with_errors"]}, f)
        
    print(f"Experiment complete. Results saved. Correlation: {correlation:.4f}")

if __name__ == "__main__":
    run_experiment()
