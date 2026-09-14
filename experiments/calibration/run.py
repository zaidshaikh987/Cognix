"""
Usage: python -m experiments.calibration.run

Experiment: Compare uncalibrated vs temperature-scaled vs conformal prediction.
Metrics: ECE, coverage, set size.
"""

import json
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

from cognix.calibration.temperature import TemperatureScaling
from cognix.calibration.conformal import ConformalPredictor, evaluate_coverage
from cognix.calibration.metrics import expected_calibration_error


def compute_ece_for_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    confidence = np.max(probs, axis=1)
    accuracy = (np.argmax(probs, axis=1) == labels).astype(float)
    return float(expected_calibration_error(confidence, accuracy))


def run_experiment():
    print("Starting Calibration Experiment...")

    # 1. Generate reproducible synthetic classification dataset
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42
    )

    # 2. Separate train, calibration, and test splits (900 train / 300 cal / 300 test)
    X_train, y_train = X[:900], y[:900]
    X_cal, y_cal = X[900:1200], y[900:1200]
    X_test, y_test = X[1200:], y[1200:]

    # 3. Train the model only on the train split
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    probs_cal = model.predict_proba(X_cal)
    probs_test = model.predict_proba(X_test)

    # 4. Uncalibrated
    uncalibrated_ece = compute_ece_for_probs(probs_test, y_test)

    # 5. Temperature Scaling using pseudo-logits: np.log(np.clip(probs, 1e-12, 1.0))
    logits_cal = np.log(np.clip(probs_cal, 1e-12, 1.0))
    logits_test = np.log(np.clip(probs_test, 1e-12, 1.0))

    ts = TemperatureScaling()
    ts.fit(logits_cal, y_cal)
    temp_scaled_probs = ts.predict(logits_test)
    temp_scaled_ece = compute_ece_for_probs(temp_scaled_probs, y_test)

    # 6. Conformal Predictor (alpha=0.05)
    cp = ConformalPredictor()
    cp.fit(probs_cal, y_cal)
    conformal_sets = cp.predict(probs_test, alpha=0.05)
    conformal_cov = float(evaluate_coverage(conformal_sets, y_test))
    conformal_avg_set_size = float(np.mean([len(s.prediction_set) for s in conformal_sets]))

    results = {
        "uncalibrated": {
            "ece": uncalibrated_ece,
            "coverage": None,
            "avg_set_size": 1.0
        },
        "temperature_scaled": {
            "ece": temp_scaled_ece,
            "coverage": None,
            "avg_set_size": 1.0,
            "temperature": float(ts.temperature)
        },
        "conformal": {
            "ece": None,
            "coverage": conformal_cov,
            "avg_set_size": conformal_avg_set_size,
            "alpha": 0.05
        }
    }

    os.makedirs("results/calibration", exist_ok=True)
    with open("results/calibration/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Calibration Experiment complete.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_experiment()

