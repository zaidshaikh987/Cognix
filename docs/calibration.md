# Calibration & Uncertainty Sets

AI models in high-stakes environments must not only be accurate, they must "know what they don't know." Calibration ensures that when a model claims to be 90% confident, it is actually correct 90% of the time.

COGNIX implements two key post-hoc mechanisms:

## 1. Temperature Scaling

**Import**: `from cognix import TemperatureScaling`

A post-processing technique for neural networks. It learns a single scalar parameter $T$ (temperature) on a hold-out calibration dataset.
- If $T > 1$, it "softens" overconfident softmax distributions.
- If $T < 1$, it "sharpens" underconfident predictions.
Importantly, it preserves the argmax (the predicted class) while reducing the Expected Calibration Error (ECE).

## 2. Conformal Predictor

**Import**: `from cognix import ConformalPredictor`

Instead of returning a single point-estimate prediction (e.g., "Pedestrian"), Conformal Prediction returns a **set** of predictions (e.g., `{Pedestrian, Bicycle}`).

It guarantees, with a user-specified probability $1 - \alpha$, that the true label is contained within the returned prediction set, assuming the test data is exchangeable with the calibration data.

```python
import numpy as np
from cognix import ConformalPredictor

# 1. Fit the calibrator on a hold-out set of probabilities and true labels
cp = ConformalPredictor()
calibration_probs = np.random.rand(100, 3) # 100 samples, 3 classes
calibration_labels = np.random.randint(0, 3, 100)
cp.fit(calibration_probs, calibration_labels)

# 2. Generate a prediction set on new test data for a 90% coverage target (alpha=0.1)
test_probs = np.random.rand(1, 3)
prediction_sets = cp.predict(test_probs, alpha=0.1)

print(prediction_sets[0].prediction_set)  # e.g., [0, 2]
```
