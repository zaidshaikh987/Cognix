# Uncertainty Quantification

COGNIX relies fundamentally on the separation of predictive uncertainty into two components: **Epistemic** and **Aleatoric**.

## Types of Uncertainty

### Epistemic Uncertainty
Also known as *model uncertainty*, this measures the model's lack of knowledge. It is high when the model encounters data that is significantly different from its training set (Out-Of-Distribution).
- **Example**: An autonomous vehicle camera sees snow for the first time.
- **Resolution**: Can be reduced by collecting more training data (e.g., training on snow images).

### Aleatoric Uncertainty
Also known as *data uncertainty*, this measures the inherent noise or randomness in the observation itself. 
- **Example**: A camera trying to read a blurry license plate at night.
- **Resolution**: Cannot be reduced by more data; requires better sensors (e.g., a higher-resolution camera).

## Implemented Methods

COGNIX supports standard approaches for approximating these uncertainties.

### Monte Carlo Dropout (MCDropout)

**Import**: `from cognix import MonteCarloDropout`

MCDropout leaves dropout layers active during inference and passes the same input through the network $T$ times. 
- **Epistemic Uncertainty**: Measured as the variance of the $T$ predictions.
- **Aleatoric Uncertainty**: Measured as the expected value of the predicted variances (e.g., the entropy of the mean prediction).

```python
from cognix import MonteCarloDropout

uq = MonteCarloDropout(n_samples=20)
result = uq.estimate(model, data)
print(result.epistemic, result.aleatoric)
```

### Deep Ensembles

**Import**: `from cognix import DeepEnsemble`

Instead of stochastic dropout, Deep Ensembles train $M$ identical architectures with different random initializations. Inference is performed across all $M$ models.
- **Advantage**: Often yields better calibrated and more robust uncertainty estimates than MCDropout.
- **Disadvantage**: Requires maintaining and executing $M$ distinct models in memory.
