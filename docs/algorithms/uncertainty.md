# Uncertainty Estimation Algorithms

COGNIX relies heavily on quantifying uncertainty. We implement several established baselines.

## 1. Monte Carlo Dropout (MC Dropout)
**Purpose**: Estimate epistemic uncertainty without training multiple models.
**Method**: Apply dropout during *inference*.
**Formulation**: $Var(y) \approx \frac{1}{N}\sum_{i=1}^{N}(f^{\hat{W}_i}(x) - \bar{y})^2$
**Reference**: *Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)*
**Limitations**: Tends to underestimate uncertainty compared to Deep Ensembles.

## 2. Deep Ensembles
**Purpose**: High-quality epistemic uncertainty estimation.
**Method**: Train $M$ independent models with different random initializations.
**Formulation**: Average the predictions; variance across models represents epistemic uncertainty.
**Reference**: *Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation" (NIPS 2017)*
**Limitations**: High computational cost ($O(M)$ during training and inference).

## 3. Aleatoric vs Epistemic Decomposition
**Purpose**: Separate data noise from model ignorance.
**Method**: Model outputs a mean $\mu(x)$ and variance $\sigma^2(x)$ (aleatoric). Total variance is aleatoric + ensemble variance (epistemic).
**Reference**: *Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning?" (NIPS 2017)*

## 4. OOD Detection (Mahalanobis Distance)
**Purpose**: Detect Out-Of-Distribution samples.
**Method**: Fit Gaussian distributions to feature representations of training data; compute Mahalanobis distance at inference.
**Reference**: *Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples" (NeurIPS 2018)*
