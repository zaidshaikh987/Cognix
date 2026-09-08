# Novelty and Prior Art

This document outlines what is established in literature versus what COGNIX proposes.

## 1. Established Algorithms (Prior Art)
- Monte Carlo Dropout (Gal 2016)
- Deep Ensembles (Lakshminarayanan 2017)
- Temperature Scaling for Calibration (Guo 2017)
- Evidential Deep Learning (Sensoy 2018)

## 2. Related Research
- Sensor fusion via attention mechanisms (e.g., TransFuser for AVs).
- Federated learning with uncertainty (various).

## 3. Proposed Cognix Mechanisms
- **Epistemic-Weighted Fusion**: Weighting agents by the inverse of their *epistemic* uncertainty specifically, rather than total uncertainty.
- **Epistemic GAT (Graph Attention)**: A Graph Attention Network where edge weights are explicitly modulated by the epistemic uncertainty of the source node.
- **Epistemic Shapley Values**: Modifying standard Shapley value computation to attribute the source of *uncertainty* in a fused prediction back to individual sensors.

## 4. Potential Differentiators
- True decoupling of aleatoric and epistemic uncertainty in the fusion layer.
- Real-time fallback triggers based specifically on out-of-distribution (epistemic) spikes.

## 5. Unknowns Requiring Prior-Art Search
- Have Epistemic Shapley values been formally proposed in cooperative game theory applied to ML?
- Has Epistemic-Weighted Fusion been patented in the context of Autonomous Vehicles?

## 6. Legal Disclaimer
*This document is for research purposes only. It does not constitute legal advice. A formal patent search by qualified legal counsel is required before claiming novel IP.*
