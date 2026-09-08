# Data Provenance and Experiment Rules

To ensure strict scientific validity, COGNIX differentiates between **Real Data** and **Synthetic Data**. All experiments run through the framework must adhere to the following documentation protocols.

## 1. Synthetic Data Guidelines
Synthetic data (e.g., Gaussian noise arrays, sine-wave modulated confidences, or mathematically defined random walks) is highly useful for specific validation tasks, but **must never be presented as real-world data**.

**Permitted Uses for Synthetic Data:**
- Controlled uncertainty/degradation tests (e.g., forcing Epistemic Uncertainty to 1.0 to trace the fusion response).
- Unit tests validating mathematical equations (e.g., `tests/test_mathematics.py`).
- Testing pipeline latency and architectural overhead.

**Strict Labeling:**
Any execution utilizing synthetic data must explicitly tag the execution environment or dashboard with:
`DATA MODE: SYNTHETIC`

## 2. Real Data Guidelines
When executing actual research experiments, COGNIX adapters must pull from publicly available, peer-reviewed datasets. 

For **every** real dataset integrated into `cognix/data/adapters.py` (which will be built in the next phase), the following provenance block MUST be generated and appended to this file:

### Example Template (To be populated upon dataset integration)
- **Dataset**: [Name of Dataset, e.g., nuScenes or CIFAR-10C]
- **Dataset Version**: [Version]
- **Dataset Source**: [URL/DOI]
- **License**: [MIT, CC-BY, etc.]
- **Preprocessing**: [Exact normalization, resizing, or feature extraction applied]
- **Train/Val/Test Split**: [Exact numbers]
- **Citation**: [BibTeX]

## 3. The Dashboard
The `dashboard/app.py` currently relies on `CognixSampleAgent` which generates synthetic arrays. Per the post-implementation audit rules, the Dashboard has been updated to explicitly render a `[SYNTHETIC]` badge to prevent any misinterpretation of its output as live sensor data.
