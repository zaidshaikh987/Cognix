# COGNIX Framework — Data Sources and Modes

To maintain research integrity, COGNIX explicitly categorizes data into three modes. The dashboard and all experiment outputs must visibly declare which mode is active.

## 1. REAL EXTERNAL DATA (Mode: DATASET)
Data sourced from verifiable external benchmarks.
- **Examples**: nuScenes, KITTI, standard UCI datasets.
- **Rules**: Must not be modified to force a result. Must be loaded via a `DatasetAdapter`.
- **Status in COGNIX**: Not yet integrated. (Priority 1 for next phase).

## 2. CONTROLLED SYNTHETIC DATA (Mode: SYNTHETIC)
Mathematically defined data generated to isolate specific algorithms.
- **Examples**: Gaussian noise injection, manual OOD manipulation, synthetic sine wave mock agents.
- **Rules**: Must be explicitly labeled as SYNTHETIC. Allowed for unit testing, algorithm validation, and dashboard UI testing.
- **Status in COGNIX**: Currently drives the Dashboard (`CognixSampleAgent` uses `math.sin` and random noise).

## 3. LIVE SIMULATION / SENSORS (Mode: LIVE / SIMULATION)
Data arriving in real-time from an external environment.
- **Examples**: CARLA simulator, physical Toy Car hardware.
- **Rules**: Must track latency. Must log the exact sequence of events for reproducibility.
- **Status in COGNIX**: Pipeline supports it (latency tracking is fully implemented), but CARLA/hardware adapters are not yet built.

## Immediate Action Required
The dashboard currently mixes "Live" UI elements with "Synthetic" data generators. The dashboard header must be updated to display a prominent `SYNTHETIC MODE` badge until real datasets or simulations are attached.
