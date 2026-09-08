# COGNIX Research Dashboard Modes

To prevent the misrepresentation of synthetic or unverified data as real live autonomous driving results, the COGNIX dashboard enforces strict **Data Modes**.

## Modes

1. **`SYNTHETIC`**
   - **Meaning**: The dashboard is visualizing a deterministic, mathematical run (e.g., `RQ_SYNTHETIC_001`).
   - **UI Status**: `RUNNING` (or `OFFLINE` if websocket disconnects).
   - **Usage**: Used for algorithm validation, unit testing, and controlled baseline comparisons.

2. **`DATASET`**
   - **Meaning**: The dashboard is replaying inferences performed over a fixed, real-world public dataset (e.g., nuScenes, CIFAR-10C).
   - **UI Status**: `RUNNING`.
   - **Usage**: Used for formal benchmarking and paper publication.

3. **`SIMULATION`**
   - **Meaning**: The dashboard is connected to a 3D physical simulator (e.g., CARLA) rendering dynamic frames in real-time.
   - **UI Status**: `SIMULATING`.
   - **Usage**: Used for closed-loop validation of the escalation engine.

4. **`LIVE`**
   - **Meaning**: The dashboard is connected to real physical sensors on real hardware.
   - **UI Status**: `LIVE` (Blinking Green).
   - **Usage**: Used for physical deployment.

## Provenance Guarantee
If a `DecisionTrace` cannot provide a valid `dataset`, `run_id`, and `model`, the dashboard will drop into a **`DEMO / UNCONFIGURED`** state and refuse to display quantitative metrics. No fields will be fabricated or assigned random plausible values.
