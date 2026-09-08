"""
Hello COGNIX: Minimal example demonstrating the framework.

This example creates 3 simple callable agents with simulated predictions
and runs them through the real COGNIX DecisionEngine.

Usage:
    python examples/basic/hello_cognix.py
"""

import numpy as np
from cognix import DecisionEngine
from cognix.agents.base import AgentPrediction


# ---------------------------------------------------------------------------
# Define three simple agents as callables
# ---------------------------------------------------------------------------

class SimpleAgent:
    """
    A minimal agent that returns a fixed prediction with simulated uncertainty.
    In a real system this would wrap a trained model.
    """

    def __init__(self, name: str, confidence: float, epistemic_uncertainty: float) -> None:
        self.agent_id = name
        self.name = name
        self._confidence = confidence
        self._epistemic = epistemic_uncertainty

    def predict(self, input_data: object) -> dict:
        """Return prediction dict compatible with the COGNIX pipeline."""
        # Simulate a GO/BRAKE binary decision as probabilities [P(BRAKE), P(GO)]
        p_go = self._confidence
        return {
            "label": "GO" if p_go >= 0.5 else "BRAKE",
            "probabilities": np.array([1.0 - p_go, p_go]),
            "confidence": p_go,
        }

    def estimate_uncertainty(self, input_data: object) -> object:
        """Return a simple uncertainty estimate."""
        from cognix.uncertainty.base import UncertaintyEstimate
        return UncertaintyEstimate(
            aleatoric=self._epistemic * 0.5,
            epistemic=self._epistemic,
            total=self._epistemic * 1.5,
            raw_samples=None,
            method="simulated",
        )


def main() -> None:
    print("=" * 60)
    print("  COGNIX — Hello World Example")
    print("=" * 60)

    # Create three agents with different confidence/uncertainty profiles
    camera_agent = SimpleAgent("Camera",    confidence=0.85, epistemic_uncertainty=0.10)
    lidar_agent  = SimpleAgent("LiDAR",     confidence=0.90, epistemic_uncertainty=0.08)
    v2v_agent    = SimpleAgent("V2V",       confidence=0.60, epistemic_uncertainty=0.45)  # degraded

    # Build the COGNIX DecisionEngine
    # Modules are connected via string shortcuts or object instances
    engine = DecisionEngine(
        uncertainty=None,          # Each agent provides its own UQ
        belief="epistemic_weighted",  # COGNIX proposed fusion mechanism
        calibrator=None,           # Calibrator not fitted in this minimal demo
        escalation=True,           # Enable escalation logic
    )

    # Run the decision pipeline
    input_data = {"frame": "synthetic_intersection_image"}
    result = engine.decide(
        agents=[camera_agent, lidar_agent, v2v_agent],
        input_data=input_data,
        context={"scenario": "normal_intersection"},
    )

    # Display results
    print(f"\n  Decision      : {result.decision.value}")
    print(f"  Confidence    : {result.confidence:.1%}")
    print(f"  Risk Level    : {result.risk_level.value}")
    print(f"  Epistemic UQ  : {result.epistemic_uncertainty:.3f}")
    print(f"  Aleatoric UQ  : {result.aleatoric_uncertainty:.3f}")
    print(f"  Total UQ      : {result.total_uncertainty:.3f}")
    print(f"  Escalation    : {result.escalation_required}")
    print(f"  Total Latency : {result.total_latency_ms:.2f} ms")

    print("\n  Agent Trust Weights:")
    for agent_id, weight in result.agent_trust_weights.items():
        print(f"    {agent_id:<12} : {weight:.3f}")

    print(f"\n  Explanation:\n    {result.explanation}")

    print("\n  Reasoning Steps:")
    for step in result.reasoning_steps:
        print(f"    {step}")

    print("\n  Summary:", result.summary())
    print("=" * 60)


if __name__ == "__main__":
    main()
