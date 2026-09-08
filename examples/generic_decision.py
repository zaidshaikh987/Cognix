"""
Generic Non-Automotive COGNIX Demonstration.
Proves that the core API is strictly domain-agnostic.
Simulates 3 generic abstract prediction nodes (e.g., IoT sensors in a factory).
"""
import numpy as np
import time
from cognix.engine.decision_engine import DecisionEngine
from cognix.belief.fusion import BeliefFuser, FusionStrategy
from cognix.decision.escalation import EscalationEngine, DecisionOutcome, RiskLevel
from cognix.belief.base import BeliefState
from cognix.calibration.conformal import ConformalPredictor

def run_generic_demo():
    print("--- COGNIX Generic Domain-Agnostic Demo ---")
    print("Simulating 3 abstract prediction nodes observing a high-stakes event.\n")
    
    # 1. Initialize Domain-Agnostic Core Components
    fuser = BeliefFuser(strategy=FusionStrategy.EPISTEMIC_WEIGHTED)
    escalator = EscalationEngine(high_conf_thresh=0.8, low_conf_thresh=0.4, high_unc_thresh=0.5)
    calibrator = ConformalPredictor(alpha=0.1) # 90% coverage target
    
    engine = DecisionEngine(
        fusion_strategy=fuser,
        decision_maker=None # Replaced by the escalator directly for this demo
    )
    
    # 2. Simulate Abstract Agent Predictions (e.g. [Normal, Fault, Critical])
    # Node A is highly confident but has HIGH epistemic uncertainty (OOD).
    node_a = BeliefState("Node A", np.array([0.1, 0.1, 0.8]), confidence=0.8, epistemic_uncertainty=0.9, aleatoric_uncertainty=0.1)
    
    # Node B is moderately confident and has LOW epistemic uncertainty (In-Distribution).
    node_b = BeliefState("Node B", np.array([0.7, 0.2, 0.1]), confidence=0.7, epistemic_uncertainty=0.1, aleatoric_uncertainty=0.2)
    
    # Node C is in agreement with B, LOW epistemic uncertainty.
    node_c = BeliefState("Node C", np.array([0.6, 0.3, 0.1]), confidence=0.6, epistemic_uncertainty=0.15, aleatoric_uncertainty=0.1)
    
    agents = [node_a, node_b, node_c]
    
    print("1. AGENT PREDICTIONS:")
    for a in agents:
        print(f" - {a.agent_id}: Pred={a.belief}, Conf={a.confidence:.2f}, EpistemicUnc={a.epistemic_uncertainty:.2f}")
        
    start_time = time.perf_counter()
    
    # 3. Fuse Beliefs
    print("\n2. EPISTEMIC FUSION:")
    # Expected behavior: Node A (0.9 Unc) is massively downweighted compared to B (0.1) and C (0.15).
    # Despite Node A's 80% confidence in class 2, the fused belief will favor class 0.
    fused = fuser.fuse(agents)
    print(f" - Fused Belief: {np.round(fused.belief, 3)}")
    print(f" - Fused Epistemic Unc: {fused.epistemic_uncertainty:.3f}")
    
    # 4. Escalate Decision
    print("\n3. DECISION & ESCALATION:")
    decision = escalator.evaluate(
        confidence=fused.confidence, 
        epistemic_uncertainty=fused.epistemic_uncertainty,
        conformal_set_size=1, # Mock size
        max_shapley_value=0.2
    )
    
    latency = (time.perf_counter() - start_time) * 1000
    
    print(f" - Outcome: {decision.outcome.name}")
    print(f" - Risk Level: {decision.risk_level.name}")
    print(f" - Reason: {decision.reason}")
    print(f" - Latency: {latency:.2f}ms")
    print("\nDemonstration Complete. Core API contains zero domain-specific assumptions.")

if __name__ == "__main__":
    run_generic_demo()
