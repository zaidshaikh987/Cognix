import sys
import numpy as np
from cognix.metrics.evaluation import expected_calibration_error, escalation_f1
from cognix.belief.loopy_bp import LoopyBeliefPropagation
from cognix.belief.base import BeliefState
from cognix.communication.top_k import TopKCommunication
from cognix.communication.information_gain import InformationGainRouter
from cognix.calibration.temperature import TemperatureScaling

def main():
    print("Testing imports and instantiation...")
    
    # 1. Metrics
    conf = np.array([0.9, 0.8, 0.4])
    acc = np.array([1, 1, 0])
    ece = expected_calibration_error(conf, acc)
    print(f"ECE: {ece:.4f}")
    
    # 2. LBP
    lbp = LoopyBeliefPropagation(max_iterations=2)
    b1 = BeliefState(agent_id="a1", belief=np.array([0.2, 0.8]), alpha=0, beta_param=0, confidence=0.8)
    b2 = BeliefState(agent_id="a2", belief=np.array([0.3, 0.7]), alpha=0, beta_param=0, confidence=0.7)
    final, kl = lbp.run(beliefs={"a1": b1, "a2": b2}, adjacency={"a1": ["a2"], "a2": ["a1"]})
    print(f"LBP Keys: {list(final.keys())}")
    
    # 3. Comm
    topk = TopKCommunication(k=1)
    preds = {"a1": 0.9, "a2": 0.5}
    unc = {"a1": 0.1, "a2": 0.6}
    selected = topk.select_broadcasters(preds, unc)
    print(f"TopK Selected: {selected}")
    
    # 4. Calibration
    ts = TemperatureScaling()
    logits = np.array([[2.0, 1.0], [0.5, 2.5]])
    labels = np.array([0, 1])
    ts.fit(logits, labels)
    print(f"Temp after fit: {ts.temperature:.4f}")
    
    print("All tests passed.")

if __name__ == "__main__":
    main()
