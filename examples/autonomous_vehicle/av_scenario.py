"""
Autonomous Vehicle Multi-Agent Scenario
"""

from enum import Enum

class DecisionOutcome(Enum):
    ACT = "ACT"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"

class MockAgent:
    def __init__(self, name, pred, unc):
        self.name = name
        self.pred = pred
        self.unc = unc
    def predict(self): return self.pred
    def get_uncertainty(self): return self.unc

class CognixAVEngine:
    def decide(self, agents):
        preds = [a.predict() for a in agents]
        uncs = [a.get_uncertainty() for a in agents]
        
        if all(u > 0.8 for u in uncs):
            return {"outcome": DecisionOutcome.ESCALATE, "action": "EMERGENCY_STOP", "reason": "All agents highly uncertain"}
        
        if len(set(preds)) > 1:
            return {"outcome": DecisionOutcome.ESCALATE, "action": "SAFE_STOP", "reason": f"Sensor conflict: {preds}"}
            
        high_unc_agents = [a.name for a, u in zip(agents, uncs) if u > 0.7]
        if high_unc_agents:
            return {"outcome": DecisionOutcome.ACT, "action": preds[0], "reason": f"Consensus with degradation in {high_unc_agents}"}
            
        return {"outcome": DecisionOutcome.ACT, "action": preds[0], "reason": "Normal operation consensus"}

def run_scenario(name, agents):
    print(f"\n=== Scenario: {name} ===")
    engine = CognixAVEngine()
    result = engine.decide(agents)
    print(f"Outcome: {result['outcome'].value}")
    print(f"Action:  {result['action']}")
    print(f"Reason:  {result['reason']}")

def main():
    # S1: Normal
    run_scenario("NORMAL", [
        MockAgent("Camera", "BRAKE", 0.1),
        MockAgent("LiDAR", "BRAKE", 0.1),
        MockAgent("V2V", "BRAKE", 0.2),
        MockAgent("Map", "BRAKE", 0.05)
    ])
    
    # S2: Camera Degradation
    run_scenario("CAMERA DEGRADATION", [
        MockAgent("Camera", "BRAKE", 0.85),
        MockAgent("LiDAR", "BRAKE", 0.1),
        MockAgent("V2V", "BRAKE", 0.2),
        MockAgent("Map", "BRAKE", 0.05)
    ])
    
    # S3: Sensor Conflict
    run_scenario("SENSOR CONFLICT", [
        MockAgent("Camera", "GO", 0.2),
        MockAgent("LiDAR", "BRAKE", 0.1),
        MockAgent("V2V", "BRAKE", 0.2),
        MockAgent("Map", "BRAKE", 0.05)
    ])
    
    # S4: All Uncertain
    run_scenario("ALL UNCERTAIN", [
        MockAgent("Camera", "WAIT", 0.9),
        MockAgent("LiDAR", "WAIT", 0.95),
        MockAgent("V2V", "WAIT", 0.85),
        MockAgent("Map", "WAIT", 0.82)
    ])

if __name__ == "__main__":
    main()
