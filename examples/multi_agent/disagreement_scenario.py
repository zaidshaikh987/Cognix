"""
Disagreement Scenario
Demonstrates how Cognix handles agent disagreement.
"""

from enum import Enum
import time

class DecisionOutcome(Enum):
    ACT = "ACT"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"

class MockAgent:
    def __init__(self, name, pred, unc):
        self.name = name
        self.pred = pred
        self.unc = unc
    def predict(self, X): return self.pred
    def get_uncertainty(self): return self.unc

class DecisionEngine:
    def __init__(self, agents):
        self.agents = agents
    def decide(self):
        preds = [a.predict(None) for a in self.agents]
        unique = set(preds)
        if len(unique) > 1:
            risk = "HIGH"
            explanation = f"Disagreement detected among agents: {preds}"
            return DecisionOutcome.ESCALATE, risk, explanation
        return DecisionOutcome.ACT, "LOW", "Consensus"

def main():
    agents = [
        MockAgent("Agent_1", "Class_A", 0.2),
        MockAgent("Agent_2", "Class_A", 0.3),
        MockAgent("Agent_3", "Class_B", 0.1),
        MockAgent("Agent_4", "Class_B", 0.15)
    ]
    
    print("Scenario: 4 agents, 2 predict Class_A, 2 predict Class_B")
    engine = DecisionEngine(agents)
    outcome, risk, exp = engine.decide()
    
    print("\n--- COGNIX Decision ---")
    print(f"Outcome: {outcome.value}")
    print(f"Risk Level: {risk}")
    print(f"Explanation: {exp}")

if __name__ == "__main__":
    main()
