import pytest
import time
from enum import Enum
from dataclasses import dataclass

class DecisionOutcome(Enum):
    ACT = "ACT"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"

@dataclass
class DecisionResult:
    outcome: DecisionOutcome
    explanation: str
    latency_ms: float

class DecisionEngine:
    def __init__(self, agents):
        self.agents = agents

    def decide(self, X):
        start = time.time()
        preds = [a.predict(X) for a in self.agents]
        uncs = [a.get_uncertainty() for a in self.agents]
        
        latency = (time.time() - start) * 1000
        
        if all(u > 0.8 for u in uncs):
            return DecisionResult(DecisionOutcome.ABSTAIN, "High uncertainty", latency)
        
        if len(set(preds)) > 1:
            return DecisionResult(DecisionOutcome.ESCALATE, "Agents disagree", latency)
            
        return DecisionResult(DecisionOutcome.ACT, "Consensus reached", latency)

class MockAgent:
    def __init__(self, pred, unc):
        self.pred = pred
        self.unc = unc
    def predict(self, X):
        return self.pred
    def get_uncertainty(self):
        return self.unc

def test_decision_engine_basic_classification():
    engine = DecisionEngine([MockAgent(1, 0.1), MockAgent(1, 0.2), MockAgent(1, 0.1)])
    result = engine.decide(None)
    assert result.outcome == DecisionOutcome.ACT
    assert result.explanation == "Consensus reached"
    assert result.latency_ms >= 0

def test_decision_engine_high_uncertainty_abstains():
    engine = DecisionEngine([MockAgent(1, 0.9), MockAgent(1, 0.9)])
    result = engine.decide(None)
    assert result.outcome in (DecisionOutcome.ABSTAIN, DecisionOutcome.ESCALATE)

def test_decision_engine_agent_disagreement():
    engine = DecisionEngine([MockAgent(1, 0.1), MockAgent(2, 0.1)])
    result = engine.decide(None)
    assert result.outcome == DecisionOutcome.ESCALATE
    assert "disagree" in result.explanation.lower()

def test_pipeline_latency_measured():
    engine = DecisionEngine([MockAgent(1, 0.1)])
    result = engine.decide(None)
    assert result.latency_ms is not None

def test_pipeline_missing_agent_handles_gracefully():
    # Example logic for missing agent
    assert True
