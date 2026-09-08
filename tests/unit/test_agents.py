import pytest
import numpy as np

class NullAgent:
    def predict(self, X):
        return None
    def get_uncertainty(self):
        return 1.0

class AgentRegistry:
    def __init__(self):
        self.agents = {}
    def register(self, name, agent):
        self.agents[name] = agent
    def unregister(self, name):
        if name in self.agents:
            del self.agents[name]
    def get_healthy(self):
        return {k: v for k, v in self.agents.items() if v.get_uncertainty() < 1.0}

class CallableAdapter:
    def __init__(self, func):
        self.func = func
    def predict(self, X):
        return self.func(X)

class SklearnAdapter:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def test_null_agent_returns_max_uncertainty():
    agent = NullAgent()
    assert agent.get_uncertainty() == 1.0

def test_agent_registry_register_unregister():
    registry = AgentRegistry()
    registry.register("a1", NullAgent())
    assert "a1" in registry.agents
    registry.unregister("a1")
    assert "a1" not in registry.agents

def test_callable_adapter_wraps_function():
    adapter = CallableAdapter(lambda x: x * 2)
    assert adapter.predict(3) == 6

def test_sklearn_adapter():
    class DummySklearn:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])
    adapter = SklearnAdapter(DummySklearn())
    np.testing.assert_array_equal(adapter.predict_proba(None), np.array([[0.1, 0.9]]))

def test_registry_healthy_agents_filter():
    registry = AgentRegistry()
    registry.register("a1", NullAgent())
    class HealthyAgent:
        def get_uncertainty(self):
            return 0.1
    registry.register("a2", HealthyAgent())
    healthy = registry.get_healthy()
    assert "a2" in healthy
    assert "a1" not in healthy
