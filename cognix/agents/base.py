import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np

@dataclass
class AgentPrediction:
    """Represents a prediction made by an agent in the Cognix framework."""
    agent_id: str
    prediction: np.ndarray
    probabilities: np.ndarray
    confidence: float
    timestamp: datetime
    raw_output: Optional[Any] = None

@dataclass
class AgentHealth:
    """Represents the health and reliability status of an agent."""
    agent_id: str
    is_healthy: bool
    reliability_score: float  # Value between 0.0 and 1.0
    last_updated: datetime
    degradation_reason: Optional[str] = None

@dataclass
class AgentMetadata:
    """Metadata describing an agent's capabilities and domain."""
    agent_id: str
    name: str
    domain: str
    modality: str
    description: str

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the COGNIX framework.
    Requires implementation of prediction, uncertainty estimation, metadata, health, and explanation.
    """
    
    @abc.abstractmethod
    def predict(self, inputs: Any) -> AgentPrediction:
        """Generate a prediction given the input."""
        pass

    @abc.abstractmethod
    def estimate_uncertainty(self, inputs: Any) -> float:
        """Estimate the uncertainty of a prediction on the given inputs."""
        pass

    @abc.abstractmethod
    def metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        pass

    @abc.abstractmethod
    def health(self) -> AgentHealth:
        """Return the current health status of the agent."""
        pass

    @abc.abstractmethod
    def explain(self, inputs: Any) -> str:
        """Provide an explanation for the agent's behavior on the given inputs."""
        pass

class NullAgent(BaseAgent):
    """
    A safe default agent that returns maximum uncertainty and a default zero prediction.
    Useful for fallbacks when active agents degrade or fail.
    """
    
    def predict(self, inputs: Any) -> AgentPrediction:
        return AgentPrediction(
            agent_id="null_agent",
            prediction=np.array([0]),
            probabilities=np.array([0.0]),
            confidence=0.0,
            timestamp=datetime.now(),
            raw_output=None
        )

    def estimate_uncertainty(self, inputs: Any) -> float:
        return 1.0  # Maximum uncertainty

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            agent_id="null_agent",
            name="Null Agent",
            domain="none",
            modality="none",
            description="Safe fallback agent returning max uncertainty."
        )

    def health(self) -> AgentHealth:
        return AgentHealth(
            agent_id="null_agent",
            is_healthy=True,
            reliability_score=1.0,
            last_updated=datetime.now()
        )

    def explain(self, inputs: Any) -> str:
        return "Null agent always returns a default zero prediction and max uncertainty."
