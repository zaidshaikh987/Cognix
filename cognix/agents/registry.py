import threading
from typing import Dict, List, Optional
from cognix.core.interfaces import AgentInterface

class AgentRegistry:
    """
    Thread-safe registry for managing active agents in the COGNIX framework.
    Provides mechanisms to register, unregister, and query agents based on their health.
    """
    def __init__(self):
        self._agents: Dict[str, AgentInterface] = {}
        self._lock = threading.RLock()

    def register(self, agent: AgentInterface) -> None:
        """Register a new agent in the registry."""
        with self._lock:
            meta = agent.metadata()
            agent_id = meta.get("id") or meta.get("agent_id") or "unknown_agent"
            self._agents[agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        """Unregister an agent by its ID."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]

    def get(self, agent_id: str) -> Optional[AgentInterface]:
        """Retrieve an agent by its ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentInterface]:
        """List all registered agents."""
        with self._lock:
            return list(self._agents.values())

    def healthy_agents(self) -> List[AgentInterface]:
        """List all registered agents that are currently healthy."""
        with self._lock:
            return [agent for agent in self._agents.values() if agent.health().is_healthy]

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
