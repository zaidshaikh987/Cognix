from .base import StandardAgent
# Provide backward compatibility aliases for tests
BaseAgent = StandardAgent

# NullAgent stub for tests
class NullAgent(StandardAgent):
    def __init__(self, agent_id="null"):
        super().__init__(agent_id, None, None)

__all__ = [
    "StandardAgent",
    "BaseAgent",
    "NullAgent"
]
