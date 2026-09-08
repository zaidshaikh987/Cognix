"""
Communication protocols and structures.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

@dataclass
class Message:
    sender_id: str
    receiver_id: str
    content: dict[str, Any]
    timestamp: float
    message_type: str

@dataclass
class CommunicationStats:
    total_messages: int
    messages_sent: int
    bandwidth_used: float
    reduction_ratio: float
    avg_latency_ms: float

class CommunicationProtocol(ABC):
    @abstractmethod
    def select_receivers(self, agent_id: str, all_agents: list[str], context: dict[str, Any]) -> list[str]:
        pass

    @abstractmethod
    def format_message(self, agent: str, prediction: dict[str, Any]) -> Message:
        pass
