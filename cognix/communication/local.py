import json
import logging
from typing import Any

from cognix.core.interfaces import TransportInterface

logger = logging.getLogger(__name__)

class LocalTransport(TransportInterface):
    """
    Local in-memory transport for research and local evaluation.
    """
    
    def __init__(self):
        self._message_queue = []
        
    def send(self, message: Any) -> None:
        try:
            # Serialize/deserialize to mimic actual transport boundaries
            serialized = json.dumps(message)
            self._message_queue.append(serialized)
        except Exception as e:
            logger.error(f"LocalTransport serialization failed: {e}")
            
    def receive(self) -> Any:
        if not self._message_queue:
            return None
        serialized = self._message_queue.pop(0)
        return json.loads(serialized)
