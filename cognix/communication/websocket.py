import json
import logging
from typing import Any
import asyncio

try:
    import websockets
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    _WEBSOCKETS_AVAILABLE = False

from cognix.core.interfaces import TransportInterface

logger = logging.getLogger(__name__)

class WebSocketTransport(TransportInterface):
    """
    WebSocket transport for distributed components (e.g., dashboard).
    """
    
    def __init__(self, uri: str = "ws://localhost:8000/ws"):
        if not _WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library is required for WebSocketTransport.")
        self.uri = uri
        self._connection = None
        
    async def connect(self):
        try:
            self._connection = await websockets.connect(self.uri)
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket at {self.uri}: {e}")
            
    async def _send_async(self, message: Any):
        if not self._connection:
            await self.connect()
        if self._connection:
            try:
                await self._connection.send(json.dumps(message))
            except Exception as e:
                logger.error(f"WebSocket send failed: {e}")
                self._connection = None
                
    def send(self, message: Any) -> None:
        """
        Synchronous wrapper for sending messages. 
        In a production pipeline, this should ideally be integrated with an async loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._send_async(message))
            else:
                loop.run_until_complete(self._send_async(message))
        except Exception as e:
            logger.error(f"Failed to schedule WebSocket send: {e}")
            
    def receive(self) -> Any:
        """
        Stub for receiving. In a real system, this would await self._connection.recv()
        """
        return None
