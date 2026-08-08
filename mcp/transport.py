"""
MCP Transport abstraction and WS support.
"""
from enum import Enum


class TransportType(Enum):
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


# Re-export for convenience
