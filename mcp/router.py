"""
MCP (Model Context Protocol) Server for MedAxis.

Exposes MedAxis functionality as MCP Tools for AI Agent integration.
Supports stdio and SSE transports.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
import json
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from .auth import PermissionLevel, check_permission


@dataclass
class MCPTool:
    name: str
    description: str
    parameters_schema: dict          # JSON Schema
    handler: Callable                # async callable(params) -> result
    permission: PermissionLevel = PermissionLevel.CONTROL


@dataclass
class MCPResource:
    uri: str
    name: str
    description: str = ""
    mime_type: str = "application/json"
    handler: Optional[Callable] = None


@dataclass
class JSONRPCRequest:
    jsonrpc: str = "2.0"
    id: Any = None
    method: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class JSONRPCResponse:
    jsonrpc: str = "2.0"
    id: Any = None
    result: Any = None
    error: Optional[dict] = None


@dataclass(frozen=True)
class AuditEvent:
    """Minimal, non-sensitive record of an MCP operation."""

    timestamp: str
    operation: str
    required_permission: PermissionLevel
    succeeded: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "required_permission": self.required_permission.value,
            "succeeded": self.succeeded,
            "error": self.error,
        }


class MCPRouter:
    """Routes JSON-RPC requests to registered tools and resources."""

    def __init__(self, permission_level: PermissionLevel = PermissionLevel.CONTROL,
                 audit_capacity: int = 1_000):
        self.tools: dict[str, MCPTool] = {}
        self.resources: dict[str, MCPResource] = {}
        self.permission_level = permission_level
        self._audit_events: deque[AuditEvent] = deque(maxlen=max(1, audit_capacity))

    def register_tool(self, tool: MCPTool):
        self.tools[tool.name] = tool

    def register_resource(self, resource: MCPResource):
        self.resources[resource.uri] = resource

    def list_tools(self) -> list[dict]:
        return [{
            "name": t.name,
            "description": t.description,
            "inputSchema": t.parameters_schema,
        } for t in self.tools.values()]

    def list_resources(self) -> list[dict]:
        return [{
            "uri": r.uri,
            "name": r.name,
            "description": r.description,
            "mimeType": r.mime_type,
        } for r in self.resources.values()]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Tool not found: {name}")
        if not check_permission(self.permission_level, tool.permission):
            message = f"Permission denied for tool: {name}"
            self._record_audit(name, tool.permission, False, message)
            raise PermissionError(message)
        try:
            result = await tool.handler(arguments)
        except Exception as exc:
            self._record_audit(name, tool.permission, False, str(exc))
            raise
        self._record_audit(name, tool.permission, True)
        return result

    def audit_events(self) -> list[dict[str, Any]]:
        """Return a safe snapshot of recent tool invocations for operators."""

        return [event.to_dict() for event in self._audit_events]

    def _record_audit(self, operation: str, required_permission: PermissionLevel,
                      succeeded: bool, error: Optional[str] = None) -> None:
        self._audit_events.append(AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            required_permission=required_permission,
            succeeded=succeeded,
            error=error,
        ))

    async def read_resource(self, uri: str) -> str:
        resource = self.resources.get(uri)
        if resource is None:
            raise ValueError(f"Resource not found: {uri}")
        if resource.handler is None:
            raise ValueError(f"Resource has no handler: {uri}")
        result = await resource.handler()
        return result if isinstance(result, str) else json.dumps(result)

    async def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        try:
            if request.method == "tools/list":
                result = self.list_tools()
            elif request.method == "tools/call":
                result = await self.call_tool(request.params.get("name", ""), request.params.get("arguments", {}))
            elif request.method == "resources/list":
                result = self.list_resources()
            elif request.method == "resources/read":
                result = await self.read_resource(request.params.get("uri", ""))
            elif request.method == "initialize":
                result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "MedAxis", "version": "0.1.0"}, "capabilities": {"tools": {}, "resources": {}}}
            else:
                return JSONRPCResponse(id=request.id, error={"code": -32601, "message": f"Method not found: {request.method}"})
            return JSONRPCResponse(id=request.id, result=result)
        except Exception as e:
            return JSONRPCResponse(id=request.id, error={"code": -32000, "message": str(e)})

    async def handle_message(self, raw: str) -> str:
        try:
            req_data = json.loads(raw)
            req = JSONRPCRequest(**req_data) if isinstance(req_data, dict) else JSONRPCRequest()
            resp = await self.handle_request(req)
            return json.dumps({"jsonrpc": resp.jsonrpc, "id": resp.id, "result": resp.result} if resp.error is None else {"jsonrpc": resp.jsonrpc, "id": resp.id, "error": resp.error})
        except json.JSONDecodeError:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
