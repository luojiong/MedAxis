"""Regression tests for the MCP JSON-RPC boundary and permissions."""

from __future__ import annotations

import asyncio
import json

from mcp.auth import PermissionLevel
from mcp.router import MCPRouter, MCPTool


def test_json_rpc_tool_call_round_trip() -> None:
    async def echo(arguments: dict) -> dict:
        return {"echo": arguments["message"]}

    router = MCPRouter()
    router.register_tool(MCPTool(
        name="echo",
        description="Echo an argument for regression testing.",
        parameters_schema={"type": "object"},
        handler=echo,
    ))

    response = asyncio.run(router.handle_message(json.dumps({
        "jsonrpc": "2.0",
        "id": "request-1",
        "method": "tools/call",
        "params": {"name": "echo", "arguments": {"message": "ok"}},
    })))

    assert json.loads(response) == {
        "jsonrpc": "2.0",
        "id": "request-1",
        "result": {"echo": "ok"},
    }


def test_readonly_router_rejects_control_tool() -> None:
    async def mutate(_arguments: dict) -> dict:
        return {"changed": True}

    router = MCPRouter(permission_level=PermissionLevel.READONLY)
    router.register_tool(MCPTool(
        name="mutate_workspace",
        description="A control operation.",
        parameters_schema={"type": "object"},
        handler=mutate,
        permission=PermissionLevel.CONTROL,
    ))

    response = asyncio.run(router.handle_message(json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "mutate_workspace", "arguments": {}},
    })))

    payload = json.loads(response)
    assert payload["error"]["code"] == -32000
    assert "Permission denied" in payload["error"]["message"]
    assert router.audit_events()[-1]["succeeded"] is False


def test_readonly_router_allows_explicitly_readonly_custom_tool() -> None:
    async def inspect_workspace(_arguments: dict) -> dict:
        return {"status": "ready"}

    router = MCPRouter(permission_level=PermissionLevel.READONLY)
    router.register_tool(MCPTool(
        name="inspect_workspace",
        description="A custom read-only operation.",
        parameters_schema={"type": "object"},
        handler=inspect_workspace,
        permission=PermissionLevel.READONLY,
    ))

    assert asyncio.run(router.call_tool("inspect_workspace", {})) == {"status": "ready"}
    audit_event = router.audit_events()[-1]
    assert audit_event["operation"] == "inspect_workspace"
    assert audit_event["succeeded"] is True
