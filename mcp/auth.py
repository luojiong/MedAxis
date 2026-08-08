"""MCP authorization policy expressed as ordered permission levels."""
from enum import Enum


class PermissionLevel(Enum):
    READONLY = "readonly"       # Can only read resources and call read-only tools
    CONTROL = "control"        # Can call all tools
    ADMIN = "admin"            # Can manage server config


_PERMISSION_RANK = {
    PermissionLevel.READONLY: 0,
    PermissionLevel.CONTROL: 1,
    PermissionLevel.ADMIN: 2,
}


def check_permission(granted: PermissionLevel, required: PermissionLevel) -> bool:
    """Return whether a client permission level satisfies a tool requirement."""

    return _PERMISSION_RANK[granted] >= _PERMISSION_RANK[required]
