"""Sansin MCP proxy — transparent safety firewall for MCP servers.

The MCP proxy requires the `mcp` package: pip install sansin-core[mcp]
"""


def __getattr__(name):
    if name == "SansinProxy":
        from sansin_core.mcp.proxy import SansinProxy
        return SansinProxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SansinProxy"]
