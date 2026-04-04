"""
MCP Safety Proxy — transparent firewall between an MCP client and upstream server.

Architecture:  Agent <-> SansinProxy (MCP server) <-> Upstream MCP server (client)

Usage::

    proxy = SansinProxy(
        upstream_command="npx",
        upstream_args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )
    proxy.run()
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

from sansin_core.gate import SansinLocal

# Log to stderr — stdout is reserved for stdio MCP transport.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("sansin.mcp.proxy")


class SansinProxy:
    """MCP safety proxy that intercepts and scores tool calls to an upstream server.

    Connects to a single upstream MCP server via stdio, discovers its tools,
    and re-exposes them with safety scoring. Blocked calls return an explanation.
    Also exposes Sansin management tools (override, status, decisions) under
    a configurable prefix.
    """

    def __init__(
        self,
        upstream_command: str,
        upstream_args: Optional[List[str]] = None,
        db_path: str = "~/.sansin/decisions.db",
        prefix: str = "sansin_",
        fail_closed: bool = False,
    ) -> None:
        self._upstream_command = upstream_command
        self._upstream_args = upstream_args or []
        self._db_path = db_path
        self._prefix = prefix
        self._fail_closed = fail_closed

        self._gate = SansinLocal(db_path=db_path, fail_closed=fail_closed)
        self._session: Optional[ClientSession] = None
        self._upstream_tools: Dict[str, Any] = {}

        self._mcp = FastMCP(
            name="sansin-proxy",
            instructions=(
                "This server proxies tools from an upstream MCP server. "
                "All tool calls are evaluated by Sansin for safety before execution."
            ),
        )
        self._register_sansin_tools()

    # ------------------------------------------------------------------
    # Sansin management tools
    # ------------------------------------------------------------------

    def _register_sansin_tools(self) -> None:
        """Register Sansin's own management tools on the proxy server."""
        mcp = self._mcp
        prefix = self._prefix

        @mcp.tool(name=f"{prefix}override")
        def sansin_override(decision_id: str, correct_action: str, reason: str = "") -> str:
            """Override a previous decision (allow/block). Teaches the safety engine."""
            try:
                ok = self._gate.override(decision_id, correct_action, reason)
                if ok:
                    return f"Override applied: decision {decision_id} -> {correct_action}"
                return f"Override failed: decision {decision_id} not found"
            except ValueError as exc:
                return f"Override error: {exc}"

        @mcp.tool(name=f"{prefix}status")
        def sansin_status() -> str:
            """Get Sansin safety engine statistics and per-tool priors."""
            stats = self._gate.status()
            return json.dumps(stats, indent=2, default=str)

        @mcp.tool(name=f"{prefix}decisions")
        def sansin_decisions() -> str:
            """List recent Sansin decisions for auditing."""
            stats = self._gate.status()
            return json.dumps(stats, indent=2, default=str)

    # ------------------------------------------------------------------
    # Upstream connection and tool discovery
    # ------------------------------------------------------------------

    async def _discover_and_register_upstream_tools(self) -> None:
        """Discover tools from the upstream server and register proxied versions."""
        assert self._session is not None, "upstream session not connected"

        result = await self._session.list_tools()
        sansin_tool_names = {
            f"{self._prefix}override",
            f"{self._prefix}status",
            f"{self._prefix}decisions",
        }

        for tool in result.tools:
            if tool.name in sansin_tool_names:
                logger.warning(
                    "Tool name collision: upstream tool '%s' conflicts with "
                    "Sansin tool (prefix='%s'). Upstream tool will be SKIPPED. "
                    "Change the prefix to resolve this.",
                    tool.name,
                    self._prefix,
                )
                continue

            self._upstream_tools[tool.name] = tool
            self._register_proxied_tool(tool)

        logger.info(
            "Discovered %d upstream tools, registered %d (skipped %d collisions)",
            len(result.tools),
            len(self._upstream_tools),
            len(result.tools) - len(self._upstream_tools),
        )

    def _register_proxied_tool(self, tool: Any) -> None:
        """Create a proxied handler for a single upstream tool."""
        tool_name = tool.name
        proxy = self

        @self._mcp.tool(name=tool_name, description=tool.description or "")
        async def _proxied_call(**kwargs: Any) -> str:
            return await proxy._handle_proxied_call(tool_name, kwargs)

    async def _handle_proxied_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Score a tool call and either forward it or return a block response."""
        logger.info("Incoming call: tool=%s", tool_name)

        try:
            decision = self._gate.check(tool_name=tool_name, context=args)
        except Exception:
            logger.exception("Scoring failed for tool '%s'", tool_name)
            if self._fail_closed:
                return (
                    f"[SANSIN BLOCKED] Tool '{tool_name}' blocked: "
                    "scoring engine error (fail-closed mode)"
                )
            # Fail open — forward the call.
            decision = None

        if decision is not None and not decision.allow:
            logger.warning(
                "BLOCKED tool=%s risk=%.3f reason=%s decision_id=%s",
                tool_name,
                decision.risk_score,
                decision.reason,
                decision.id,
            )
            return (
                f"[SANSIN BLOCKED] Tool '{tool_name}' was blocked.\n"
                f"  Risk score: {decision.risk_score:.3f}\n"
                f"  Reason: {decision.reason}\n"
                f"  Decision ID: {decision.id}\n"
                f"  To override: call {self._prefix}override with this decision_id"
            )

        if decision is not None:
            logger.info(
                "ALLOWED tool=%s risk=%.3f decision_id=%s",
                tool_name,
                decision.risk_score,
                decision.id,
            )

        # Forward the call to the upstream server.
        try:
            assert self._session is not None, "upstream session not connected"
            result = await self._session.call_tool(tool_name, args)
        except Exception:
            logger.exception("Upstream call failed for tool '%s'", tool_name)
            return f"[SANSIN ERROR] Upstream call to '{tool_name}' failed."

        # Extract text from the upstream result.
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _connect_upstream(self) -> AsyncIterator[ClientSession]:
        """Connect to the upstream MCP server via stdio transport."""
        server_params = StdioServerParameters(
            command=self._upstream_command,
            args=self._upstream_args,
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def run_async(self) -> None:
        """Start the proxy: connect upstream, discover tools, serve clients."""
        logger.info(
            "Starting Sansin MCP proxy -> upstream: %s %s",
            self._upstream_command,
            " ".join(self._upstream_args),
        )
        async with self._connect_upstream() as session:
            self._session = session
            await self._discover_and_register_upstream_tools()
            logger.info("Proxy ready — serving on stdio")
            await self._mcp.run_async(transport="stdio")

    def run(self) -> None:
        """Blocking entry point. Runs the async event loop."""
        import asyncio

        asyncio.run(self.run_async())
