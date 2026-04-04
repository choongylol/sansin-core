"""Tests for the Sansin MCP proxy and CLI parsing."""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the mcp SDK before any sansin_core.mcp imports touch it.
for mod in ("mcp", "mcp.client", "mcp.client.stdio", "mcp.server", "mcp.server.fastmcp"):
    sys.modules.setdefault(mod, MagicMock())

from sansin_core.engine import Decision
from sansin_core.mcp.cli import _parse_upstream


# ── CLI argument parsing ─────────────────────────────────────────────


class TestParseUpstream:
    def test_stdio_uri_with_args(self):
        cmd, args = _parse_upstream("stdio://npx -y @mcp/server-fs /tmp")
        assert cmd == "npx"
        assert args == ["-y", "@mcp/server-fs", "/tmp"]

    def test_stdio_uri_bare_path(self):
        cmd, args = _parse_upstream("stdio:///usr/local/bin/my-server")
        assert cmd == "/usr/local/bin/my-server"
        assert args == []

    def test_stdio_uri_empty_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="nothing after stdio://"):
            _parse_upstream("stdio://")

    def test_bare_python_file(self):
        cmd, args = _parse_upstream("/path/to/server.py")
        assert cmd == sys.executable
        assert args == ["/path/to/server.py"]

    def test_bare_js_file(self):
        cmd, args = _parse_upstream("/path/to/server.js")
        assert cmd == "node"
        assert args == ["/path/to/server.js"]

    def test_bare_mjs_file(self):
        cmd, args = _parse_upstream("server.mjs --port 3000")
        assert cmd == "node"
        assert args == ["server.mjs", "--port", "3000"]

    def test_bare_binary(self):
        cmd, args = _parse_upstream("/usr/bin/my-server --verbose")
        assert cmd == "/usr/bin/my-server"
        assert args == ["--verbose"]

    def test_python_with_extra_args(self):
        cmd, args = _parse_upstream("my_server.py --host 0.0.0.0")
        assert cmd == sys.executable
        assert args == ["my_server.py", "--host", "0.0.0.0"]


# ── Helpers for proxy tests ──────────────────────────────────────────


@dataclass
class FakeTool:
    name: str
    description: str = ""


@dataclass
class FakeListResult:
    tools: List[FakeTool]


@pytest.fixture
def proxy(tmp_path):
    """Create a SansinProxy with mocked FastMCP."""
    with patch("sansin_core.mcp.proxy.FastMCP"):
        from sansin_core.mcp.proxy import SansinProxy
        return SansinProxy(
            upstream_command="echo",
            upstream_args=["hello"],
            db_path=str(tmp_path / "test.db"),
        )


# ── Tool collision detection ─────────────────────────────────────────


class TestToolCollision:
    def test_colliding_tool_is_skipped(self, proxy):
        proxy._session = AsyncMock()
        proxy._session.list_tools.return_value = FakeListResult(
            tools=[FakeTool(name="sansin_override"), FakeTool(name="safe_tool")]
        )
        proxy._register_proxied_tool = MagicMock()

        asyncio.run(proxy._discover_and_register_upstream_tools())

        assert "sansin_override" not in proxy._upstream_tools
        assert "safe_tool" in proxy._upstream_tools
        proxy._register_proxied_tool.assert_called_once()

    def test_no_collision_registers_all(self, proxy):
        proxy._session = AsyncMock()
        proxy._session.list_tools.return_value = FakeListResult(
            tools=[FakeTool(name="read_file"), FakeTool(name="write_file")]
        )
        proxy._register_proxied_tool = MagicMock()

        asyncio.run(proxy._discover_and_register_upstream_tools())

        assert len(proxy._upstream_tools) == 2
        assert proxy._register_proxied_tool.call_count == 2


# ── Scoring integration ─────────────────────────────────────────────


class TestScoringIntegration:
    def test_high_risk_call_is_blocked(self, proxy):
        proxy._session = AsyncMock()
        result = asyncio.run(
            proxy._handle_proxied_call(
                "send_email", {"recipients": 500, "irreversible": True}
            )
        )
        assert "[SANSIN BLOCKED]" in result
        assert "send_email" in result
        proxy._session.call_tool.assert_not_called()

    def test_low_risk_call_is_forwarded(self, proxy):
        upstream_content = MagicMock(text="file contents here")
        proxy._session = AsyncMock()
        proxy._session.call_tool.return_value = MagicMock(content=[upstream_content])

        result = asyncio.run(
            proxy._handle_proxied_call("read_file", {"path": "/tmp/x"})
        )
        assert "[SANSIN BLOCKED]" not in result
        assert "file contents here" in result
        proxy._session.call_tool.assert_called_once()

    def test_fail_closed_blocks_on_scoring_error(self, tmp_path):
        with patch("sansin_core.mcp.proxy.FastMCP"):
            from sansin_core.mcp.proxy import SansinProxy
            p = SansinProxy(
                upstream_command="echo",
                upstream_args=[],
                db_path=str(tmp_path / "test.db"),
                fail_closed=True,
            )
            p._session = AsyncMock()
            p._gate.check = MagicMock(side_effect=RuntimeError("boom"))

            result = asyncio.run(p._handle_proxied_call("any_tool", {}))
            assert "[SANSIN BLOCKED]" in result
            assert "fail-closed" in result
            p._session.call_tool.assert_not_called()


# ── Override flow ────────────────────────────────────────────────────


class TestOverrideFlow:
    @pytest.fixture
    def gate(self, tmp_path):
        from sansin_core.gate import SansinLocal
        return SansinLocal(db_path=str(tmp_path / "test.db"), load_community=False)

    def test_override_updates_priors(self, gate):
        decision = gate.check("send_email", {"recipients": 500, "irreversible": True})
        assert decision.allow is False

        ok = gate.override(decision.id, "allow", "approved by admin")
        assert ok is True

        stats = gate.status()
        tools = stats.get("tools", {})
        assert "send_email" in tools
        assert tools["send_email"]["override_count"] >= 1

    def test_override_invalid_action_raises(self, gate):
        with pytest.raises(ValueError, match="correct_action must be"):
            gate.override("some-id", "maybe")


# ── Status and decisions tools ───────────────────────────────────────


class TestStatusAndDecisions:
    @pytest.fixture
    def gate(self, tmp_path):
        from sansin_core.gate import SansinLocal
        return SansinLocal(db_path=str(tmp_path / "test.db"), load_community=False)

    def test_status_returns_counts(self, gate):
        gate.check("read_file", {})
        gate.check("send_email", {"recipients": 5})

        stats = gate.status()
        assert "decisions_count" in stats
        assert stats["decisions_count"] >= 2

    def test_status_includes_tool_priors(self, gate):
        gate.check("read_file", {})
        stats = gate.status()
        assert "tools" in stats
        assert "read_file" in stats["tools"]

    def test_status_is_json_serializable(self, gate):
        gate.check("read_file", {})
        stats = gate.status()
        output = json.dumps(stats, indent=2, default=str)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert parsed["decisions_count"] >= 1
