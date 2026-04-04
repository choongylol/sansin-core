# sansin-core

A local-first safety engine for AI agents. It scores tool calls, learns from your corrections, and doesn't phone home.

Under the hood it's Thompson Sampling paired with keyword heuristics. It ships with pre-trained priors so you don't have to babysit it from scratch. Stdlib only, no API keys needed.

```
pip install sansin-core
```

## Quick start

```python
from sansin_core import SansinLocal

gate = SansinLocal()

# Score a tool call
decision = gate.check("send_email", {"recipients": 500, "irreversible": True})
print(decision.allow)        # False
print(decision.risk_score)   # 0.9
print(decision.reason)       # "deny"

# Override to teach it
gate.override(decision.id, "allow", "approved by admin")

# Next time it sees the same call, it remembers
decision = gate.check("send_email", {"recipients": 500})
# risk_score drops — the model updated its prior
```

## How it works

Each tool call gets a risk score between 0 and 1.

**Heuristics** do the initial classification. `read_file` starts low-risk, `delete_database` starts high. Context like `{"recipients": 500}` or `{"irreversible": True}` pushes the score up.

**Thompson Sampling** handles the learning. When you override a blocked call with "allow", the engine updates its Beta distribution for that tool. After roughly 10 overrides it starts blending learned behavior with the heuristics. Around 50 overrides it mostly trusts what it learned from you.

**Community priors** come bundled — 15 common tools at roughly 20-override quality each, so fresh installs have something to work with.

## MCP safety proxy

Works as a drop-in firewall between any MCP client and an upstream server. The agent on the other side has no idea Sansin is there.

```
pip install sansin-core[mcp]
sansin-mcp-proxy --upstream stdio://npx -y @modelcontextprotocol/server-filesystem /tmp
```

Point your client at the proxy instead of the real server. Every tool call gets scored before it's forwarded. High-risk ones get blocked with an explanation.

The proxy adds three management tools (prefix is configurable, defaults to `sansin_`):

- `sansin_override` — override a blocked decision
- `sansin_status` — engine stats and per-tool priors
- `sansin_decisions` — recent decisions for auditing

```
# Change the prefix if it collides with something
sansin-mcp-proxy --upstream stdio:///path/to/server --prefix safety_

# Fail-closed for regulated environments
sansin-mcp-proxy --upstream stdio:///path/to/server --fail-closed
```

## Interactive demo

There's a demo that simulates 50 decisions so you can watch the engine learn:

```
sansin-demo
```

It prints ASCII confidence bars as the Thompson Sampling updates in real time.

## API

### `SansinLocal(db_path, fail_closed, load_community)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `~/.sansin/decisions.db` | SQLite database path |
| `fail_closed` | `False` | Block on errors instead of allowing |
| `load_community` | `True` | Load community priors on first run |

### `gate.check(tool_name, context) -> Decision`

Scores a tool call. Returns immediately.

```python
decision = gate.check("delete_file", {"path": "/etc/passwd"})
decision.allow             # bool
decision.risk_score        # float, 0.0 to 1.0
decision.safe_probability  # P(safe) from Thompson Sampling
decision.certainty         # how confident the engine is
decision.reason            # "allow", "escalate", or "deny"
decision.id                # UUID, use this for overrides
```

### `gate.override(decision_id, correct_action, reason) -> bool`

Teach the engine. `correct_action` is either `"allow"` or `"block"`.

### `gate.status() -> dict`

Returns decision counts and per-tool priors (alpha, beta, override count).

### `gate.export_priors(path)` / `gate.import_priors(path)`

Export or import learned priors as JSON. Useful for sharing across machines or teams.

## Why these defaults

**Fail-open by default.** A safety layer that breaks your agent's workflow will just get uninstalled. If you need fail-closed, it's a flag away.

**Context-aware keywords.** "write" alone is medium risk. It only goes high when combined with blast-radius words like "all", "every", "bulk", "batch", "mass" in the context.

**SQLite in WAL mode.** Thread-safe, concurrent reads, single writer. Handles up to ~50 calls/sec which is more than enough for local agents.

**No required dependencies.** The core is pure stdlib. MCP proxy pulls in `mcp` as an optional extra.

## Requirements

- Python 3.10+
- No required dependencies (stdlib only)
- Optional: `mcp>=1.2.0` for the proxy

## License

MIT. See [LICENSE](LICENSE).

## Links

- [PyPI](https://pypi.org/project/sansin-core/)
- [Sansin Cloud](https://sansin.ai)
