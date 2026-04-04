"""
Interactive demo showing Thompson Sampling learning in real time.

Simulates 50 agent tool calls across 3 tool categories with sprinkled
overrides. Prints ASCII confidence bars that update as the engine learns.

Run via: sansin-demo
Or: gate.demo(decisions=50)
"""

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sansin_core.gate import SansinLocal

# Demo tool scenarios
SCENARIOS = [
    ("send_email", {"recipients": 500, "irreversible": True}, True),   # risky
    ("send_email", {"recipients": 1}, False),                           # safe
    ("query_database", {"sql": "SELECT * FROM users"}, False),          # safe
    ("delete_file", {"path": "/data/export.csv", "irreversible": True}, True),  # risky
    ("delete_file", {"path": "/tmp/cache.txt"}, False),                 # borderline
    ("query_database", {"sql": "SELECT count(*) FROM orders"}, False),  # safe
    ("send_email", {"recipients": 50}, False),                          # moderate
    ("delete_file", {"path": "/var/log/old.log"}, False),               # borderline
    ("query_database", {"sql": "DROP TABLE users"}, True),              # risky (but query tool)
    ("send_email", {"recipients": 1, "subject": "test"}, False),        # safe
]

# Override schedule: which decisions to override and with what
# (decision_index, correct_action) — weighted toward first half
OVERRIDE_SCHEDULE = {
    1: "allow",    # send_email to 1 person was blocked? allow it
    2: "allow",    # query was blocked? allow it
    5: "allow",    # query was blocked? allow it
    8: "allow",    # delete tmp was blocked? allow it
    10: "allow",   # send to 1 was blocked? allow
    12: "allow",   # query
    15: "allow",   # another safe send
    18: "block",   # delete important file — block it
    20: "allow",   # safe query
    22: "allow",   # safe send
    25: "block",   # mass send — block
    30: "allow",   # safe query
    35: "allow",   # safe operation
    40: "block",   # risky delete
    45: "allow",   # safe query
}


def _confidence_bar(safe_prob: float, certainty: float, width: int = 30) -> str:
    """Render an ASCII confidence bar."""
    filled = int(safe_prob * width)
    empty = width - filled
    cert_indicator = "!" if certainty > 0.8 else "~" if certainty > 0.5 else "?"
    return f"[{'=' * filled}{' ' * empty}] {safe_prob:.0%} {cert_indicator}"


def _print_header():
    """Print the demo header."""
    print()
    print("=" * 60)
    print("  SANSIN DEMO: Watch Your Agent Learn")
    print("=" * 60)
    print()
    print("  Simulating agent tool calls with human corrections.")
    print("  Watch the confidence bars improve as the engine learns.")
    print()
    print("  Legend: [====      ] 40% ?  (? = uncertain, ~ = learning, ! = confident)")
    print()


def _print_decision(i: int, decision, overridden: bool, override_action: str = ""):
    """Print a single decision."""
    status = "ALLOW" if decision.allow else "BLOCK"
    color_start = "\033[32m" if decision.allow else "\033[31m"
    color_end = "\033[0m"

    line = f"  #{i:3d} {decision.tool_name:20s} {color_start}{status:5s}{color_end} risk={decision.risk_score:.2f}"

    if overridden:
        override_color = "\033[33m"
        line += f"  {override_color}OVERRIDE -> {override_action}{color_end}"

    print(line)


def _print_priors(gate: "SansinLocal"):
    """Print current prior status for all tools."""
    stats = gate.status()
    tools = stats.get("tools", {})

    if not tools:
        return

    print()
    print("  --- Learning Status ---")
    for tool_name in sorted(tools.keys()):
        info = tools[tool_name]
        bar = _confidence_bar(info["safe_probability"], info["certainty"])
        overrides = info["override_count"]
        print(f"  {tool_name:20s} {bar}  ({overrides} overrides)")
    print()


def run_demo(gate: "SansinLocal", decisions: int = 50):
    """Run the interactive learning demo."""
    _print_header()

    decision_ids = []

    for i in range(1, decisions + 1):
        scenario = SCENARIOS[(i - 1) % len(SCENARIOS)]
        tool_name, context, _is_risky = scenario

        decision = gate.check(tool_name=tool_name, context=context)
        decision_ids.append(decision.id)

        overridden = i in OVERRIDE_SCHEDULE
        override_action = OVERRIDE_SCHEDULE.get(i, "")

        _print_decision(i, decision, overridden, override_action)

        if overridden and decision.id:
            gate.override(
                decision_id=decision.id,
                correct_action=override_action,
                reason=f"Demo override #{i}",
            )

        # Show learning status every 10 decisions
        if i % 10 == 0:
            _print_priors(gate)
            time.sleep(0.5)
        else:
            time.sleep(0.1)

    # Final summary
    stats = gate.status()
    print()
    print("=" * 60)
    print(f"  DONE: {stats['decisions_count']} decisions, {stats['overrides_count']} overrides")
    print()
    print("  The engine learned from your corrections.")
    print("  Next time, it will make better decisions automatically.")
    print()
    print("  Try it yourself:")
    print("    from sansin_core import SansinLocal")
    print("    gate = SansinLocal()")
    print('    decision = gate.check(tool_name="send_email", context={"recipients": 500})')
    print("=" * 60)
    print()


def main():
    """CLI entry point for sansin-demo."""
    import tempfile
    import os

    from sansin_core.gate import SansinLocal

    # Use a temp DB for the demo so it doesn't pollute the user's real data
    demo_db = os.path.join(tempfile.gettempdir(), "sansin-demo.db")

    # Clean previous demo data
    if os.path.exists(demo_db):
        os.remove(demo_db)

    gate = SansinLocal(db_path=demo_db, load_community=False)

    try:
        decisions = 50
        if len(sys.argv) > 1:
            try:
                decisions = int(sys.argv[1])
            except ValueError:
                pass

        run_demo(gate, decisions)
    finally:
        # Clean up demo DB
        try:
            os.remove(demo_db)
        except OSError:
            pass


if __name__ == "__main__":
    main()
