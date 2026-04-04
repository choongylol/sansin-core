"""
sansin-core: Local-first AI agent safety engine.

Score every tool call. Learn from corrections. Zero cloud dependency.

Usage:
    from sansin_core import SansinLocal

    gate = SansinLocal()
    decision = gate.check(tool_name="send_email", context={"recipients": 500})

    if decision.allow:
        send_email(**args)
    else:
        print(f"Blocked: {decision.reason}")

    # Teach the engine when it's wrong
    gate.override(decision_id=decision.id, correct_action="allow", reason="Verified list")
"""

from sansin_core.engine import Decision, Prior
from sansin_core.gate import SansinLocal

__all__ = ["SansinLocal", "Decision", "Prior"]
__version__ = "0.1.0"
