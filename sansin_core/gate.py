"""
SansinLocal: the main entry point for sansin-core.

Provides check(), override(), status(), export_priors(), import_priors(),
and demo(). All state is in local SQLite. No cloud dependency.
"""

import logging
from typing import Any, Dict, Optional

from sansin_core.engine import Decision, make_decision
from sansin_core.priors import export_priors as _export, import_priors as _import, load_community_priors
from sansin_core.store import Store

logger = logging.getLogger(__name__)


class SansinLocal:
    """
    Local-first AI agent safety gate.

    Scores tool calls using Thompson Sampling + context-aware heuristics.
    Learns from human corrections via override(). All state stored in
    local SQLite. Thread-safe.

    Args:
        db_path: Path to SQLite database file. Created if it doesn't exist.
        fail_closed: If True, block tool calls on internal errors.
            If False (default), allow tool calls on internal errors.
        load_community: If True (default), load bundled community priors
            on first run for tools that don't have local data yet.
    """

    def __init__(
        self,
        db_path: str = "~/.sansin/decisions.db",
        fail_closed: bool = False,
        load_community: bool = True,
    ):
        self._store = Store(db_path=db_path)
        self._fail_closed = fail_closed

        if load_community:
            loaded = load_community_priors(self._store)
            if loaded > 0:
                logger.info(f"Loaded {loaded} community priors for cold start")

    def check(
        self,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """
        Evaluate whether a tool call should be allowed.

        Args:
            tool_name: Name of the tool being called (e.g., "send_email").
            context: Freeform dict of context for risk scoring.

        Returns:
            Decision with allow, risk_score, safe_probability, certainty, reason.

        Raises:
            ValueError: If tool_name is None/empty or context is not a dict.
        """
        if not tool_name:
            raise ValueError("tool_name is required and cannot be empty")
        if context is not None and not isinstance(context, dict):
            raise ValueError(f"context must be a dict, got {type(context).__name__}")

        context = context or {}

        try:
            prior = self._store.get_prior(tool_name)
            decision = make_decision(tool_name, context, prior)
            self._store.save_decision(decision, context, source="sdk_check")
            return decision

        except Exception as e:
            logger.error(f"Error evaluating tool '{tool_name}': {e}", exc_info=True)

            if self._fail_closed:
                from sansin_core.engine import Decision
                return Decision(
                    id="",
                    allow=False,
                    risk_score=1.0,
                    safe_probability=0.5,
                    certainty=0.0,
                    reason=f"Internal error, failing closed: {e}",
                    action="deny",
                    tool_name=tool_name,
                )
            else:
                from sansin_core.engine import Decision
                return Decision(
                    id="",
                    allow=True,
                    risk_score=0.0,
                    safe_probability=0.5,
                    certainty=0.0,
                    reason=f"Internal error, failing open: {e}",
                    action="allow",
                    tool_name=tool_name,
                )

    def override(
        self,
        decision_id: str,
        correct_action: str,
        reason: str = "",
    ) -> bool:
        """
        Override a previous decision. Teaches the Thompson Sampling engine.

        Args:
            decision_id: The id from a Decision object.
            correct_action: "allow" or "block".
            reason: Human-provided reason for the override.

        Returns:
            True if the override was applied, False if decision not found.

        Raises:
            ValueError: If decision_id is empty or correct_action is invalid.
        """
        if not decision_id:
            raise ValueError("decision_id is required")
        if correct_action not in ("allow", "block"):
            raise ValueError(f"correct_action must be 'allow' or 'block', got '{correct_action}'")

        return self._store.save_override(decision_id, correct_action, reason)

    def status(self) -> Dict:
        """
        Get aggregate statistics and per-tool prior information.

        Returns:
            Dict with decisions_count, overrides_count, and tools
            (per-tool alpha, beta, safe_probability, certainty).
        """
        return self._store.get_stats()

    def export_priors(self, path: str, include_decisions: bool = False) -> None:
        """
        Export priors to a portable JSON file.

        Args:
            path: File path for the export.
            include_decisions: If True, include last 100 decisions.
        """
        _export(self._store, path, include_decisions=include_decisions)

    def import_priors(self, path: str) -> int:
        """
        Import priors from a JSON file, merging with existing.

        Returns the number of priors imported/updated.
        """
        return _import(self._store, path)

    def demo(self, decisions: int = 50) -> None:
        """
        Run an interactive demo showing the learning curve.

        Simulates tool calls with auto-overrides to demonstrate
        how Thompson Sampling improves over time.
        """
        from sansin_core.demo import run_demo
        run_demo(self, decisions)
