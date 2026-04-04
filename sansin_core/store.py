"""
SQLite WAL persistence layer for sansin-core.

Stores decisions and Thompson Sampling priors in a local SQLite database.
Thread-safe via WAL mode (concurrent readers + single writer). Zero
external dependencies (uses Python's built-in sqlite3 module).
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from sansin_core.engine import Decision, Prior, args_hash

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS priors (
    tool_name TEXT PRIMARY KEY,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    decision_count INTEGER NOT NULL DEFAULT 0,
    override_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    args_hash TEXT,
    risk_score REAL NOT NULL,
    safe_probability REAL NOT NULL,
    certainty REAL NOT NULL,
    action TEXT NOT NULL,
    reasoning TEXT,
    override TEXT,
    override_reason TEXT,
    source TEXT NOT NULL DEFAULT 'sdk_check',
    created_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_decisions_tool ON decisions(tool_name);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at);
"""


class Store:
    """SQLite-backed persistence for decisions and priors."""

    def __init__(self, db_path: str = "~/.sansin/decisions.db"):
        self._db_path = str(Path(db_path).expanduser())
        self._lock = threading.Lock()

        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection (one per call for thread safety)."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema and run migrations."""
        conn = self._get_conn()
        try:
            conn.executescript(_SCHEMA_SQL)

            # Check/set schema version
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            else:
                stored_version = int(row["value"])
                if stored_version > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Database schema version {stored_version} is newer than "
                        f"this version of sansin-core supports ({SCHEMA_VERSION}). "
                        f"Please upgrade sansin-core or use export_priors() to back up "
                        f"and create a fresh database."
                    )
                if stored_version < SCHEMA_VERSION:
                    self._migrate(conn, stored_version)

            conn.commit()
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run forward migrations. Add migration steps here as schema evolves."""
        # Currently at v1, no migrations needed yet.
        # Future: if from_version < 2: ... migrate to v2 ...
        conn.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION),),
        )

    def get_prior(self, tool_name: str) -> Prior:
        """Get or create a Thompson Sampling prior for a tool."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM priors WHERE tool_name = ?", (tool_name,)
            ).fetchone()

            if row:
                return Prior(
                    tool_name=row["tool_name"],
                    alpha=row["alpha"],
                    beta=row["beta"],
                    decision_count=row["decision_count"],
                    override_count=row["override_count"],
                )

            # Create with Beta(1,1) uniform prior
            now = _now_iso()
            conn.execute(
                "INSERT INTO priors (tool_name, alpha, beta, decision_count, override_count, created_at, updated_at) "
                "VALUES (?, 1.0, 1.0, 0, 0, ?, ?)",
                (tool_name, now, now),
            )
            conn.commit()

            return Prior(tool_name=tool_name)
        finally:
            conn.close()

    def save_decision(self, decision: Decision, context: Dict, source: str = "sdk_check") -> None:
        """Persist a decision record."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO decisions (id, tool_name, args_hash, risk_score, safe_probability, "
                "certainty, action, reasoning, source, created_at, schema_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.id,
                    decision.tool_name,
                    args_hash(context),
                    decision.risk_score,
                    decision.safe_probability,
                    decision.certainty,
                    decision.action,
                    decision.reason,
                    source,
                    _now_iso(),
                    SCHEMA_VERSION,
                ),
            )

            # Increment decision count on prior
            conn.execute(
                "UPDATE priors SET decision_count = decision_count + 1, updated_at = ? "
                "WHERE tool_name = ?",
                (_now_iso(), decision.tool_name),
            )

            conn.commit()
        finally:
            conn.close()

    def save_override(
        self, decision_id: str, correct_action: str, reason: str = ""
    ) -> bool:
        """
        Record a human override and update Thompson Sampling priors.

        Returns True if the override was applied, False if decision not found.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT tool_name FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()

            if not row:
                return False

            tool_name = row["tool_name"]

            # Update decision record
            conn.execute(
                "UPDATE decisions SET override = ?, override_reason = ? WHERE id = ?",
                (correct_action, reason, decision_id),
            )

            # Update Thompson Sampling prior
            if correct_action == "allow":
                conn.execute(
                    "UPDATE priors SET alpha = alpha + 1, override_count = override_count + 1, "
                    "updated_at = ? WHERE tool_name = ?",
                    (_now_iso(), tool_name),
                )
            elif correct_action == "block":
                conn.execute(
                    "UPDATE priors SET beta = beta + 1, override_count = override_count + 1, "
                    "updated_at = ? WHERE tool_name = ?",
                    (_now_iso(), tool_name),
                )

            conn.commit()
            return True
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) as c FROM decisions").fetchone()["c"]
            overrides = conn.execute(
                "SELECT COUNT(*) as c FROM decisions WHERE override IS NOT NULL"
            ).fetchone()["c"]

            tools = {}
            for row in conn.execute("SELECT * FROM priors"):
                from sansin_core.engine import compute_safe_probability, compute_certainty

                tools[row["tool_name"]] = {
                    "alpha": row["alpha"],
                    "beta": row["beta"],
                    "decision_count": row["decision_count"],
                    "override_count": row["override_count"],
                    "safe_probability": compute_safe_probability(row["alpha"], row["beta"]),
                    "certainty": compute_certainty(row["alpha"], row["beta"]),
                }

            return {
                "decisions_count": total,
                "overrides_count": overrides,
                "tools": tools,
            }
        finally:
            conn.close()

    def get_decisions(self, limit: int = 50, tool_filter: Optional[str] = None) -> List[Dict]:
        """Get recent decisions."""
        conn = self._get_conn()
        try:
            if tool_filter:
                rows = conn.execute(
                    "SELECT * FROM decisions WHERE tool_name = ? ORDER BY created_at DESC LIMIT ?",
                    (tool_filter, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_all_priors(self) -> List[Prior]:
        """Get all priors for export."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM priors").fetchall()
            return [
                Prior(
                    tool_name=row["tool_name"],
                    alpha=row["alpha"],
                    beta=row["beta"],
                    decision_count=row["decision_count"],
                    override_count=row["override_count"],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def import_priors(self, priors: List[Prior]) -> int:
        """
        Import priors by merging (summing alphas and betas).

        Returns the number of priors imported/updated.
        """
        conn = self._get_conn()
        imported = 0
        try:
            now = _now_iso()
            for p in priors:
                if p.alpha < 1.0 or p.beta < 1.0:
                    continue  # Skip invalid priors

                existing = conn.execute(
                    "SELECT * FROM priors WHERE tool_name = ?", (p.tool_name,)
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE priors SET alpha = alpha + ? - 1.0, beta = beta + ? - 1.0, "
                        "decision_count = decision_count + ?, override_count = override_count + ?, "
                        "updated_at = ? WHERE tool_name = ?",
                        (
                            p.alpha,
                            p.beta,
                            p.decision_count,
                            p.override_count,
                            now,
                            p.tool_name,
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO priors (tool_name, alpha, beta, decision_count, override_count, "
                        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (p.tool_name, p.alpha, p.beta, p.decision_count, p.override_count, now, now),
                    )
                imported += 1

            conn.commit()
            return imported
        finally:
            conn.close()


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
