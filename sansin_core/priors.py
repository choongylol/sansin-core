"""
Prior export/import and bundled community priors.

Exports priors as a portable JSON file. Import merges priors by summing
alphas and betas (mathematically correct for Beta distributions).

Community priors are bundled with the package to solve cold start.
New installs start at ~20-override quality, not zero.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sansin_core.engine import Prior
from sansin_core.store import SCHEMA_VERSION, Store

# Bundled community priors. These represent reasonable defaults derived
# from common agent tool patterns. They provide a starting point so new
# installs don't start from pure heuristics.
#
# Each prior has alpha/beta set to reflect ~20 synthetic overrides worth
# of learning, enough to nudge Thompson Sampling but not enough to
# override strong local signals.
COMMUNITY_PRIORS = [
    Prior(tool_name="send_email", alpha=8.0, beta=14.0, decision_count=20, override_count=20),
    Prior(tool_name="send_message", alpha=10.0, beta=12.0, decision_count=20, override_count=20),
    Prior(tool_name="delete_file", alpha=3.0, beta=19.0, decision_count=20, override_count=20),
    Prior(tool_name="delete_record", alpha=4.0, beta=18.0, decision_count=20, override_count=20),
    Prior(tool_name="drop_table", alpha=1.5, beta=20.5, decision_count=20, override_count=20),
    Prior(tool_name="execute_command", alpha=5.0, beta=17.0, decision_count=20, override_count=20),
    Prior(tool_name="execute_sql", alpha=6.0, beta=16.0, decision_count=20, override_count=20),
    Prior(tool_name="query_database", alpha=18.0, beta=4.0, decision_count=20, override_count=20),
    Prior(tool_name="read_file", alpha=19.0, beta=3.0, decision_count=20, override_count=20),
    Prior(tool_name="search_web", alpha=19.0, beta=3.0, decision_count=20, override_count=20),
    Prior(tool_name="create_file", alpha=14.0, beta=8.0, decision_count=20, override_count=20),
    Prior(tool_name="write_file", alpha=10.0, beta=12.0, decision_count=20, override_count=20),
    Prior(tool_name="update_record", alpha=12.0, beta=10.0, decision_count=20, override_count=20),
    Prior(tool_name="list_files", alpha=19.0, beta=3.0, decision_count=20, override_count=20),
    Prior(tool_name="get_user", alpha=19.0, beta=3.0, decision_count=20, override_count=20),
]


def export_priors(store: Store, path: str, include_decisions: bool = False) -> None:
    """
    Export priors (and optionally recent decisions) to a JSON file.

    Args:
        store: The Store instance to export from.
        path: File path to write the JSON export.
        include_decisions: If True, include last 100 decisions.
    """
    priors = store.get_all_priors()

    export_data = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "priors": [
            {
                "tool_name": p.tool_name,
                "alpha": p.alpha,
                "beta": p.beta,
                "decision_count": p.decision_count,
                "override_count": p.override_count,
            }
            for p in priors
        ],
    }

    if include_decisions:
        export_data["decisions"] = store.get_decisions(limit=100)

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)


def import_priors(store: Store, path: str) -> int:
    """
    Import priors from a JSON file, merging with existing priors.

    Rejects files with schema_version > current.
    Auto-migrates files with schema_version < current.

    Returns the number of priors imported/updated.
    """
    with open(Path(path).expanduser()) as f:
        data = json.load(f)

    file_version = data.get("schema_version", 1)
    if file_version > SCHEMA_VERSION:
        raise ValueError(
            f"Export file schema version {file_version} is newer than "
            f"this version of sansin-core ({SCHEMA_VERSION}). "
            f"Please upgrade sansin-core."
        )

    priors = [
        Prior(
            tool_name=p["tool_name"],
            alpha=p["alpha"],
            beta=p["beta"],
            decision_count=p.get("decision_count", 0),
            override_count=p.get("override_count", 0),
        )
        for p in data.get("priors", [])
    ]

    return store.import_priors(priors)


def load_community_priors(store: Store) -> int:
    """
    Load bundled community priors into the store.

    Only imports priors for tools that don't already have local data.
    This is safe to call on every startup.

    Returns the number of community priors loaded.
    """
    existing_priors = {p.tool_name for p in store.get_all_priors()}
    new_priors = [p for p in COMMUNITY_PRIORS if p.tool_name not in existing_priors]

    if not new_priors:
        return 0

    return store.import_priors(new_priors)
