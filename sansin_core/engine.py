"""
Thompson Sampling + context-aware heuristic risk scoring engine.

Scores AI agent tool calls and learns from human corrections. The engine
maintains a Beta(alpha, beta) distribution per tool name. Alpha tracks
"this tool was safe" signals, beta tracks "this tool was risky" signals.

Heuristic scoring provides baseline risk assessment from tool name keywords
and context inspection. After MIN_OVERRIDES_FOR_THOMPSON overrides per tool,
Thompson Sampling blends learned priors with heuristics.

Extracted from sansin-ai/sansin/services/agent_decision.py and adapted
for local-only, stdlib-only operation.
"""

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Risk score thresholds (from agent_decision.py:23-33)
THRESHOLD_LOW = 0.3
THRESHOLD_HIGH = 0.7
THRESHOLD_BLOCK = 0.9

# Minimum override count before Thompson Sampling kicks in
MIN_OVERRIDES_FOR_THOMPSON = 10

# Blending weights: heuristic vs Thompson Sampling
HEURISTIC_WEIGHT = 0.6
THOMPSON_WEIGHT = 0.4

# Context-aware keyword classification.
# "write" and "update" are MEDIUM by default, escalate to HIGH
# only when combined with blast radius modifiers.
HIGH_RISK_KEYWORDS = {"delete", "remove", "drop", "send", "execute"}
MEDIUM_RISK_KEYWORDS = {"write", "update", "modify", "create", "post", "put", "patch"}
LOW_RISK_KEYWORDS = {"get", "read", "list", "query", "search", "fetch"}

# Keywords that escalate MEDIUM_RISK to HIGH_RISK when present in context
BLAST_RADIUS_ESCALATION_KEYWORDS = {"all", "every", "bulk", "batch", "mass"}


@dataclass
class Decision:
    """
    Result of a tool call evaluation.

    Attributes:
        id: Unique decision ID for override tracking.
        allow: True if the tool call should proceed.
        risk_score: 0.0-1.0 composite risk score.
        safe_probability: Thompson Sampling posterior mean (alpha / (alpha + beta)).
            0.5 at cold start. Approaches 1.0 as "safe" overrides accumulate.
        certainty: 1 - Beta variance. How sure the system is, regardless of direction.
        reason: Human-readable explanation of the decision.
        action: "allow", "deny", or "escalate".
        tool_name: The tool that was evaluated.
    """

    id: str
    allow: bool
    risk_score: float
    safe_probability: float
    certainty: float
    reason: str
    action: str
    tool_name: str


@dataclass
class Prior:
    """Thompson Sampling prior for a single tool."""

    tool_name: str
    alpha: float = 1.0
    beta: float = 1.0
    decision_count: int = 0
    override_count: int = 0


def compute_risk_heuristics(
    tool_name: str,
    context: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Compute risk score from context-aware heuristic rules.

    "write" and "update" start at MEDIUM_RISK (0.4) and only escalate
    to HIGH_RISK (0.7) when blast radius modifiers are present.

    Returns:
        Tuple of (risk_score clamped to [0.0, 1.0], reasoning string).
    """
    factors = []
    tool_lower = tool_name.lower()

    # Determine base risk level from tool name
    is_high = any(kw in tool_lower for kw in HIGH_RISK_KEYWORDS)
    is_medium = any(kw in tool_lower for kw in MEDIUM_RISK_KEYWORDS)
    is_low = any(kw in tool_lower for kw in LOW_RISK_KEYWORDS)

    # Check for blast radius indicators in context
    has_blast_radius = _check_blast_radius(context)

    if is_high:
        matched = [kw for kw in HIGH_RISK_KEYWORDS if kw in tool_lower]
        base_score = 0.7
        factors.append(f"High-risk keywords: {matched} -> base=0.7")
    elif is_medium and has_blast_radius:
        # Context-aware escalation: medium keyword + blast radius = high
        matched = [kw for kw in MEDIUM_RISK_KEYWORDS if kw in tool_lower]
        base_score = 0.7
        factors.append(f"Medium-risk keywords {matched} escalated to HIGH by blast radius")
    elif is_medium:
        matched = [kw for kw in MEDIUM_RISK_KEYWORDS if kw in tool_lower]
        base_score = 0.4
        factors.append(f"Medium-risk keywords: {matched} -> base=0.4")
    elif is_low:
        matched = [kw for kw in LOW_RISK_KEYWORDS if kw in tool_lower]
        base_score = 0.1
        factors.append(f"Low-risk keywords: {matched} -> base=0.1")
    else:
        base_score = 0.1
        factors.append(f"No recognized keywords in '{tool_name}' -> base=0.1")

    # Blast radius modifiers from context values
    modifiers = 0.0

    for key, value in context.items():
        if isinstance(value, str):
            value_lower = value.lower()
            if any(bk in value_lower for bk in BLAST_RADIUS_ESCALATION_KEYWORDS):
                modifiers += 0.2
                factors.append(f"Context '{key}' contains blast radius keyword -> +0.2")
                break

    for key, value in context.items():
        if isinstance(value, list) and len(value) > 10:
            modifiers += 0.1
            factors.append(f"Context '{key}' is a list with {len(value)} items (>10) -> +0.1")
            break

    # Specific context keys
    recipients = context.get("recipients", context.get("recipients_count", 0))
    if isinstance(recipients, (int, float)) and recipients > 100:
        modifiers += 0.2
        factors.append(f"recipients={recipients} (>100) -> +0.2")

    if context.get("irreversible", False):
        modifiers += 0.15
        factors.append("irreversible=True -> +0.15")

    final_score = max(0.0, min(1.0, base_score + modifiers))
    reasoning = f"Heuristic score={final_score:.3f}: " + ", ".join(factors)

    return final_score, reasoning


def _check_blast_radius(context: Dict[str, Any]) -> bool:
    """Check if context contains blast radius indicators."""
    for value in context.values():
        if isinstance(value, str):
            if any(bk in value.lower() for bk in BLAST_RADIUS_ESCALATION_KEYWORDS):
                return True
        if isinstance(value, list) and len(value) > 10:
            return True

    recipients = context.get("recipients", context.get("recipients_count", 0))
    if isinstance(recipients, (int, float)) and recipients > 100:
        return True

    return False


def score_to_decision(risk_score: float) -> Tuple[bool, str]:
    """
    Map a risk score to an allow/block decision and action string.

    Thresholds:
    - < 0.3  -> allow, action="allow"
    - [0.3, 0.7) -> allow, action="allow" (elevated but allowed)
    - [0.7, 0.9) -> block, action="escalate"
    - >= 0.9 -> block, action="deny"
    """
    if risk_score >= THRESHOLD_BLOCK:
        return False, "deny"
    elif risk_score >= THRESHOLD_HIGH:
        return False, "escalate"
    else:
        return True, "allow"


def compute_safe_probability(alpha: float, beta: float) -> float:
    """Posterior mean of the Beta distribution: alpha / (alpha + beta)."""
    total = alpha + beta
    if total == 0:
        return 0.5
    return alpha / total


def compute_certainty(alpha: float, beta: float) -> float:
    """
    1 - Beta variance. Higher = more certain.

    Beta variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    """
    total = alpha + beta
    if total == 0:
        return 0.0
    variance = (alpha * beta) / (total * total * (total + 1))
    return max(0.0, min(1.0, 1.0 - variance))


def evaluate(
    tool_name: str,
    context: Dict[str, Any],
    prior: Prior,
) -> Tuple[float, str]:
    """
    Evaluate risk score blending heuristics with Thompson Sampling.

    Returns:
        Tuple of (final_risk_score, reasoning_string).
    """
    heuristic_score, heuristic_reasoning = compute_risk_heuristics(tool_name, context)

    reasoning_parts = [heuristic_reasoning]

    if prior.override_count >= MIN_OVERRIDES_FOR_THOMPSON:
        thompson_sample = random.betavariate(
            max(1.0, prior.alpha), max(1.0, prior.beta)
        )
        # Higher sample = more likely safe = lower risk
        final_score = (
            HEURISTIC_WEIGHT * heuristic_score
            + THOMPSON_WEIGHT * (1 - thompson_sample)
        )
        final_score = max(0.0, min(1.0, final_score))
        reasoning_parts.append(
            f"Thompson Sampling blended: sample={thompson_sample:.3f}, "
            f"alpha={prior.alpha:.1f}, beta={prior.beta:.1f}, "
            f"overrides={prior.override_count}"
        )
    else:
        final_score = heuristic_score
        reasoning_parts.append(
            f"Heuristic-only (overrides={prior.override_count} < {MIN_OVERRIDES_FOR_THOMPSON})"
        )

    return final_score, "; ".join(reasoning_parts)


def make_decision(
    tool_name: str,
    context: Dict[str, Any],
    prior: Prior,
) -> Decision:
    """
    Create a full Decision object for a tool call.

    Computes risk score, maps to allow/deny, generates decision ID.
    Does NOT persist anything (that's the store's job).
    """
    risk_score, reasoning = evaluate(tool_name, context, prior)
    allow, action = score_to_decision(risk_score)

    return Decision(
        id=str(uuid.uuid4()),
        allow=allow,
        risk_score=risk_score,
        safe_probability=compute_safe_probability(prior.alpha, prior.beta),
        certainty=compute_certainty(prior.alpha, prior.beta),
        reason=reasoning,
        action=action,
        tool_name=tool_name,
    )


def args_hash(context: Dict[str, Any]) -> str:
    """SHA-256 hash of context dict for privacy-preserving storage."""
    raw = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()
