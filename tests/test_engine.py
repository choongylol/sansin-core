"""Tests for the Thompson Sampling + heuristic scoring engine."""

import pytest
from sansin_core.engine import (
    Decision,
    Prior,
    compute_certainty,
    compute_risk_heuristics,
    compute_safe_probability,
    evaluate,
    make_decision,
    score_to_decision,
    args_hash,
)


class TestHeuristicScoring:
    """Tests for compute_risk_heuristics()."""

    def test_high_risk_keyword_delete(self):
        score, _ = compute_risk_heuristics("delete_file", {})
        assert score == pytest.approx(0.7, abs=0.01)

    def test_high_risk_keyword_send(self):
        score, _ = compute_risk_heuristics("send_email", {})
        assert score == pytest.approx(0.7, abs=0.01)

    def test_low_risk_keyword_read(self):
        score, _ = compute_risk_heuristics("read_file", {})
        assert score == pytest.approx(0.1, abs=0.01)

    def test_low_risk_keyword_query(self):
        score, _ = compute_risk_heuristics("query_database", {})
        assert score == pytest.approx(0.1, abs=0.01)

    def test_medium_risk_write_without_blast_radius(self):
        """write should be MEDIUM (0.4) without blast radius indicators."""
        score, _ = compute_risk_heuristics("write_file", {})
        assert score == pytest.approx(0.4, abs=0.01)

    def test_medium_risk_update_without_blast_radius(self):
        """update should be MEDIUM (0.4) without blast radius indicators."""
        score, _ = compute_risk_heuristics("update_record", {})
        assert score == pytest.approx(0.4, abs=0.01)

    def test_write_escalates_with_blast_radius(self):
        """write + blast radius keywords should escalate to HIGH (0.7)."""
        score, reason = compute_risk_heuristics("write_file", {"target": "all users"})
        assert score >= 0.7
        assert "escalated" in reason.lower() or "blast" in reason.lower()

    def test_update_escalates_with_large_list(self):
        """update + large list should escalate."""
        score, _ = compute_risk_heuristics("update_records", {"ids": list(range(50))})
        assert score > 0.4  # At least medium + modifier

    def test_unknown_tool_defaults_low(self):
        score, _ = compute_risk_heuristics("custom_tool", {})
        assert score == pytest.approx(0.1, abs=0.01)

    def test_recipients_modifier(self):
        score_low, _ = compute_risk_heuristics("send_email", {"recipients": 1})
        score_high, _ = compute_risk_heuristics("send_email", {"recipients": 500})
        assert score_high > score_low

    def test_irreversible_modifier(self):
        score_normal, _ = compute_risk_heuristics("delete_file", {})
        score_irreversible, _ = compute_risk_heuristics(
            "delete_file", {"irreversible": True}
        )
        assert score_irreversible > score_normal

    def test_multiple_modifiers_stack(self):
        score, _ = compute_risk_heuristics(
            "send_email",
            {"recipients": 500, "irreversible": True, "target": "all"},
        )
        assert score > 0.9

    def test_score_clamped_to_1(self):
        score, _ = compute_risk_heuristics(
            "delete_file",
            {"recipients": 1000, "irreversible": True, "target": "all users", "ids": list(range(100))},
        )
        assert score <= 1.0


class TestScoreToDecision:
    def test_low_risk_allows(self):
        allow, action = score_to_decision(0.1)
        assert allow is True
        assert action == "allow"

    def test_medium_risk_allows(self):
        allow, action = score_to_decision(0.5)
        assert allow is True
        assert action == "allow"

    def test_high_risk_escalates(self):
        allow, action = score_to_decision(0.7)
        assert allow is False
        assert action == "escalate"

    def test_very_high_risk_denies(self):
        allow, action = score_to_decision(0.95)
        assert allow is False
        assert action == "deny"


class TestThompsonSampling:
    def test_heuristic_only_below_threshold(self):
        prior = Prior(tool_name="test", override_count=5)
        score, reasoning = evaluate("send_email", {}, prior)
        assert "Heuristic-only" in reasoning

    def test_thompson_blends_above_threshold(self):
        prior = Prior(tool_name="test", alpha=15.0, beta=5.0, override_count=15)
        score, reasoning = evaluate("send_email", {}, prior)
        assert "Thompson Sampling blended" in reasoning

    def test_safe_tool_with_safe_prior_lowers_score(self):
        """A tool with many 'allow' overrides should get a lower risk score."""
        prior_cold = Prior(tool_name="test", override_count=0)
        prior_safe = Prior(tool_name="test", alpha=50.0, beta=5.0, override_count=50)

        # Run multiple times and average (Thompson Sampling is stochastic)
        scores_cold = [evaluate("send_email", {}, prior_cold)[0] for _ in range(100)]
        scores_safe = [evaluate("send_email", {}, prior_safe)[0] for _ in range(100)]

        avg_cold = sum(scores_cold) / len(scores_cold)
        avg_safe = sum(scores_safe) / len(scores_safe)

        assert avg_safe < avg_cold


class TestSafeProbabilityAndCertainty:
    def test_uniform_prior_is_half(self):
        assert compute_safe_probability(1.0, 1.0) == pytest.approx(0.5)

    def test_safe_prior(self):
        assert compute_safe_probability(50.0, 5.0) > 0.8

    def test_risky_prior(self):
        assert compute_safe_probability(5.0, 50.0) < 0.2

    def test_certainty_increases_with_data(self):
        cert_cold = compute_certainty(1.0, 1.0)
        cert_warm = compute_certainty(50.0, 50.0)
        assert cert_warm > cert_cold

    def test_certainty_bounds(self):
        assert 0.0 <= compute_certainty(1.0, 1.0) <= 1.0
        assert 0.0 <= compute_certainty(100.0, 100.0) <= 1.0


class TestMakeDecision:
    def test_returns_decision(self):
        prior = Prior(tool_name="test")
        d = make_decision("read_file", {}, prior)
        assert isinstance(d, Decision)
        assert d.allow is True
        assert d.tool_name == "read_file"
        assert d.id  # non-empty UUID

    def test_high_risk_tool_blocked(self):
        prior = Prior(tool_name="test")
        d = make_decision("send_email", {"recipients": 500, "irreversible": True}, prior)
        assert d.allow is False


class TestArgsHash:
    def test_deterministic(self):
        h1 = args_hash({"a": 1, "b": 2})
        h2 = args_hash({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys=True

    def test_different_args_different_hash(self):
        h1 = args_hash({"a": 1})
        h2 = args_hash({"a": 2})
        assert h1 != h2
