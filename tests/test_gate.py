"""Tests for SansinLocal (the main gate class)."""

import os
import tempfile
import threading

import pytest
from sansin_core import SansinLocal, Decision


@pytest.fixture
def gate(tmp_path):
    """Create a SansinLocal with a temporary database."""
    db = str(tmp_path / "test.db")
    return SansinLocal(db_path=db, load_community=False)


@pytest.fixture
def gate_with_community(tmp_path):
    """Create a SansinLocal with community priors loaded."""
    db = str(tmp_path / "test-community.db")
    return SansinLocal(db_path=db, load_community=True)


class TestCheck:
    def test_low_risk_allows(self, gate):
        d = gate.check(tool_name="read_file", context={})
        assert d.allow is True
        assert d.risk_score < 0.3

    def test_high_risk_blocks(self, gate):
        d = gate.check(tool_name="send_email", context={"recipients": 500, "irreversible": True})
        assert d.allow is False
        assert d.risk_score > 0.7

    def test_returns_decision_type(self, gate):
        d = gate.check(tool_name="read_file")
        assert isinstance(d, Decision)

    def test_none_tool_name_raises(self, gate):
        with pytest.raises(ValueError, match="tool_name is required"):
            gate.check(tool_name=None)

    def test_empty_tool_name_raises(self, gate):
        with pytest.raises(ValueError, match="tool_name is required"):
            gate.check(tool_name="")

    def test_non_dict_context_raises(self, gate):
        with pytest.raises(ValueError, match="context must be a dict"):
            gate.check(tool_name="test", context="not a dict")

    def test_none_context_defaults_to_empty(self, gate):
        d = gate.check(tool_name="read_file", context=None)
        assert d.allow is True

    def test_context_omitted(self, gate):
        d = gate.check(tool_name="read_file")
        assert d.allow is True


class TestOverride:
    def test_valid_override(self, gate):
        d = gate.check(tool_name="send_email", context={})
        result = gate.override(decision_id=d.id, correct_action="allow", reason="test")
        assert result is True

    def test_override_not_found(self, gate):
        result = gate.override(
            decision_id="00000000-0000-0000-0000-000000000000",
            correct_action="allow",
        )
        assert result is False

    def test_empty_decision_id_raises(self, gate):
        with pytest.raises(ValueError, match="decision_id is required"):
            gate.override(decision_id="", correct_action="allow")

    def test_invalid_correct_action_raises(self, gate):
        d = gate.check(tool_name="test", context={})
        with pytest.raises(ValueError, match="correct_action must be"):
            gate.override(decision_id=d.id, correct_action="maybe")

    def test_override_updates_prior(self, gate):
        # Make some decisions and override to allow
        for _ in range(3):
            d = gate.check(tool_name="send_email", context={})
            gate.override(decision_id=d.id, correct_action="allow", reason="safe")

        stats = gate.status()
        tool_info = stats["tools"].get("send_email", {})
        assert tool_info.get("override_count", 0) == 3
        assert tool_info.get("alpha", 1.0) > 1.0  # Should have increased


class TestStatus:
    def test_empty_stats(self, gate):
        stats = gate.status()
        assert stats["decisions_count"] == 0
        assert stats["overrides_count"] == 0

    def test_stats_after_decisions(self, gate):
        gate.check(tool_name="test1", context={})
        gate.check(tool_name="test2", context={})

        stats = gate.status()
        assert stats["decisions_count"] == 2

    def test_stats_per_tool(self, gate):
        gate.check(tool_name="send_email", context={})
        gate.check(tool_name="read_file", context={})

        stats = gate.status()
        assert "send_email" in stats["tools"]
        assert "read_file" in stats["tools"]


class TestFailMode:
    def test_fail_open_on_error(self, tmp_path):
        """fail_closed=False should allow on internal error."""
        gate = SansinLocal(db_path=str(tmp_path / "test.db"), fail_closed=False, load_community=False)
        # Force an error by making the DB read-only after init
        os.chmod(str(tmp_path / "test.db"), 0o444)
        d = gate.check(tool_name="test", context={})
        assert d.allow is True
        assert "failing open" in d.reason.lower()
        # Restore permissions for cleanup
        os.chmod(str(tmp_path / "test.db"), 0o644)

    def test_fail_closed_on_error(self, tmp_path):
        """fail_closed=True should block on internal error."""
        gate = SansinLocal(db_path=str(tmp_path / "test.db"), fail_closed=True, load_community=False)
        os.chmod(str(tmp_path / "test.db"), 0o444)
        d = gate.check(tool_name="test", context={})
        assert d.allow is False
        assert "failing closed" in d.reason.lower()
        os.chmod(str(tmp_path / "test.db"), 0o644)


class TestExportImport:
    def test_round_trip(self, gate, tmp_path):
        """Export then import preserves priors."""
        gate.check(tool_name="send_email", context={})
        d = gate.check(tool_name="send_email", context={})
        gate.override(decision_id=d.id, correct_action="allow", reason="safe")

        export_path = str(tmp_path / "export.json")
        gate.export_priors(export_path)

        # Create fresh gate and import
        gate2 = SansinLocal(db_path=str(tmp_path / "test2.db"), load_community=False)
        imported = gate2.import_priors(export_path)
        assert imported > 0

        stats = gate2.status()
        assert "send_email" in stats["tools"]

    def test_import_rejects_future_schema(self, gate, tmp_path):
        """Import should reject files with newer schema versions."""
        import json

        export_path = str(tmp_path / "future.json")
        with open(export_path, "w") as f:
            json.dump({"schema_version": 999, "priors": []}, f)

        with pytest.raises(ValueError, match="newer than"):
            gate.import_priors(export_path)


class TestCommunityPriors:
    def test_community_priors_loaded(self, gate_with_community):
        stats = gate_with_community.status()
        tools = stats["tools"]
        assert "send_email" in tools
        assert "delete_file" in tools
        assert "query_database" in tools

    def test_community_priors_not_loaded_when_disabled(self, gate):
        stats = gate.status()
        assert len(stats["tools"]) == 0


class TestThreadSafety:
    def test_concurrent_checks(self, gate):
        """Multiple threads calling check() should not corrupt state."""
        errors = []

        def worker(tool_name, n):
            try:
                for _ in range(10):
                    d = gate.check(tool_name=tool_name, context={})
                    assert d.id  # Should always get a valid ID
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"tool_{i}", 10))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        stats = gate.status()
        assert stats["decisions_count"] == 50  # 5 threads * 10 decisions
