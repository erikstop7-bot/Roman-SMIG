"""
scripts/test_validate_splits.py
================================
Regression tests for :func:`validate_manifest` and :func:`_parse_seed` in
``scripts/validate_splits.py``.

Coverage
--------
VS-1   Empty manifest (0 events) fails with a Validation Failure message.
VS-2   Missing required top-level key 'events' fails cleanly.
VS-3   Missing required per-event keys fail cleanly without exception.
VS-4   Invalid split labels outside {train, val, test} fail cleanly.
VS-5   Duplicate event_id within the same split fails cleanly.
VS-6   Duplicate starfield_seed within the same split fails cleanly.
VS-7   Duplicate event_id across splits fails cleanly.
VS-8   Duplicate starfield_seed across splits fails cleanly.
VS-9   Float seed value fails cleanly (no silent truncation, no crash).
VS-10  Float-string seed fails cleanly.
VS-11  Hex-string seed fails cleanly.
VS-12  Non-numeric string seed fails cleanly.
VS-13  Malformed manifest (wrong type for 'events') fails cleanly.
VS-14  Transitive parameter leakage (A~B, B~C, A and C in different splits)
       is caught and flagged.
VS-15  Non-transitive similarity (same split) passes cleanly.
VS-16  Clean manifest (no violations) returns empty list and exits 0.
VS-17  Provenance: nested dict serialization is deterministic across key
       insertion orders.
VS-18  Set values in sanitize_rng_state are converted to sorted lists.

Run from the SMIG project root::

    python -m pytest scripts/test_validate_splits.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Make the scripts/ directory importable.
sys.path.insert(0, str(Path(__file__).parent))

from validate_splits import _parse_seed, validate_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_id: str = "ev001",
    split: str = "train",
    starfield_seed: Any = 42,
    params: dict | None = None,
) -> dict:
    if params is None:
        params = {"t_E": 30.0, "u_0": 0.1, "s": 1.0, "q": 0.001}
    return {
        "event_id": event_id,
        "split": split,
        "starfield_seed": starfield_seed,
        "params": params,
    }


def _make_manifest(events: list[dict]) -> dict:
    return {"events": events}


def _assert_passes(events: list[dict]) -> None:
    violations = validate_manifest(_make_manifest(events))
    assert violations == [], f"Expected no violations, got: {violations}"


def _assert_fails(events: list[dict], match: str | None = None) -> list[str]:
    violations = validate_manifest(_make_manifest(events))
    assert violations, "Expected at least one violation, got none."
    if match:
        combined = " ".join(violations)
        assert match.lower() in combined.lower(), (
            f"Expected {match!r} in violations, got: {violations}"
        )
    return violations


# ---------------------------------------------------------------------------
# VS-1: Empty manifest
# ---------------------------------------------------------------------------

class TestEmptyManifest:
    def test_empty_events_list_fails(self) -> None:
        violations = validate_manifest({"events": []})
        assert violations, "Empty events list should produce a violation."
        assert any("0 events" in v or "empty" in v.lower() for v in violations)

    def test_empty_events_no_exception(self) -> None:
        """Must not raise — failure is communicated via the return value."""
        result = validate_manifest({"events": []})
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# VS-2 & VS-13: Structural failures
# ---------------------------------------------------------------------------

class TestStructuralFailures:
    def test_missing_events_key(self) -> None:
        violations = validate_manifest({"not_events": []})
        assert violations
        assert any("events" in v for v in violations)

    def test_events_not_a_list(self) -> None:
        violations = validate_manifest({"events": "should_be_a_list"})
        assert violations
        assert any("list" in v.lower() or "events" in v for v in violations)

    def test_root_not_dict(self) -> None:
        violations = validate_manifest([])  # type: ignore[arg-type]
        assert violations

    def test_event_not_a_dict(self) -> None:
        violations = validate_manifest({"events": ["string_not_dict"]})
        assert violations
        assert any("not a JSON object" in v or "object" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# VS-3: Missing per-event keys
# ---------------------------------------------------------------------------

class TestMissingEventKeys:
    @pytest.mark.parametrize("missing_key", ["event_id", "split", "starfield_seed", "params"])
    def test_missing_key_fails_cleanly(self, missing_key: str) -> None:
        ev = _make_event()
        del ev[missing_key]
        violations = validate_manifest({"events": [ev]})
        assert violations, f"Expected failure when '{missing_key}' is missing."
        # Must not raise — verified by reaching this assertion.

    def test_missing_key_does_not_raise(self) -> None:
        ev = {"event_id": "ev001"}  # missing split, starfield_seed, params
        try:
            violations = validate_manifest({"events": [ev]})
        except (KeyError, TypeError) as exc:
            pytest.fail(f"validate_manifest raised {type(exc).__name__}: {exc}")
        assert violations


# ---------------------------------------------------------------------------
# VS-4: Invalid split labels
# ---------------------------------------------------------------------------

class TestInvalidSplitLabels:
    @pytest.mark.parametrize("bad_split", ["Train", "TRAIN", "validation", "holdout", "", "0"])
    def test_invalid_split_fails(self, bad_split: str) -> None:
        ev = _make_event(split=bad_split)
        _assert_fails([ev], match="invalid split")

    @pytest.mark.parametrize("good_split", ["train", "val", "test"])
    def test_valid_splits_pass(self, good_split: str) -> None:
        ev = _make_event(split=good_split)
        _assert_passes([ev])


# ---------------------------------------------------------------------------
# VS-5: Duplicate event_id within the same split
# ---------------------------------------------------------------------------

class TestDuplicateEventIdWithinSplit:
    def test_duplicate_event_id_same_split_fails(self) -> None:
        ev1 = _make_event("dup_id", "train", 1)
        ev2 = _make_event("dup_id", "train", 2)
        _assert_fails([ev1, ev2], match="duplicate")

    def test_same_event_id_different_splits_cross_split_violation(self) -> None:
        ev1 = _make_event("shared_id", "train", 1)
        ev2 = _make_event("shared_id", "val", 2)
        violations = _assert_fails([ev1, ev2])
        assert any("shared_id" in v for v in violations)


# ---------------------------------------------------------------------------
# VS-6: Duplicate starfield_seed within the same split
# ---------------------------------------------------------------------------

class TestDuplicateSeedWithinSplit:
    def test_duplicate_seed_same_split_passes(self) -> None:
        # B.5: same seed in same split is allowed (no leakage risk).
        ev1 = _make_event("ev001", "train", 99)
        ev2 = _make_event("ev002", "train", 99)
        _assert_passes([ev1, ev2])

    def test_same_seed_different_event_ids_different_splits_flagged(self) -> None:
        ev1 = _make_event("ev001", "train", 77)
        ev2 = _make_event("ev002", "val", 77)
        violations = _assert_fails([ev1, ev2])
        assert any("77" in v for v in violations)


# ---------------------------------------------------------------------------
# VS-7: Duplicate event_id across splits
# ---------------------------------------------------------------------------

class TestDuplicateEventIdAcrossSplits:
    def test_cross_split_event_id_fails(self) -> None:
        ev1 = _make_event("shared", "train", 10)
        ev2 = _make_event("shared", "test", 20)
        _assert_fails([ev1, ev2])

    def test_unique_event_ids_pass(self) -> None:
        events = [_make_event(f"ev{i:03d}", "train", i) for i in range(5)]
        _assert_passes(events)


# ---------------------------------------------------------------------------
# VS-8: Duplicate starfield_seed across splits (already in VS-6 above,
#        but covered explicitly here for clarity)
# ---------------------------------------------------------------------------

class TestDuplicateSeedAcrossSplits:
    def test_seed_shared_train_val_fails(self) -> None:
        ev1 = _make_event("ev001", "train", 555)
        ev2 = _make_event("ev002", "val", 555)
        violations = _assert_fails([ev1, ev2])
        assert any("555" in v for v in violations)

    def test_distinct_seeds_pass(self) -> None:
        events = [_make_event(f"ev{i}", "train", i * 100) for i in range(1, 4)]
        _assert_passes(events)


# ---------------------------------------------------------------------------
# VS-9 / VS-10 / VS-11 / VS-12: Seed parsing
# ---------------------------------------------------------------------------

class TestSeedParsing:
    def test_float_seed_fails(self) -> None:
        ev = _make_event(starfield_seed=42.9)
        _assert_fails([ev], match="float")

    def test_float_zero_seed_fails(self) -> None:
        ev = _make_event(starfield_seed=0.0)
        _assert_fails([ev], match="float")

    def test_float_string_seed_fails(self) -> None:
        ev = _make_event(starfield_seed="42.0")
        _assert_fails([ev])

    def test_hex_string_seed_fails(self) -> None:
        ev = _make_event(starfield_seed="0x2a")
        _assert_fails([ev])

    def test_non_numeric_string_seed_fails(self) -> None:
        ev = _make_event(starfield_seed="abc")
        _assert_fails([ev])

    def test_int_seed_passes(self) -> None:
        ev = _make_event(starfield_seed=42)
        _assert_passes([ev])

    def test_base10_string_seed_passes(self) -> None:
        ev = _make_event(starfield_seed="42")
        _assert_passes([ev])

    def test_negative_int_seed_passes(self) -> None:
        ev = _make_event(starfield_seed=-7)
        _assert_passes([ev])

    def test_zero_int_seed_passes(self) -> None:
        ev = _make_event(starfield_seed=0)
        _assert_passes([ev])

    # _parse_seed unit tests
    def test_parse_seed_int(self) -> None:
        assert _parse_seed(42, "ev") == 42

    def test_parse_seed_string(self) -> None:
        assert _parse_seed("42", "ev") == 42

    def test_parse_seed_float_returns_error(self) -> None:
        result = _parse_seed(42.9, "ev")
        assert isinstance(result, str)

    def test_parse_seed_hex_returns_error(self) -> None:
        result = _parse_seed("0x2a", "ev")
        assert isinstance(result, str)

    def test_parse_seed_float_string_returns_error(self) -> None:
        result = _parse_seed("3.14", "ev")
        assert isinstance(result, str)

    def test_parse_seed_does_not_truncate_float(self) -> None:
        result = _parse_seed(42.9, "ev")
        # Must NOT silently return 42.
        assert result != 42


# ---------------------------------------------------------------------------
# VS-14: Transitive parameter leakage (Union-Find)
# ---------------------------------------------------------------------------

class TestTransitiveLeakage:
    """A~B and B~C but A and C in different splits → leakage."""

    def _similar_params(self, base: float = 30.0) -> dict:
        return {"t_E": base, "u_0": 0.1, "s": 1.0, "q": 0.001}

    def test_transitive_leakage_detected(self) -> None:
        """A (train) ~ B (val) ~ C (test): A, B, C form one component → violation."""
        pa = self._similar_params(30.0)
        pb = self._similar_params(30.0 * 1.02)   # within 5% of pa
        pc = self._similar_params(30.0 * 1.04)   # within 5% of pb → transitively connected

        ev_a = _make_event("ev_A", "train", 1, pa)
        ev_b = _make_event("ev_B", "val",   2, pb)
        ev_c = _make_event("ev_C", "test",  3, pc)

        violations = validate_manifest(_make_manifest([ev_a, ev_b, ev_c]))
        assert violations, "Expected a transitive leakage violation."
        assert any("transitive" in v.lower() or "connected component" in v.lower()
                   for v in violations)

    def test_direct_cross_split_leakage_detected(self) -> None:
        """A (train) ~ B (val) directly — should still be caught."""
        pa = self._similar_params(30.0)
        pb = self._similar_params(30.0 * 1.01)

        ev_a = _make_event("ev_A", "train", 1, pa)
        ev_b = _make_event("ev_B", "val",   2, pb)

        violations = validate_manifest(_make_manifest([ev_a, ev_b]))
        assert violations

    def test_same_split_similarity_passes(self) -> None:
        """Events that are parameter-similar but in the same split are fine."""
        pa = self._similar_params(30.0)
        pb = self._similar_params(30.0 * 1.01)

        ev_a = _make_event("ev_A", "train", 1, pa)
        ev_b = _make_event("ev_B", "train", 2, pb)

        _assert_passes([ev_a, ev_b])

    def test_dissimilar_events_different_splits_passes(self) -> None:
        """Dissimilar events in different splits should pass."""
        pa = {"t_E": 30.0, "u_0": 0.1, "s": 1.0, "q": 0.001}
        pb = {"t_E": 200.0, "u_0": 0.9, "s": 0.5, "q": 0.1}

        ev_a = _make_event("ev_A", "train", 1, pa)
        ev_b = _make_event("ev_B", "val",   2, pb)

        _assert_passes([ev_a, ev_b])

    def test_three_event_chain_same_split_passes(self) -> None:
        """A~B~C all in train: no leakage."""
        pa = self._similar_params(30.0)
        pb = self._similar_params(30.0 * 1.02)
        pc = self._similar_params(30.0 * 1.04)

        ev_a = _make_event("ev_A", "train", 1, pa)
        ev_b = _make_event("ev_B", "train", 2, pb)
        ev_c = _make_event("ev_C", "train", 3, pc)

        _assert_passes([ev_a, ev_b, ev_c])


# ---------------------------------------------------------------------------
# VS-15: Non-transitive similarity (same split) passes
# ---------------------------------------------------------------------------

class TestSameSplitSimilarityPasses:
    def test_many_similar_same_split(self) -> None:
        events = [
            _make_event(f"ev{i:03d}", "train", i, {"t_E": 30.0, "u_0": 0.1})
            for i in range(10)
        ]
        _assert_passes(events)


# ---------------------------------------------------------------------------
# VS-16: Clean manifest passes
# ---------------------------------------------------------------------------

class TestCleanManifest:
    def test_clean_manifest_returns_empty_list(self) -> None:
        events = [
            _make_event("ob230001", "train", 1, {"t_E": 30.0, "u_0": 0.1}),
            _make_event("ob230002", "val",   2, {"t_E": 200.0, "u_0": 0.9}),
            _make_event("ob230003", "test",  3, {"t_E": 100.0, "u_0": 0.5}),
        ]
        violations = validate_manifest(_make_manifest(events))
        assert violations == []

    def test_clean_manifest_all_three_splits(self) -> None:
        events = [
            _make_event(f"ev_{sp}_{i}", sp, i + idx * 100,
                        {"t_E": float(i * 10 + idx * 1000), "u_0": 0.1})
            for idx, sp in enumerate(["train", "val", "test"])
            for i in range(1, 4)
        ]
        violations = validate_manifest(_make_manifest(events))
        assert violations == []


# ---------------------------------------------------------------------------
# VS-17: Provenance deterministic nested dict serialization
# ---------------------------------------------------------------------------

class TestProvenanceDeterministicSerialization:
    def test_nested_dict_keys_sorted(self) -> None:
        from smig.provenance.schema import sanitize_rng_state

        state_a = {"z_key": 1, "a_key": 2, "m_key": {"z": 10, "a": 20}}
        state_b = {"a_key": 2, "m_key": {"a": 20, "z": 10}, "z_key": 1}

        result_a = sanitize_rng_state(state_a)
        result_b = sanitize_rng_state(state_b)

        json_a = json.dumps(result_a)
        json_b = json.dumps(result_b)

        assert json_a == json_b, (
            f"Same logical dict with different insertion order produced "
            f"different JSON:\n  {json_a}\n  {json_b}"
        )

    def test_nested_keys_are_sorted_ascending(self) -> None:
        from smig.provenance.schema import sanitize_rng_state

        state = {"z": 1, "a": 2, "m": 3}
        result = sanitize_rng_state(state)
        assert list(result.keys()) == sorted(result.keys())

    def test_numpy_arrays_still_converted(self) -> None:
        import numpy as np
        from smig.provenance.schema import sanitize_rng_state

        import numpy as np
        state = {"arr": np.array([1, 2, 3]), "scalar": np.uint64(7)}
        result = sanitize_rng_state(state)
        assert result["arr"] == [1, 2, 3]
        assert result["scalar"] == 7


# ---------------------------------------------------------------------------
# VS-18: Set values converted to sorted lists
# ---------------------------------------------------------------------------

class TestSetHandling:
    def test_set_converted_to_sorted_list(self) -> None:
        from smig.provenance.schema import sanitize_rng_state

        state = {"my_set": {3, 1, 2}}
        result = sanitize_rng_state(state)
        assert result["my_set"] == [1, 2, 3]

    def test_nested_set_converted(self) -> None:
        from smig.provenance.schema import sanitize_rng_state

        state = {"outer": {"inner_set": {"c", "a", "b"}}}
        result = sanitize_rng_state(state)
        assert result["outer"]["inner_set"] == ["a", "b", "c"]

    def test_set_is_json_serializable(self) -> None:
        from smig.provenance.schema import sanitize_rng_state

        state = {"tags": {"beta", "alpha", "gamma"}}
        result = sanitize_rng_state(state)
        # Must not raise
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["tags"] == ["alpha", "beta", "gamma"]
