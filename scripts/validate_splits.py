#!/usr/bin/env python3
"""
scripts/validate_splits.py
==========================
Data leakage prevention CLI for SMIG v2 training datasets.

Reads a JSON manifest and checks four types of problems:

(a) **Manifest structure**: required keys present, correct types, non-empty
    ``events`` list.
(b) **Valid split labels**: only ``"train"``, ``"val"``, ``"test"`` are allowed.
(c) **Duplicate event IDs within the same split** and
    **duplicate event IDs across splits**.
(d) **Shared starfield seeds across splits** (same seed in different splits → leakage).
    Duplicate seeds within the *same* split are permitted.
(e) **Parameter similarity (transitive)**: events whose *all* microlensing
    parameters are within 5%% of each other are connected by an undirected
    edge; every connected component must be contained entirely within one
    split label.

Manifest schema
---------------
.. code-block:: json

    {
        "events": [
            {
                "event_id": "ob230001",
                "split": "train",
                "starfield_seed": 42,
                "params": {
                    "t_E": 30.0,
                    "u_0": 0.1,
                    "s": 1.0,
                    "q": 0.001
                }
            },
            ...
        ]
    }

``split`` must be one of ``"train"``, ``"val"``, or ``"test"``.

``starfield_seed`` must be an integer or a base-10 integer string.
Float values and float-like strings are rejected.

Parameter similarity definition
---------------------------------
Two events are "parameter-similar" when, for every key present in their
``params`` dicts, the symmetric relative difference satisfies:

    abs(a - b) / max(abs(a), abs(b), 1e-9) <= 0.05

This O(N²) MVP check is intentionally simple; production use at scale would
require an indexed approximate-nearest-neighbour search.

Transitive leakage detection
-----------------------------
A Union-Find (connected-components) algorithm groups events that are
parameter-similar (directly or transitively).  Every component must fall
entirely within one split label; components that span multiple labels are
flagged as leakage violations.

Exit codes
----------
* ``0`` — all checks pass (safe to proceed with training).
* ``1`` — at least one violation found (training must NOT proceed).

Usage
-----
::

    python scripts/validate_splits.py path/to/manifest.json

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})
_REQUIRED_EVENT_KEYS: tuple[str, ...] = ("event_id", "split", "starfield_seed", "params")


# ---------------------------------------------------------------------------
# Seed parsing
# ---------------------------------------------------------------------------

def _parse_seed(raw: object, event_id: str) -> int | str:
    """Parse *raw* into an integer seed, or return an error string.

    Accepts:
    * ``int`` values directly.
    * ``str`` values that represent a base-10 integer (no leading/trailing
      whitespace after stripping, no decimal point).

    Rejects (returns an error string):
    * ``float`` values (would silently truncate, e.g. ``42.9 → 42``).
    * Strings that contain a decimal point or are not purely numeric.
    * Any other type.

    Returns
    -------
    int
        Parsed seed value.
    str
        Human-readable error message if parsing failed.
    """
    if isinstance(raw, float):
        return (
            f"event '{event_id}': starfield_seed {raw!r} is a float — "
            "integer or base-10 integer string required"
        )
    if isinstance(raw, bool):
        return (
            f"event '{event_id}': starfield_seed {raw!r} is a bool — "
            "integer required"
        )
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if "." in s:
            return (
                f"event '{event_id}': starfield_seed {raw!r} looks like a "
                "float string — integer string required"
            )
        try:
            return int(s, 10)
        except ValueError:
            return (
                f"event '{event_id}': starfield_seed {raw!r} is not a valid "
                "base-10 integer string"
            )
    return (
        f"event '{event_id}': starfield_seed {raw!r} has unexpected type "
        f"{type(raw).__name__!r} — integer or base-10 integer string required"
    )


# ---------------------------------------------------------------------------
# Similarity check
# ---------------------------------------------------------------------------

def _params_within_5pct(params_a: dict, params_b: dict) -> bool:
    """Return ``True`` if every parameter in both dicts is within 5%% of the other.

    Uses a symmetric relative difference with an absolute floor to handle
    near-zero parameter values safely:

        |a - b| / max(|a|, |b|, 1e-9) <= 0.05

    Parameters
    ----------
    params_a, params_b:
        Dicts mapping parameter name → float value.  Both dicts must contain
        the same keys; any key absent from either dict causes ``False``.

    Returns
    -------
    bool
        ``True`` when ALL pairwise comparisons satisfy the 5%% criterion.
    """
    all_keys = set(params_a) | set(params_b)
    for key in all_keys:
        if key not in params_a or key not in params_b:
            return False
        a = float(params_a[key])
        b = float(params_b[key])
        denom = max(abs(a), abs(b), 1e-9)
        if abs(a - b) / denom > 0.05:
            return False
    return True


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Weighted Union-Find with path compression."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_manifest(manifest: dict) -> list[str]:
    """Validate *manifest* for data leakage.  Return list of violation strings.

    An empty list means the manifest is clean.

    Parameters
    ----------
    manifest:
        Parsed JSON object.

    Returns
    -------
    list[str]
        Human-readable violation messages, one per detected problem.
        Structural problems are flagged as Validation Failures; leakage
        problems are flagged as Leakage Violations.
    """
    violations: list[str] = []

    # ------------------------------------------------------------------
    # Manifest structure validation
    # ------------------------------------------------------------------
    if not isinstance(manifest, dict):
        violations.append(
            "manifest root is not a JSON object"
        )
        return violations

    if "events" not in manifest:
        violations.append(
            "manifest is missing required top-level key 'events'"
        )
        return violations

    events_raw = manifest["events"]
    if not isinstance(events_raw, list):
        violations.append(
            f"'events' must be a list, got {type(events_raw).__name__!r}"
        )
        return violations

    if len(events_raw) == 0:
        violations.append(
            "manifest contains 0 events — at least one event is required"
        )
        return violations

    # ------------------------------------------------------------------
    # Per-event structural validation; collect clean events for deeper checks
    # ------------------------------------------------------------------
    clean_events: list[dict] = []
    for idx, ev in enumerate(events_raw):
        if not isinstance(ev, dict):
            violations.append(
                f"event at index {idx} is not a JSON object"
            )
            continue

        # Identify the event for error messages (best-effort).
        label = repr(ev.get("event_id", f"<index {idx}>"))

        missing = [k for k in _REQUIRED_EVENT_KEYS if k not in ev]
        if missing:
            violations.append(
                f"event {label} is missing required keys: {missing}"
            )
            continue

        if not isinstance(ev["params"], dict):
            violations.append(
                f"event {label}: 'params' must be a dict, "
                f"got {type(ev['params']).__name__!r}"
            )
            continue

        split_val = ev["split"]
        if not isinstance(split_val, str):
            violations.append(
                f"event {label}: 'split' must be a string, "
                f"got {type(split_val).__name__!r}"
            )
            continue

        if split_val not in _VALID_SPLITS:
            violations.append(
                f"event {label}: invalid split label {split_val!r} — "
                f"must be one of {sorted(_VALID_SPLITS)}"
            )
            continue

        seed_result = _parse_seed(ev["starfield_seed"], str(ev["event_id"]))
        if isinstance(seed_result, str):
            violations.append(seed_result)
            continue

        clean_events.append({
            "event_id": str(ev["event_id"]),
            "split": split_val,
            "starfield_seed": seed_result,
            "params": ev["params"],
        })

    if violations:
        # Structural problems found: skip semantic checks to avoid cascading noise.
        return violations

    # ------------------------------------------------------------------
    # (a) Duplicate event_id — both within and across splits
    # ------------------------------------------------------------------
    event_id_to_split: dict[str, str] = {}
    seen_ids_per_split: dict[str, set[str]] = {}
    for ev in clean_events:
        eid = ev["event_id"]
        split = ev["split"]
        seen_ids_per_split.setdefault(split, set())
        if eid in seen_ids_per_split[split]:
            violations.append(
                f"duplicate event_id '{eid}' within split '{split}'"
            )
        else:
            seen_ids_per_split[split].add(eid)

        if eid in event_id_to_split:
            if event_id_to_split[eid] != split:
                violations.append(
                    f"event_id '{eid}' appears in both split "
                    f"'{event_id_to_split[eid]}' and split '{split}'"
                )
        else:
            event_id_to_split[eid] = split

    # ------------------------------------------------------------------
    # (b) Duplicate starfield_seed across splits only
    # Duplicate seeds within the same split are allowed (no leakage risk).
    # ------------------------------------------------------------------
    seed_to_split: dict[int, str] = {}
    seed_to_event: dict[int, str] = {}
    for ev in clean_events:
        seed = ev["starfield_seed"]
        split = ev["split"]
        eid = ev["event_id"]

        if seed in seed_to_split:
            if seed_to_split[seed] != split:
                violations.append(
                    f"starfield_seed {seed} is shared between "
                    f"event '{seed_to_event[seed]}' (split='{seed_to_split[seed]}') "
                    f"and event '{eid}' (split='{split}')"
                )
        else:
            seed_to_split[seed] = split
            seed_to_event[seed] = eid

    # ------------------------------------------------------------------
    # (c) Transitive parameter leakage via Union-Find
    # ------------------------------------------------------------------
    n = len(clean_events)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _params_within_5pct(clean_events[i]["params"], clean_events[j]["params"]):
                uf.union(i, j)

    # Group indices by component root.
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        components.setdefault(root, []).append(i)

    for indices in components.values():
        splits_in_component = {clean_events[i]["split"] for i in indices}
        if len(splits_in_component) > 1:
            members = [
                f"'{clean_events[i]['event_id']}' (split='{clean_events[i]['split']}')"
                for i in indices
            ]
            violations.append(
                f"parameter-similar connected component spans multiple splits "
                f"{sorted(splits_in_component)}: {', '.join(members)} — "
                f"transitive parameter-space leakage"
            )

    return violations


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, validate manifest, print results, return exit code."""
    parser = argparse.ArgumentParser(
        prog="validate_splits",
        description=(
            "Validate a SMIG dataset manifest for data leakage before training."
        ),
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the JSON manifest file.",
    )
    args = parser.parse_args(argv)

    manifest_path: Path = args.manifest
    if not manifest_path.exists():
        print(f"ERROR: manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in {manifest_path}: {exc}", file=sys.stderr)
        return 1

    violations = validate_manifest(manifest_data)

    if violations:
        print(
            f"FAILED — {len(violations)} violation(s) detected:",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  VIOLATION: {v}", file=sys.stderr)
        return 1

    n_events = len(manifest_data.get("events", []))
    print(
        f"OK — all leakage checks passed ({n_events} events validated).",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
