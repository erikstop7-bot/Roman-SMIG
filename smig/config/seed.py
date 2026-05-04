"""
smig/config/seed.py
===================
Deterministic, hierarchical seed derivation for reproducible Phase 2 simulations.

Contract
--------
* Same inputs always produce the same seed (determinism).
* Different inputs are highly likely to produce different seeds, but mathematical
  collision immunity is impossible: the 256-bit SHA-256 digest is reduced to a
  31-bit GalSim seed space (2**31 - 1 possible values), so collisions are
  mathematically unavoidable with sufficiently many distinct inputs.
* All output seeds are strictly positive 32-bit integers: 0 < seed < 2**31.
  This satisfies GalSim's requirement that ``galsim.BaseDeviate(seed)`` receives
  a positive integer — GalSim silently misuses a seed of 0.

Implementation notes
--------------------
SHA-256 (from :mod:`hashlib`) is used for all hashing.  Python's built-in
``hash()`` is intentionally avoided because it is salted per-process by
``PYTHONHASHSEED`` and therefore non-reproducible across interpreter invocations.

The hash payload is serialized with :func:`json.dumps` (canonical separators,
sorted keys) to avoid the ambiguity that arises when a colon-delimited string
format is used and a namespace or argument itself contains a colon character.

Seed normalisation
------------------
The SHA-256 digest (256-bit integer) is mapped to ``(0, 2**31)`` via::

    1 + (value % (2**31 - 1))

This yields a seed in ``[1, 2**31 - 1]``, which is strictly inside ``(0, 2**31)``.
The ``+ 1`` shift prevents the fatal seed-zero case even when
``value % (2**31 - 1) == 0``.

Domain separation
-----------------
Each function embeds a distinct namespace string in its payload::

    derive_event_seed : "smig/v2/event"
    derive_stage_seed : "smig/v2/stage"

This prevents collisions between the two seed families even when they receive
numerically identical arguments.
"""

from __future__ import annotations

import hashlib
import json

# Maximum exclusive bound for GalSim-compatible seed range.
_GALSIM_SEED_MAX = 2**31  # exclusive upper bound: seeds live in [1, 2**31 - 1]
_MODULUS = _GALSIM_SEED_MAX - 1  # 2**31 - 1


def derive_event_seed(
    master_seed: int,
    event_id: str,
    namespace: str = "smig/v2/event",
) -> int:
    """Derive a reproducible seed for a single microlensing event.

    Combines the master simulation seed, a unique event identifier, and a
    domain-separation namespace via SHA-256 to produce a deterministic,
    GalSim-compatible seed.

    Parameters
    ----------
    master_seed:
        Top-level integer seed for the simulation run.  Must be representable
        as a Python int (arbitrary precision is fine; it is embedded as a
        decimal string in the hash payload).
    event_id:
        Unique string identifier for the event (e.g. ``"OB240123"`` or
        ``"field07_star042"``).  Must be non-empty.
    namespace:
        Domain-separation prefix.  The default ``"smig/v2/event"`` must be
        used for all production event seeds; override only in tests.

    Returns
    -------
    int
        A seed in the open interval ``(0, 2**31)``, i.e. ``1 <= seed <= 2**31 - 1``.

    Examples
    --------
    >>> s1 = derive_event_seed(42, "OB240123")
    >>> s2 = derive_event_seed(42, "OB240123")
    >>> assert s1 == s2                          # deterministic
    >>> assert 0 < s1 < 2**31                   # GalSim-safe range
    """
    if not event_id:
        raise ValueError("event_id must be a non-empty string.")
    payload = json.dumps(
        {"namespace": namespace, "master_seed": master_seed, "event_id": event_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    value = int(digest, 16)
    return 1 + (value % _MODULUS)


def derive_stage_seed(
    event_seed: int,
    stage_name: str,
    namespace: str = "smig/v2/stage",
) -> int:
    """Derive a reproducible seed for a named pipeline stage within an event.

    Takes the integer seed returned by :func:`derive_event_seed` (not the
    original master seed) as input, ensuring full hierarchical dependency:
    ``master_seed → event_seed → stage_seed``.

    Parameters
    ----------
    event_seed:
        The integer seed returned by :func:`derive_event_seed` for this event.
    stage_name:
        Name of the pipeline stage (e.g. ``"psf_rendering"``, ``"noise"``,
        ``"source_injection"``).  Must be non-empty.
    namespace:
        Domain-separation prefix.  The default ``"smig/v2/stage"`` must be
        used for all production stage seeds; override only in tests.

    Returns
    -------
    int
        A seed in the open interval ``(0, 2**31)``, i.e. ``1 <= seed <= 2**31 - 1``.

    Examples
    --------
    >>> event_seed = derive_event_seed(42, "OB240123")
    >>> s_psf = derive_stage_seed(event_seed, "psf_rendering")
    >>> s_noise = derive_stage_seed(event_seed, "noise")
    >>> assert s_psf != s_noise                  # independent
    >>> assert 0 < s_psf < 2**31               # GalSim-safe range
    """
    if not stage_name:
        raise ValueError("stage_name must be a non-empty string.")
    payload = json.dumps(
        {"namespace": namespace, "event_seed": event_seed, "stage_name": stage_name},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    value = int(digest, 16)
    return 1 + (value % _MODULUS)
