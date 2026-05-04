"""
smig/catalogs/sampler.py
=========================
``sample_field`` orchestrator with on-disk caching for Phase 3 catalog
ingestion.

Cache
-----
Results are pickled to ``SMIG_CATALOG_CACHE`` (env var) or
``~/.smig/catalog_cache/`` by default.  Cache keys are stable SHA-256
digests of ``(provider_class_name, rounded_l, rounded_b, rounded_fov)``
so that equivalent calls return without re-querying the provider.

Cache key
---------
``hashlib.sha256(repr((cls_name, round(l_deg, 6), round(b_deg, 6), round(fov_deg, 6))).encode()).hexdigest()``

No new dependencies — only stdlib (``hashlib``, ``pickle``, ``pathlib``,
``os``).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path

import numpy as np

from smig.catalogs.base import CatalogProvider, StarRecord

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache root (configurable via env var)
# ---------------------------------------------------------------------------

def _cache_root() -> Path:
    env = os.environ.get("SMIG_CATALOG_CACHE")
    if env:
        root = Path(env)
    else:
        root = Path.home() / ".smig" / "catalog_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(provider: CatalogProvider, l_deg: float, b_deg: float, fov_deg: float) -> str:
    key_repr = repr((
        provider.__class__.__name__,
        round(l_deg, 6),
        round(b_deg, 6),
        round(fov_deg, 6),
    ))
    return hashlib.sha256(key_repr.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_field(
    provider: CatalogProvider,
    l_deg: float,
    b_deg: float,
    fov_deg: float,
    rng: np.random.Generator,
    *,
    use_cache: bool = True,
) -> list[StarRecord]:
    """Return stars from ``provider`` within the requested field, with caching.

    On the first call for a given ``(provider_class, l_deg, b_deg, fov_deg)``
    combination the provider is queried and the result is pickled to the cache
    directory.  Subsequent calls with the same arguments return the cached
    list without calling the provider.

    The cache stores only the ``list[StarRecord]`` — the RNG state is NOT
    cached, so stochastic providers may return different results on cache miss.

    Parameters
    ----------
    provider:
        Any :class:`~smig.catalogs.base.CatalogProvider` instance.
    l_deg:
        Galactic longitude of the field centre (degrees).
    b_deg:
        Galactic latitude of the field centre (degrees).
    fov_deg:
        Square FOV full width (degrees).
    rng:
        NumPy Generator forwarded to the provider on cache miss.
    use_cache:
        If ``False``, always call the provider and skip read/write to disk.
        Useful in tests.

    Returns
    -------
    list[StarRecord]
    """
    if not use_cache:
        return provider.sample_field(l_deg, b_deg, fov_deg, rng)

    root = _cache_root()
    key = _cache_key(provider, l_deg, b_deg, fov_deg)
    cache_file = root / f"{key}.pkl"

    if cache_file.exists():
        try:
            with cache_file.open("rb") as fh:
                stars: list[StarRecord] = pickle.load(fh)
            log.debug("catalog cache hit: %s", key[:12])
            return stars
        except Exception as exc:
            log.warning("catalog cache read failed (%s), re-querying provider", exc)

    stars = provider.sample_field(l_deg, b_deg, fov_deg, rng)

    try:
        tmp = cache_file.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(stars, fh)
        tmp.replace(cache_file)
        log.debug("catalog cache written: %s", key[:12])
    except Exception as exc:
        log.warning("catalog cache write failed (%s), continuing without cache", exc)

    return stars
