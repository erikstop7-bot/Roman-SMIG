"""
smig/catalogs/synthetic.py
===========================
Synthetic catalog provider for Phase 3 testing and smoke invocation.

Phase 3.1 does NOT wire real catalogs into :class:`~smig.rendering.pipeline.SceneSimulator`.
Phase 2's ``pipeline._generate_catalog`` remains the active code path until
Phase 3.5 worker integration.

:class:`SyntheticCatalogProvider` exposes the :class:`~smig.catalogs.base.CatalogProvider`
interface with a uniform-random star population, allowing the full
catalog → adapter → renderer chain to be exercised without real catalog data.

The star population model mirrors the Phase 2 ``_generate_catalog`` logic:

- Positions: drawn uniformly within the requested FOV.
- F146 magnitudes: uniform in ``[20, 26]`` AB mag.
- Physical parameters: drawn from plausible ranges for Galactic-bulge stars.

This duplication resolves when Phase 3.5 wires catalogs through the worker.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from smig.catalogs.base import CatalogProvider, StarRecord

# Physical parameter ranges for synthetic stars (representative Galactic bulge).
_MAG_MIN: float = 20.0
_MAG_MAX: float = 26.0
_DIST_MIN_KPC: float = 1.0
_DIST_MAX_KPC: float = 12.0
_MASS_MIN: float = 0.1
_MASS_MAX: float = 2.0
_TEFF_MIN: float = 3500.0
_TEFF_MAX: float = 8000.0
_LOGG_MIN: float = 1.0
_LOGG_MAX: float = 5.0
_FEH_MIN: float = -1.5
_FEH_MAX: float = 0.5


class SyntheticCatalogProvider(CatalogProvider):
    """Synthetic uniform-random star catalog provider.

    Generates stars with positions drawn uniformly within the requested FOV
    and physical parameters drawn from plausible Galactic-bulge ranges.

    Used for smoke tests, the CLI entry point, and Phase 3.5 integration
    development.  Does NOT replace ``pipeline._generate_catalog``.

    Parameters
    ----------
    n_stars:
        Number of stars to generate per :meth:`sample_field` call.
        Default is 50.
    tile_id:
        Catalog tile identifier string embedded in every :class:`StarRecord`.
    """

    def __init__(
        self,
        n_stars: int = 50,
        tile_id: str = "synthetic",
    ) -> None:
        self._n_stars = n_stars
        self._tile_id = tile_id

    def sample_field(
        self,
        l_deg: float,
        b_deg: float,
        fov_deg: float,
        rng: np.random.Generator,
    ) -> list[StarRecord]:
        """Generate ``n_stars`` synthetic stars within a square FOV.

        Star positions are drawn uniformly in Galactic (l, b) within
        ``[l_deg ± fov_deg/2, b_deg ± fov_deg/2]``.

        Parameters
        ----------
        l_deg:
            Field centre Galactic longitude (degrees).
        b_deg:
            Field centre Galactic latitude (degrees).
        fov_deg:
            Square FOV full width (degrees).
        rng:
            NumPy Generator used for all random draws.

        Returns
        -------
        list[StarRecord]
            ``n_stars`` frozen StarRecord objects.
        """
        n = self._n_stars
        half = fov_deg / 2.0

        ls = rng.uniform(l_deg - half, l_deg + half, size=n)
        bs = rng.uniform(b_deg - half, b_deg + half, size=n)
        dists = rng.uniform(_DIST_MIN_KPC, _DIST_MAX_KPC, size=n)
        masses = rng.uniform(_MASS_MIN, _MASS_MAX, size=n)
        teffs = rng.uniform(_TEFF_MIN, _TEFF_MAX, size=n)
        loggs = rng.uniform(_LOGG_MIN, _LOGG_MAX, size=n)
        fehs = rng.uniform(_FEH_MIN, _FEH_MAX, size=n)
        mags = rng.uniform(_MAG_MIN, _MAG_MAX, size=n)

        return [
            StarRecord(
                galactic_l_deg=float(ls[i]),
                galactic_b_deg=float(bs[i]),
                distance_kpc=float(dists[i]),
                mass_msun=float(masses[i]),
                teff_K=float(teffs[i]),
                log_g=float(loggs[i]),
                metallicity_feh=float(fehs[i]),
                mag_F146_ab=float(mags[i]),
                source_id=f"synth_{i:06d}",
                catalog_tile_id=self._tile_id,
            )
            for i in range(n)
        ]

    def list_bands(self) -> tuple[str, ...]:
        return ("F146",)
