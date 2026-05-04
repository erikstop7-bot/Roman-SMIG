"""
smig/catalogs/base.py
=====================
Abstract base types for Phase 3 catalog ingestion.

Defines :class:`StarRecord` (frozen dataclass), :class:`CatalogProvider`
(ABC), and :class:`MissingColumnError`.  All concrete providers validate
their expected column schema on load and raise ``MissingColumnError`` with
a list of missing column names.

Phase 3.1 is single-band F146.  ``StarRecord.mag_other_ab`` defaults to an
empty immutable :class:`~types.MappingProxyType` for forward-compatibility
with multi-band extensions.

Fields required downstream in Phase 3.2 for microlensing source radii and
limb-darkening coefficients: ``mass_msun``, ``teff_K``, ``log_g``,
``metallicity_feh``.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


class MissingColumnError(Exception):
    """Raised when a catalog file is missing required columns.

    Parameters
    ----------
    missing:
        List of column names that are absent from the loaded file.
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(f"Catalog is missing required columns: {missing}")


@dataclass(frozen=True)
class StarRecord:
    """Immutable record for a single star from any catalog provider.

    Mutation raises :class:`dataclasses.FrozenInstanceError`.

    Fields
    ------
    galactic_l_deg:
        Galactic longitude (degrees).
    galactic_b_deg:
        Galactic latitude (degrees).
    distance_kpc:
        Heliocentric distance (kpc).
    mass_msun:
        Stellar mass (solar masses).  Consumed by Phase 3.2 for source-radius
        and limb-darkening derivation.
    teff_K:
        Effective temperature (K).  Consumed by Phase 3.2.
    log_g:
        Log surface gravity (dex, CGS).  Consumed by Phase 3.2.
    metallicity_feh:
        Metallicity [Fe/H] (dex).  Consumed by Phase 3.2.
    mag_F146_ab:
        Roman WFI F146 AB magnitude.  Maps to the ``mag_w146`` column in the
        :class:`~smig.rendering.crowding.CrowdedFieldRenderer` DataFrame.
    mag_other_ab:
        Optional additional AB magnitudes keyed by band name (e.g. ``"F106"``).
        Defaults to an empty :class:`~types.MappingProxyType` so the field is
        always present but immutable.  Phase 3.1 populates only F146.
    source_id:
        Catalog-specific unique source identifier (string).
    catalog_tile_id:
        Identifier of the catalog spatial tile from which this star was drawn.
    """

    galactic_l_deg: float
    galactic_b_deg: float
    distance_kpc: float
    mass_msun: float
    teff_K: float
    log_g: float
    metallicity_feh: float
    mag_F146_ab: float
    source_id: str
    catalog_tile_id: str
    mag_other_ab: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )


class CatalogProvider(abc.ABC):
    """Abstract base class for star catalog providers.

    Concrete subclasses load a catalog from disk (or generate it
    synthetically) and return lists of :class:`StarRecord` objects.

    All providers MUST validate their expected column schema on load and
    raise :class:`MissingColumnError` with the list of missing names.
    """

    @abc.abstractmethod
    def sample_field(
        self,
        l_deg: float,
        b_deg: float,
        fov_deg: float,
        rng: np.random.Generator,
    ) -> list[StarRecord]:
        """Return all stars within a square FOV centred on ``(l_deg, b_deg)``.

        Parameters
        ----------
        l_deg:
            Galactic longitude of the field centre (degrees).
        b_deg:
            Galactic latitude of the field centre (degrees).
        fov_deg:
            Full side-length of the square field of view (degrees).
        rng:
            NumPy random Generator for stochastic providers.  Deterministic
            providers may ignore this parameter.

        Returns
        -------
        list[StarRecord]
            Stars within the requested field, in arbitrary order.
        """

    @abc.abstractmethod
    def list_bands(self) -> tuple[str, ...]:
        """Return the photometric band names available in this catalog.

        Returns
        -------
        tuple[str, ...]
            Band names, e.g. ``("F146",)``.  Phase 3.1 returns only
            ``("F146",)`` for all providers.
        """
