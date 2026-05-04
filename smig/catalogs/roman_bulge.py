"""
smig/catalogs/roman_bulge.py
=============================
Roman core-bulge catalog ingestor based on Penny et al. 2019.

Reference: Penny et al. 2019, ApJS, 245, 16 — "Predictions of the WFIRST
Microlensing Survey. I. Bound Planet Detection Rates."
DOI: 10.3847/1538-4365/ab21f0

Expected FITS column schema
----------------------------
This provider reads the first BINTABLE extension of a FITS file with the
following columns.  This schema mirrors the stellar-population columns in the
Penny et al. 2019 simulated source-star catalog:

+-------------------+-------+---------------------------------------------+
| Column name       | Type  | Description                                 |
+===================+=======+=============================================+
| ``galactic_l``    | float | Galactic longitude (°)                      |
+-------------------+-------+---------------------------------------------+
| ``galactic_b``    | float | Galactic latitude (°)                       |
+-------------------+-------+---------------------------------------------+
| ``dist_kpc``      | float | Heliocentric distance (kpc)                 |
+-------------------+-------+---------------------------------------------+
| ``mass``          | float | Stellar mass (M☉)                           |
+-------------------+-------+---------------------------------------------+
| ``teff``          | float | Effective temperature (K)                   |
+-------------------+-------+---------------------------------------------+
| ``logg``          | float | Log surface gravity (dex, CGS)              |
+-------------------+-------+---------------------------------------------+
| ``feh``           | float | Metallicity [Fe/H] (dex)                    |
+-------------------+-------+---------------------------------------------+
| ``mag_F146``      | float | Roman WFI F146 AB magnitude                 |
+-------------------+-------+---------------------------------------------+

Optional columns:

+-------------------+-------+---------------------------------------------+
| ``source_id``     | str   | Catalog row identifier (auto if absent)     |
+-------------------+-------+---------------------------------------------+
| ``tile_id``       | str   | Spatial tile ID (defaults to "roman_bulge") |
+-------------------+-------+---------------------------------------------+

Raises :class:`~smig.catalogs.base.MissingColumnError` on construction if
any required column is absent.  Requires ``astropy`` for FITS I/O.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from smig.catalogs.base import CatalogProvider, MissingColumnError, StarRecord

# ---------------------------------------------------------------------------
# Required FITS column names
# ---------------------------------------------------------------------------

_REQUIRED_COLS: tuple[str, ...] = (
    "galactic_l",
    "galactic_b",
    "dist_kpc",
    "mass",
    "teff",
    "logg",
    "feh",
    "mag_F146",
)

# ---------------------------------------------------------------------------
# Optional astropy import
# ---------------------------------------------------------------------------

_FITS_AVAILABLE: bool = False
_fits: Any = None
try:
    from astropy.io import fits as _fits  # type: ignore[assignment]
    _FITS_AVAILABLE = True
except ImportError:
    pass


class RomanBulgeProvider(CatalogProvider):
    """Roman core-bulge stellar catalog provider (Penny et al. 2019).

    Reads a FITS BinTable and validates the required column schema on
    construction.

    Parameters
    ----------
    catalog_path:
        Path to a FITS file (``.fits`` or ``.fit``).

    Raises
    ------
    MissingColumnError
        If any required column is absent from the FITS table.
    ImportError
        If ``astropy`` is not installed.
    ValueError
        If no BINTABLE extension is found.
    """

    def __init__(self, catalog_path: Path) -> None:
        if not _FITS_AVAILABLE:
            raise ImportError(
                "astropy is required for RomanBulgeProvider. "
                "Install the Phase 2 extras: pip install -e '.[phase2]'"
            )
        self._path = Path(catalog_path)
        self._rows = self._load_fits(self._path)
        self._validate_columns(self._rows)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_fits(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with _fits.open(path) as hdul:
            tbl = next(
                (hdu for hdu in hdul if hdu.name != "PRIMARY"),
                None,
            )
            if tbl is None:
                raise ValueError(f"No BINTABLE extension found in {path}")
            col_names = tbl.columns.names
            for i in range(len(tbl.data)):
                rows.append({c: tbl.data[c][i] for c in col_names})
        return rows

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        present = set(rows[0].keys())
        missing = [c for c in _REQUIRED_COLS if c not in present]
        if missing:
            raise MissingColumnError(missing)

    # ------------------------------------------------------------------
    # CatalogProvider interface
    # ------------------------------------------------------------------

    def sample_field(
        self,
        l_deg: float,
        b_deg: float,
        fov_deg: float,
        rng: np.random.Generator,
    ) -> list[StarRecord]:
        """Return stars within a square FOV centred on ``(l_deg, b_deg)``.

        Filters the pre-loaded catalog to rows within ±fov_deg/2 in both
        Galactic longitude and latitude.

        Parameters
        ----------
        l_deg:
            Field centre Galactic longitude (degrees).
        b_deg:
            Field centre Galactic latitude (degrees).
        fov_deg:
            Square FOV full width (degrees).
        rng:
            Unused for this deterministic ingestor.

        Returns
        -------
        list[StarRecord]
        """
        half = fov_deg / 2.0
        results: list[StarRecord] = []
        for idx, row in enumerate(self._rows):
            rl = float(row["galactic_l"])
            rb = float(row["galactic_b"])
            if abs(rl - l_deg) <= half and abs(rb - b_deg) <= half:
                results.append(self._row_to_star(row, idx))
        return results

    def list_bands(self) -> tuple[str, ...]:
        return ("F146",)

    # ------------------------------------------------------------------
    # Row → StarRecord
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_star(row: dict[str, Any], idx: int) -> StarRecord:
        source_id = str(row.get("source_id", f"roman_bulge_{idx}"))
        tile_id = str(row.get("tile_id", "roman_bulge"))
        return StarRecord(
            galactic_l_deg=float(row["galactic_l"]),
            galactic_b_deg=float(row["galactic_b"]),
            distance_kpc=float(row["dist_kpc"]),
            mass_msun=float(row["mass"]),
            teff_K=float(row["teff"]),
            log_g=float(row["logg"]),
            metallicity_feh=float(row["feh"]),
            mag_F146_ab=float(row["mag_F146"]),
            source_id=source_id,
            catalog_tile_id=tile_id,
        )
