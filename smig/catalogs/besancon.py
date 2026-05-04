"""
smig/catalogs/besancon.py
==========================
Besançon Model catalog ingestor for Phase 3.

Supports CSV and FITS output from the Besançon web-form query at
https://model.obs-besancon.fr/.

Expected column schema
----------------------
This provider expects the following column names in the input file.  The
Besançon web form can produce many output columns; the user must ensure
these columns are selected (or the raw output is pre-processed to produce
them):

**Required columns (CSV or FITS BinTable)**:

+------------------+-------+---------------------------------------------+
| Column name      | Type  | Description                                 |
+==================+=======+=============================================+
| ``l_deg``        | float | Galactic longitude (°)                      |
+------------------+-------+---------------------------------------------+
| ``b_deg``        | float | Galactic latitude (°)                       |
+------------------+-------+---------------------------------------------+
| ``dist_kpc``     | float | Heliocentric distance (kpc)                 |
+------------------+-------+---------------------------------------------+
| ``mass_msun``    | float | Stellar mass (M☉)                           |
+------------------+-------+---------------------------------------------+
| ``teff_K``       | float | Effective temperature (K)                   |
+------------------+-------+---------------------------------------------+
| ``log_g``        | float | Log surface gravity (dex, CGS)              |
+------------------+-------+---------------------------------------------+
| ``feh``          | float | Metallicity [Fe/H] (dex)                    |
+------------------+-------+---------------------------------------------+
| ``mag_F146_ab``  | float | Roman WFI F146 AB magnitude                 |
+------------------+-------+---------------------------------------------+

Optional columns:

+------------------+-------+---------------------------------------------+
| ``source_id``    | str   | Catalog-specific ID (auto-generated if      |
|                  |       | absent)                                     |
+------------------+-------+---------------------------------------------+
| ``tile_id``      | str   | Spatial tile identifier (defaults to        |
|                  |       | ``"besancon"`` if absent)                   |
+------------------+-------+---------------------------------------------+

File format
-----------
- **CSV**: Plain comma-separated, header on row 0.  Lines beginning with
  ``#`` are treated as comments and skipped.
- **FITS**: Binary table (``BINTABLE``).  The first ``BINTABLE`` extension
  is read.

Raises :class:`~smig.catalogs.base.MissingColumnError` on construction if
any required column is absent from the file.
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from smig.catalogs.base import CatalogProvider, MissingColumnError, StarRecord

# ---------------------------------------------------------------------------
# Required column names
# ---------------------------------------------------------------------------

_REQUIRED_COLS: tuple[str, ...] = (
    "l_deg",
    "b_deg",
    "dist_kpc",
    "mass_msun",
    "teff_K",
    "log_g",
    "feh",
    "mag_F146_ab",
)

# ---------------------------------------------------------------------------
# Optional astropy/fits import for FITS support
# ---------------------------------------------------------------------------

_FITS_AVAILABLE: bool = False
_fits: Any = None
try:
    from astropy.io import fits as _fits  # type: ignore[assignment]
    _FITS_AVAILABLE = True
except ImportError:
    pass


class BesanconProvider(CatalogProvider):
    """Besançon Model catalog provider.

    Loads a CSV or FITS file produced by the Besançon web-form query and
    validates the required column schema on construction.

    Parameters
    ----------
    catalog_path:
        Path to a CSV (``.csv``) or FITS (``.fits``, ``.fit``) file.

    Raises
    ------
    MissingColumnError
        If any required column is absent from the file.
    ImportError
        If a FITS file is supplied but ``astropy`` is not installed.
    ValueError
        If the file extension is not recognised.
    """

    def __init__(self, catalog_path: Path) -> None:
        self._path = Path(catalog_path)
        suffix = self._path.suffix.lower()
        if suffix == ".csv":
            self._rows = self._load_csv(self._path)
        elif suffix in (".fits", ".fit"):
            self._rows = self._load_fits(self._path)
        else:
            raise ValueError(
                f"Unsupported file extension {suffix!r}. Expected '.csv' or '.fits'."
            )
        self._validate_columns(self._rows)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(newline="") as fh:
            # Skip comment lines starting with '#'.
            lines = (line for line in fh if not line.lstrip().startswith("#"))
            reader = csv.DictReader(lines)
            for row in reader:
                rows.append(dict(row))
        return rows

    @staticmethod
    def _load_fits(path: Path) -> list[dict[str, Any]]:
        if not _FITS_AVAILABLE:
            raise ImportError(
                "astropy is required to load FITS catalogs. "
                "Install the Phase 2 extras: pip install -e '.[phase2]'"
            )
        rows: list[dict[str, Any]] = []
        with _fits.open(path) as hdul:
            # Find the first BINTABLE extension.
            tbl = next(
                (hdu for hdu in hdul if hdu.name != "PRIMARY"),
                None,
            )
            if tbl is None:
                raise ValueError(f"No BINTABLE extension found in {path}")
            col_names = tbl.columns.names
            n_rows = len(tbl.data)
            for i in range(n_rows):
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
            rl = float(row["l_deg"])
            rb = float(row["b_deg"])
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
        source_id = str(row.get("source_id", f"besancon_{idx}"))
        tile_id = str(row.get("tile_id", "besancon"))
        return StarRecord(
            galactic_l_deg=float(row["l_deg"]),
            galactic_b_deg=float(row["b_deg"]),
            distance_kpc=float(row["dist_kpc"]),
            mass_msun=float(row["mass_msun"]),
            teff_K=float(row["teff_K"]),
            log_g=float(row["log_g"]),
            metallicity_feh=float(row["feh"]),
            mag_F146_ab=float(row["mag_F146_ab"]),
            source_id=source_id,
            catalog_tile_id=tile_id,
        )
