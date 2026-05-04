"""
smig/catalogs/adapter.py
=========================
ProjectedStarTable adapter — converts a list of :class:`~smig.catalogs.base.StarRecord`
objects to a :class:`pandas.DataFrame` with the exact columns required by
:class:`~smig.rendering.crowding.CrowdedFieldRenderer`.

Required columns (from ``smig.rendering.crowding._REQUIRED_COLUMNS``)
----------------------------------------------------------------------
``("x_pix", "y_pix", "flux_e", "mag_w146")``

Column semantics
----------------
- **``x_pix``** (float64): SCA pixel x-coordinate, derived from
  :func:`~smig.catalogs.wcs.galactic_to_sca_pixel`.  Field centre maps to
  ``128.0``.
- **``y_pix``** (float64): SCA pixel y-coordinate.  Field centre maps to
  ``128.0``.
- **``flux_e``** (float64): Total integrated electrons over the exposure,
  computed by :func:`~smig.catalogs.photometry.mag_ab_to_electrons` from
  ``StarRecord.mag_F146_ab``.  Matches Phase 2 ``flux_e`` column semantics.
- **``mag_w146``** (float64): Roman WFI F146 AB magnitude (= ``StarRecord.mag_F146_ab``).
  The column name ``mag_w146`` is the Phase 2 renderer's convention;
  ``mag_w146 ≡ mag_F146_ab`` for all Phase 3 records.

Equivalence note
----------------
``mag_w146`` in the output DataFrame stores the F146 AB magnitude exactly as
held in ``StarRecord.mag_F146_ab``.  The ``flux_e`` column is the authoritative
flux quantity passed to GalSim; ``mag_w146`` is metadata used by the renderer's
brightness-cap filter.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from smig.catalogs.base import StarRecord
from smig.catalogs.photometry import mag_ab_to_electrons
from smig.catalogs.wcs import galactic_to_sca_pixel

# Import _REQUIRED_COLUMNS from the renderer to stay in sync with any future
# changes to that tuple.  This import defines the authoritative column order.
from smig.rendering.crowding import _REQUIRED_COLUMNS as _RENDERER_REQUIRED_COLUMNS

# Pandas is a phase2 dependency; guard for base installs.
_PANDAS_AVAILABLE: bool = False
_pd: Any = None
try:
    import pandas as _pd  # type: ignore[assignment]
    _PANDAS_AVAILABLE = True
except ImportError:
    pass


def project_to_sca_dataframe(
    stars: list[StarRecord],
    sca_id: int,
    field_center_l_deg: float,
    field_center_b_deg: float,
    exposure_s: float,
) -> "Any":  # pandas.DataFrame at runtime
    """Project a list of StarRecords to a CrowdedFieldRenderer-compatible DataFrame.

    Converts Galactic coordinates to SCA pixel positions via
    :func:`~smig.catalogs.wcs.galactic_to_sca_pixel` and AB magnitudes to
    total electrons via :func:`~smig.catalogs.photometry.mag_ab_to_electrons`.

    Parameters
    ----------
    stars:
        Stars to project.  May be empty (returns zero-row DataFrame).
    sca_id:
        SCA identifier passed through to the WCS projector.
    field_center_l_deg:
        Galactic longitude of the field centre (degrees).  Maps to x=128.0.
    field_center_b_deg:
        Galactic latitude of the field centre (degrees).  Maps to y=128.0.
    exposure_s:
        Exposure time in seconds used to convert mag → flux_e.

    Returns
    -------
    pandas.DataFrame
        Columns exactly equal to
        ``smig.rendering.crowding._REQUIRED_COLUMNS``
        = ``("x_pix", "y_pix", "flux_e", "mag_w146")`` with float64 dtype.

    Raises
    ------
    ImportError
        If ``pandas`` or ``astropy`` is not installed.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for project_to_sca_dataframe. "
            "Install the Phase 2 extras: pip install -e '.[phase2]'"
        )

    n = len(stars)
    x_arr = np.empty(n, dtype=np.float64)
    y_arr = np.empty(n, dtype=np.float64)
    flux_arr = np.empty(n, dtype=np.float64)
    mag_arr = np.empty(n, dtype=np.float64)

    for i, star in enumerate(stars):
        x_pix, y_pix = galactic_to_sca_pixel(
            star.galactic_l_deg,
            star.galactic_b_deg,
            sca_id,
            field_center_l_deg,
            field_center_b_deg,
        )
        x_arr[i] = x_pix
        y_arr[i] = y_pix
        flux_arr[i] = mag_ab_to_electrons(star.mag_F146_ab, "F146", exposure_s)
        mag_arr[i] = star.mag_F146_ab

    # Build DataFrame with columns in exactly _RENDERER_REQUIRED_COLUMNS order.
    df = _pd.DataFrame(
        {
            "x_pix": x_arr,
            "y_pix": y_arr,
            "flux_e": flux_arr,
            "mag_w146": mag_arr,
        }
    )
    # Enforce column order to match _REQUIRED_COLUMNS exactly.
    return df[list(_RENDERER_REQUIRED_COLUMNS)]
