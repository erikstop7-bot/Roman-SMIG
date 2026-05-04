"""
smig/catalogs/wcs.py
=====================
Galactic → ICRS → SCA-pixel projection for Phase 3 catalog adapter.

Coordinate chain
----------------
1. Input Galactic (l, b) coordinates via
   :class:`astropy.coordinates.SkyCoord` (``frame='galactic'``).
2. Transform to ICRS (RA, Dec) — uses IAU 1958 definition via astropy.
3. Apply a tangent-plane (gnomonic) projection centred on the field centre
   to compute (ΔRA·cos(Dec₀), ΔDec) offsets in arcseconds.
4. Divide by the plate scale to get pixel offsets; add the stamp centre to
   obtain absolute SCA pixel coordinates.

Plate-scale constant
--------------------
The plate scale of 0.11 arcsec/pixel is imported from
:data:`smig.optics.psf._NATIVE_PIXEL_SCALE_ARCSEC` — the Phase 2
definition that is the single source of truth for this constant.  No
redefinition here.

Pixel origin convention
-----------------------
For Phase 3 datasets the field centre (l, b) maps to **exactly
(128.0, 128.0)**, the centre of the Phase 2 256×256 context stamp.

SCA ID
------
The ``sca_id`` parameter is accepted for interface compatibility with future
per-SCA distortion models.  In Phase 3.1 it is unused; all SCAs share the
same affine projection.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Reuse the Phase 2 plate-scale constant (single source of truth).
# Guard against galsim not being installed — psf.py probes galsim at import
# time but only raises inside STPSFProvider.__init__.
# ---------------------------------------------------------------------------

from smig.optics.psf import _NATIVE_PIXEL_SCALE_ARCSEC  # 0.11 arcsec/pixel

_PLATE_SCALE_ARCSEC_PER_PIX: float = _NATIVE_PIXEL_SCALE_ARCSEC

# Phase 3.1 stamp centre: centre of the 256×256 Phase 2 context stamp.
_STAMP_CENTER_PIX: float = 128.0

# ---------------------------------------------------------------------------
# Optional astropy import — guard for environments without phase2 extras.
# ---------------------------------------------------------------------------

_ASTROPY_AVAILABLE: bool = False
_SkyCoord: Any = None
_units: Any = None

try:
    import astropy.coordinates as _acoords
    import astropy.units as _units  # type: ignore[assignment]
    _SkyCoord = _acoords.SkyCoord
    _ASTROPY_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def galactic_to_sca_pixel(
    l_deg: float,
    b_deg: float,
    sca_id: int | str,
    field_center_l_deg: float,
    field_center_b_deg: float,
) -> tuple[float, float]:
    """Project Galactic coordinates to SCA pixel coordinates.

    Converts ``(l_deg, b_deg)`` to a pixel position within the Phase 3
    256×256 context stamp via a tangent-plane (gnomonic) projection centred
    on ``(field_center_l_deg, field_center_b_deg)``.

    The field centre maps to exactly ``(128.0, 128.0)``.

    Pixel offsets scale linearly with angular offset at the plate scale
    of 0.11 arcsec/pixel for separations up to a few degrees (gnomonic
    approximation).

    Parameters
    ----------
    l_deg:
        Galactic longitude of the star (degrees).
    b_deg:
        Galactic latitude of the star (degrees).
    sca_id:
        SCA identifier.  Unused in Phase 3.1; reserved for future per-SCA
        distortion models.
    field_center_l_deg:
        Galactic longitude of the field centre (degrees).
    field_center_b_deg:
        Galactic latitude of the field centre (degrees).

    Returns
    -------
    tuple[float, float]
        ``(x_pix, y_pix)`` in the 256×256 stamp coordinate system.
        The field centre maps to ``(128.0, 128.0)``.

    Raises
    ------
    ImportError
        If ``astropy`` is not installed.
    """
    if not _ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for galactic_to_sca_pixel. "
            "Install the Phase 2 extras: pip install -e '.[phase2]'"
        )

    # Build SkyCoord objects in Galactic frame.
    star = _SkyCoord(
        l=l_deg * _units.deg, b=b_deg * _units.deg, frame="galactic"
    )
    center = _SkyCoord(
        l=field_center_l_deg * _units.deg,
        b=field_center_b_deg * _units.deg,
        frame="galactic",
    )

    # Transform to ICRS.
    star_icrs = star.icrs
    center_icrs = center.icrs

    # Tangent-plane (gnomonic) offsets in arcseconds.
    # dRA is multiplied by cos(Dec₀) to account for the convergence of
    # meridians at non-zero declination.
    dec0_rad = center_icrs.dec.rad
    dra_arcsec = float(
        ((star_icrs.ra - center_icrs.ra).wrap_at(180 * _units.deg)).to(_units.arcsec).value
        * np.cos(dec0_rad)
    )
    ddec_arcsec = float(
        (star_icrs.dec - center_icrs.dec).to(_units.arcsec).value
    )

    # Convert arcsec offsets to pixel offsets.
    dx_pix = dra_arcsec / _PLATE_SCALE_ARCSEC_PER_PIX
    dy_pix = ddec_arcsec / _PLATE_SCALE_ARCSEC_PER_PIX

    x_pix = _STAMP_CENTER_PIX + dx_pix
    y_pix = _STAMP_CENTER_PIX + dy_pix

    return x_pix, y_pix
