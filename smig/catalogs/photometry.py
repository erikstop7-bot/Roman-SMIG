"""
smig/catalogs/photometry.py
============================
AB-magnitude photometry converter for Phase 3 catalog ingestion.

Single source of truth for flux conversion
------------------------------------------
The F146 AB zero-point is loaded at module import from
``data/catalogs/bandpasses/f146_zero_point.yaml``.  Do NOT hardcode the
numeric value here; update the YAML if the zero-point changes.  The
human-readable provenance mirror is ``docs/phase3_photometry_reference.md``.

Equivalence with Phase 2 ``flux_e`` semantics
----------------------------------------------
``mag_ab_to_electrons(mag_ab, band, exposure_s)`` returns the *total
integrated electron count* over the full exposure, matching the Phase 2
renderer's ``flux_e`` column semantics (the value passed into
``source_params_sequence`` and the ``flux_e`` column of the
:class:`~smig.rendering.crowding.CrowdedFieldRenderer` DataFrame).

Internally:
  ``flux_e_per_s = 10 ** ((ZP - mag_ab) / 2.5)``
  ``flux_e = flux_e_per_s * exposure_s``

where ``ZP = f146_ab_zero_point`` from the YAML.  By definition, a source
with ``mag_ab == ZP`` produces exactly ``exposure_s`` total electrons (i.e.
1 e⁻/s × t_exp).

Equivalence with Phase 2 ``mag_w146``
--------------------------------------
The ``mag_w146`` column in the adapter DataFrame stores the F146 AB magnitude
directly.  This module converts that magnitude to electrons via the zero-point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

# ---------------------------------------------------------------------------
# Locate and load the zero-point YAML relative to the repo root.
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).parents[2]
_ZP_YAML_PATH: Path = _REPO_ROOT / "data" / "catalogs" / "bandpasses" / "f146_zero_point.yaml"

def _load_zero_point() -> float:
    with _ZP_YAML_PATH.open() as fh:
        data = yaml.safe_load(fh)
    zp = data["f146_ab_zero_point"]
    if not isinstance(zp, (int, float)):
        raise TypeError(
            f"f146_ab_zero_point in {_ZP_YAML_PATH} must be numeric, got {type(zp)}"
        )
    return float(zp)


_F146_AB_ZERO_POINT: Final[float] = _load_zero_point()

# ---------------------------------------------------------------------------
# Supported bands (Phase 3.1: F146 only)
# ---------------------------------------------------------------------------

_SUPPORTED_BANDS: frozenset[str] = frozenset({"F146"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_f146_zero_point() -> float:
    """Return the F146 AB zero-point loaded from the YAML file.

    Exposed for tests that need the numeric value without hardcoding it.
    """
    return _F146_AB_ZERO_POINT


def mag_ab_to_electrons(mag_ab: float, band: str, exposure_s: float) -> float:
    """Convert an AB magnitude to total integrated electrons over an exposure.

    This is the Phase 2-compatible ``flux_e`` value: the total number of
    photo-electrons collected over the full exposure, matching the ``flux_e``
    column in :class:`~smig.rendering.crowding.CrowdedFieldRenderer`.

    The conversion uses the AB zero-point definition::

        flux_e_per_s = 10 ** ((ZP - mag_ab) / 2.5)
        flux_e       = flux_e_per_s * exposure_s

    where ``ZP`` is loaded from
    ``data/catalogs/bandpasses/f146_zero_point.yaml``.  By construction,
    ``mag_ab_to_electrons(ZP, "F146", t_exp)`` returns exactly ``t_exp``
    (1 e⁻/s × t_exp).

    Parameters
    ----------
    mag_ab:
        AB magnitude of the source.
    band:
        Photometric band name.  Only ``"F146"`` is supported in Phase 3.1.
        Unknown band names raise :class:`ValueError`.
    exposure_s:
        Exposure time in seconds.

    Returns
    -------
    float
        Total integrated electrons (``flux_e``) over the exposure.

    Raises
    ------
    ValueError
        If ``band`` is not in the Phase 3.1 supported set ``{"F146"}``.
    """
    if band not in _SUPPORTED_BANDS:
        raise ValueError(
            f"Unknown band {band!r}. Supported bands: {sorted(_SUPPORTED_BANDS)}"
        )
    flux_e_per_s = 10.0 ** ((_F146_AB_ZERO_POINT - mag_ab) / 2.5)
    return flux_e_per_s * exposure_s
