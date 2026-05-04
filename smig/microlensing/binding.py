"""
smig/microlensing/binding.py
============================
Phase 3.3 adapter: bind a MicrolensingEvent + StarRecord to the Phase 2
source_params_sequence contract consumed by SceneSimulator.simulate_event.

CENTERED SOURCE ASSUMPTION
--------------------------
The microlensed source is drawn at the stamp center (centroid_offset_pix is
absent from the returned dicts, so Phase 2 defaults to (0.0, 0.0)). Off-center
microlensed sources are a versioned future extension; the caller must add
centroid_offset_pix explicitly when that is needed.

source_params_sequence[i] dict shape (Phase 2 contract, from pipeline.py)
--------------------------------------------------------------------------
Mandatory:
  flux_e : float
      Total integrated electrons over the exposure window.
      == A(t_i) * mag_ab_to_electrons(source_star.mag_F146_ab, band, exposure_s)

Optional (fall back to Phase 2 defaults if absent):
  centroid_offset_pix : tuple[float, float]
      (dx, dy) offset from stamp centre in pixels.  Default (0.0, 0.0).
  rho_star_arcsec : float
      Projected stellar radius in arcseconds.  Default 0.0 (point source).
  limb_darkening_coeffs : tuple[float, float] | None
      Quadratic LD coefficients (u1, u2), or None.

bind_event_to_source populates ONLY flux_e; optional fields are omitted so
that Phase 2 defaults take effect. The binding layer makes no rendering decisions.
"""
from __future__ import annotations

import numpy as np

from smig.catalogs.base import StarRecord
from smig.catalogs.photometry import mag_ab_to_electrons
from smig.microlensing.event import MicrolensingEvent, SourceProperties


def bind_event_to_source(
    event: MicrolensingEvent,
    source_star: StarRecord,
    epoch_times_mjd: np.ndarray,
    exposure_s: float,
    band: str = "F146",
) -> list[dict]:
    """Produce source_params_sequence matching Phase 2 SceneSimulator.simulate_event.

    flux_e semantics: TOTAL integrated electrons over the exposure window,
    == A(t_i) * mag_ab_to_electrons(source_star.mag_F146_ab, band, exposure_s).

    The microlensed source is drawn at the stamp center. This phase does not
    support off-center microlensed sources; that is a versioned future extension.

    Parameters
    ----------
    event:
        Frozen MicrolensingEvent whose .magnification() computes A(t).
    source_star:
        Stellar source record providing photometry and atmospheric parameters.
    epoch_times_mjd:
        1-D array of epoch times in Modified Julian Date.
    exposure_s:
        Exposure duration in seconds. Must be provided by the caller — not read
        from config inside this function.
    band:
        Photometric band name. Defaults to "F146" (Roman WFI F146).

    Returns
    -------
    list[dict]
        One dict per epoch with key "flux_e" (Python float). Length equals
        len(epoch_times_mjd).
    """
    source_props = SourceProperties(
        teff_K=source_star.teff_K,
        log_g=source_star.log_g,
        metallicity_feh=source_star.metallicity_feh,
        distance_kpc=source_star.distance_kpc,
        mass_msun=source_star.mass_msun,
    )
    t = np.asarray(epoch_times_mjd, dtype=float)
    A_t = event.magnification(t, band, source_props)
    F0_total_e = mag_ab_to_electrons(source_star.mag_F146_ab, band, exposure_s)
    return [{"flux_e": float(A_t[i] * F0_total_e)} for i in range(len(t))]
