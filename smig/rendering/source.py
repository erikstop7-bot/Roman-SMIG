"""
smig/rendering/source.py
========================
Phase 2 source-profile rendering for the SMIG v2 pipeline.

Provides :class:`FiniteSourceRenderer`, which renders a single microlensing
source (point source or finite limb-darkened disk) by convolving a base
profile with a caller-supplied PSF and drawing into a pre-allocated GalSim
Image stamp.

Phase 2 module — requires ``galsim``.
Install with: ``pip install -e '.[phase2]'``.

Design notes
------------
* The PSF is **always provided by the caller per call** — no PSF provider is
  stored on the instance.  This keeps the renderer stateless with respect to
  optics and makes it straightforward to swap PSFs across render calls.
* Resolved sources use ``galsim.Sersic(n=1, ...)`` (exponential disk) as a
  limb-darkening approximation when coefficients are provided, and a uniform
  ``galsim.TopHat`` otherwise.  Full non-parametric limb-darkening is a
  Phase 3 deliverable.
* The stamp buffer is owned by the caller.  This method adds flux to
  whatever is already in the buffer (``add_to_image=True``), so callers can
  accumulate multiple sources on a single stamp.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import galsim as _galsim_type

_GALSIM_AVAILABLE: bool = False
_galsim: Any = None
try:
    import galsim as _galsim  # type: ignore[assignment]
    _GALSIM_AVAILABLE = True
except ImportError:
    pass


class FiniteSourceRenderer:
    """Render a single source profile into a GalSim Image stamp.

    Supports two rendering regimes based on the projected stellar radius:

    * **Unresolved** (``rho_star_arcsec <= UNRESOLVED_THRESHOLD_ARCSEC``):
      base profile is a :class:`galsim.DeltaFunction`.
    * **Resolved** (``rho_star_arcsec > UNRESOLVED_THRESHOLD_ARCSEC``):
      base profile is an exponential Sersic (n=1) if limb-darkening
      coefficients are provided, or a uniform :class:`galsim.TopHat`
      otherwise.

    The base profile is convolved with the caller-supplied PSF and drawn
    in-place into the provided stamp buffer.
    """

    #: Projected stellar radius threshold below which a source is treated as
    #: a point source (~3 Roman WFI native pixels at 0.11 arcsec/pixel).
    UNRESOLVED_THRESHOLD_ARCSEC: float = 0.33

    def __init__(self) -> None:
        if not _GALSIM_AVAILABLE:
            raise ImportError(
                "galsim is required for FiniteSourceRenderer. "
                "Install the Phase 2 extras: pip install -e '.[phase2]'"
            )

    def render_source(
        self,
        flux_e: float,
        centroid_offset_pix: tuple[float, float],
        rho_star_arcsec: float,
        limb_darkening_coeffs: tuple[float, float] | None,
        psf: "_galsim_type.InterpolatedImage",
        stamp: "_galsim_type.Image",
    ) -> None:
        """Render a single source into *stamp* in-place.

        Parameters
        ----------
        flux_e:
            Total source flux in electrons.
        centroid_offset_pix:
            ``(dx, dy)`` offset from the stamp centre in pixels.  Positive x
            is right, positive y is up (standard GalSim convention).
        rho_star_arcsec:
            Projected stellar radius in arcseconds.  Values ≤
            ``UNRESOLVED_THRESHOLD_ARCSEC`` are treated as point sources.
        limb_darkening_coeffs:
            Two-parameter limb-darkening coefficients ``(u1, u2)``, or
            ``None``.  When provided and the source is resolved, a Sersic
            n=1 (exponential disk) profile is used as an approximation.
            When ``None``, a uniform :class:`galsim.TopHat` is used.
        psf:
            Caller-supplied polychromatic PSF as a
            :class:`galsim.InterpolatedImage`.  Its pixel scale must match
            *stamp*.
        stamp:
            Pre-allocated :class:`galsim.Image` buffer.  The method adds
            rendered flux to existing pixel values (``add_to_image=True``).
            Must carry a simple pixel-scale WCS (accessible via
            ``stamp.scale``).

        Returns
        -------
        None
            Draws into *stamp* in-place; returns nothing.
        """
        # --- Input validation ---
        if not np.isfinite(flux_e) or flux_e < 0.0:
            raise ValueError(
                f"flux_e must be finite and non-negative; got {flux_e!r}."
            )
        dx_off, dy_off = centroid_offset_pix
        if not np.isfinite(dx_off) or not np.isfinite(dy_off):
            raise ValueError(
                f"centroid_offset_pix must be finite; got {centroid_offset_pix!r}."
            )
        if not np.isfinite(rho_star_arcsec) or rho_star_arcsec < 0.0:
            raise ValueError(
                f"rho_star_arcsec must be finite and non-negative; got {rho_star_arcsec!r}."
            )

        pixel_scale: float = stamp.scale  # arcsec / pixel

        # --- Choose base profile based on source size ---
        if rho_star_arcsec <= self.UNRESOLVED_THRESHOLD_ARCSEC:
            profile = _galsim.DeltaFunction(flux=flux_e)
        else:
            if limb_darkening_coeffs is not None:
                # Sersic n=1 (exponential disk) as a limb-darkening
                # approximation.  Full non-parametric LD is Phase 3.
                profile = _galsim.Sersic(
                    n=1,
                    half_light_radius=rho_star_arcsec,
                    flux=flux_e,
                )
            else:
                profile = _galsim.TopHat(
                    radius=rho_star_arcsec,
                    flux=flux_e,
                )

        # --- Convolve with the caller-supplied PSF ---
        convolved = _galsim.Convolve([profile, psf])

        # --- Apply centroid offset (pixels → arcseconds) ---
        dx, dy = centroid_offset_pix
        shifted = convolved.shift(dx * pixel_scale, dy * pixel_scale)

        # --- Draw into stamp, accumulating onto existing pixels ---
        shifted.drawImage(image=stamp, add_to_image=True, method="auto")
