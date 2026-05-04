"""
smig/optics/validation/test_psf_regression.py
==============================================
Fast deterministic guardrails against PSF support clipping, rectangular FFT
artefacts, and excessive ringing/negativity introduced by aggressive runtime
optimizations (smaller fov_native_pixels, lower oversample, fewer wavelengths).

These tests run through the analytic Airy+Gaussian backend so they require
``galsim`` (mandatory) but not ``webbpsf`` (optional).  WebbPSF-specific
behaviour is covered separately in ``test_psf.py::TestWebbPSFBackend``.

Each test pins one regression at a time:

  * Severe rim/border energy: tapered small-FOV PSFs do not pile flux at the
    array boundary even after renormalization.
  * Support clipping: encircled-energy at the radius of the candidate FOV
    matches the larger-support analytic reference within a small tolerance.
  * Rectangular FFT artefacts: outer-rim quadrants stay symmetric; no axis
    direction carries more rim energy than its 90° rotation.
  * Excessive ringing or negativity: jitter convolution under reflect-mode
    blur cannot drive any pixel below a small negative tolerance.

Run from the project root::

    python3 -m pytest smig/optics/validation/test_psf_regression.py -v
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from smig.config.optics_schemas import PSFConfig

# ---------------------------------------------------------------------------
# galsim is required for STPSFProvider construction
# ---------------------------------------------------------------------------
galsim = pytest.importorskip(
    "galsim", reason="galsim required for STPSFProvider construction"
)

from smig.optics.psf import STPSFProvider  # noqa: E402  (after importorskip)


# ---------------------------------------------------------------------------
# Constants — pinned thresholds
# ---------------------------------------------------------------------------

# One field position is sufficient: rim/clipping artefacts are dominated by
# the array boundary geometry, not the field-position aberration term.
_FIELD_POS: tuple[float, float] = (0.5, 0.5)
_SCA_ID: int = 1
_WAVELENGTH_UM: float = 1.5

# Reference FOV is at least 2× the candidate so the candidate's outermost
# pixels still fall inside the reference's well-sampled core.
_REF_FOV_MULTIPLIER: int = 2

# Encircled-energy radius (in candidate-PSF native pixels) at which the two
# arrays are compared.  Conservative: the candidate's half-width minus a
# 2-pixel margin so we sample inside its support, not at the boundary.
_EE_MARGIN_PIXELS: int = 2

# Maximum acceptable encircled-energy drift between candidate and reference.
# 5 % is loose enough to accept second-moment differences from finite-FOV
# diffraction-pattern truncation but tight enough to flag gross clipping.
_EE_DRIFT_TOLERANCE: float = 0.05

# Maximum allowed fraction of total PSF energy in the outermost rim pixel.
# A well-tapered PSF with adequate FOV puts << 1 % of its flux in the rim.
_RIM_ENERGY_FRACTION_MAX: float = 0.02

# Maximum azimuthal asymmetry of rim energy (max/min over four quadrants).
# Pure rectangular FFT clipping shows up as one axis carrying ~all the rim
# energy; a 5× asymmetry is very forgiving but still catches that pathology.
_RIM_QUADRANT_ASYMMETRY_MAX: float = 5.0

# Maximum allowed negative pixel value as a fraction of the PSF peak.
# Reflect-mode jitter introduces tiny (~1e-7 relative) FP residuals at the
# wings; constant-mode (the pre-fix path) used to introduce ~1e-3 relative
# axial ringing — this threshold catches the bad regime without false-firing
# on harmless interpolation noise.
_NEGATIVITY_RELATIVE_TOLERANCE: float = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_provider(
    fov: int, oversample: int = 2, taper: float | None = 4.0
) -> STPSFProvider:
    """Return a fast analytic STPSFProvider with the candidate FOV.

    Defaults match the fast_dev profile envelope (small oversample, taper
    enabled) so the regression tests exercise the realistic optimization path.
    """
    cfg = PSFConfig(
        n_wavelengths=2,
        oversample=oversample,
        jitter_rms_mas=0.0,
        fov_native_pixels=fov,
        psf_edge_taper_pixels=taper,
    )
    return STPSFProvider(cfg)


def _reference_provider(candidate_fov: int, oversample: int = 2) -> STPSFProvider:
    """Larger-support reference PSF for encircled-energy comparison.

    Same oversample so the per-pixel scale matches; FOV is multiplied so the
    candidate's footprint sits well inside the reference array.  No taper —
    we want the reference to expose what the untruncated PSF looks like.
    """
    cfg = PSFConfig(
        n_wavelengths=2,
        oversample=oversample,
        jitter_rms_mas=0.0,
        fov_native_pixels=candidate_fov * _REF_FOV_MULTIPLIER,
        psf_edge_taper_pixels=None,
    )
    return STPSFProvider(cfg)


def _encircled_energy(arr: np.ndarray, radius_pixels: float) -> float:
    """Sum of array values within a circular aperture centred on the array."""
    h, w = arr.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[:h, :w]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = r2 <= radius_pixels ** 2
    return float(arr[mask].sum())


def _rim_energy_fraction(arr: np.ndarray, rim_width: int = 1) -> float:
    """Sum of pixels in the outermost ``rim_width``-px border, normalized."""
    mask = np.zeros(arr.shape, dtype=bool)
    mask[:rim_width, :] = True
    mask[-rim_width:, :] = True
    mask[:, :rim_width] = True
    mask[:, -rim_width:] = True
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    return float(arr[mask].sum()) / total


def _quadrant_rim_energies(arr: np.ndarray, rim_width: int = 2) -> tuple[float, float, float, float]:
    """Sum of rim pixels in each (top, bottom, left, right) edge band."""
    h, w = arr.shape
    top = float(arr[:rim_width, :].sum())
    bottom = float(arr[-rim_width:, :].sum())
    left = float(arr[:, :rim_width].sum())
    right = float(arr[:, -rim_width:].sum())
    return top, bottom, left, right


# ---------------------------------------------------------------------------
# Backend availability gate
# ---------------------------------------------------------------------------


def _require_analytic_backend(provider: STPSFProvider) -> None:
    """Skip the test if WebbPSF kicked in — analytic backend is the contract."""
    if provider._backend != "analytic":
        pytest.skip(
            "regression test pins analytic Airy+Gaussian behaviour; "
            "WebbPSF backend has its own test class in test_psf.py."
        )


# ---------------------------------------------------------------------------
# Rim-energy guardrail: small-FOV tapered PSF must not pile flux at the edge
# ---------------------------------------------------------------------------


def test_small_fov_tapered_psf_rim_energy_bounded() -> None:
    """Even at the smallest analytic FOV the outermost rim carries < 2 % of flux.

    A failure here would mean an aggressive fov_native_pixels reduction is
    truncating real PSF support and the renormalization step is artificially
    pushing flux back into the rim.  The cosine taper exists specifically to
    prevent that — this test guards against accidentally disabling it in
    fast_dev profiles.
    """
    provider = _candidate_provider(fov=32, taper=4.0)
    _require_analytic_backend(provider)

    arr = provider.get_psf_at_wavelength(_SCA_ID, _FIELD_POS, _WAVELENGTH_UM)

    rim_frac = _rim_energy_fraction(arr, rim_width=1)
    assert rim_frac < _RIM_ENERGY_FRACTION_MAX, (
        f"PSF rim energy fraction {rim_frac:.4f} exceeds limit "
        f"{_RIM_ENERGY_FRACTION_MAX:.4f} — small-FOV taper may have failed."
    )


def test_psf_no_negative_pixels_after_jitter() -> None:
    """Reflect-mode jitter must not introduce ringing-style negative pixels.

    Tests the polychromatic + jitter path through ``get_psf`` with a moderate
    20 mas jitter, which drives the gaussian_filter near the array boundary.
    Reflect-mode boundary handling should preserve sign; constant-mode (the
    pre-fix behaviour) used to produce ~1e-3 negatives at the rim.
    """
    cfg = PSFConfig(
        n_wavelengths=2,
        oversample=2,
        jitter_rms_mas=20.0,
        fov_native_pixels=32,
        psf_edge_taper_pixels=4.0,
    )
    provider = STPSFProvider(cfg)
    _require_analytic_backend(provider)

    result = provider.get_psf(_SCA_ID, _FIELD_POS, jitter_seed=42)
    if not hasattr(result, "image"):
        pytest.skip("Cannot extract array on this GalSim version.")
    arr = result.image.array

    assert np.all(np.isfinite(arr)), "PSF array contains non-finite values."
    peak = float(arr.max())
    threshold = -_NEGATIVITY_RELATIVE_TOLERANCE * peak
    assert arr.min() >= threshold, (
        f"PSF minimum {arr.min():.3e} below {_NEGATIVITY_RELATIVE_TOLERANCE:.0e}× "
        f"peak={peak:.3e} (threshold {threshold:.3e}) — reflect-mode jitter "
        "may have regressed to constant-mode ringing."
    )


# ---------------------------------------------------------------------------
# Rectangular-FFT artefact guardrail
# ---------------------------------------------------------------------------


def test_psf_rim_quadrant_symmetry() -> None:
    """No edge band carries more than 5× the energy of the lightest band.

    A clipped rectangular FFT typically dumps energy into a single axis; that
    asymmetry is what makes the artefact visually obvious in rendered stamps.
    A centred PSF on a square grid should distribute the (small) rim energy
    roughly evenly across the four edges.
    """
    provider = _candidate_provider(fov=32, taper=4.0)
    _require_analytic_backend(provider)

    arr = provider.get_psf_at_wavelength(_SCA_ID, _FIELD_POS, _WAVELENGTH_UM)
    quads = _quadrant_rim_energies(arr, rim_width=2)

    # Use a small floor to keep the ratio finite when an edge is exactly zero
    # (which happens with the cosine taper at width >= rim_width).
    floor = max(max(quads) * 1e-12, 1e-30)
    ratio = max(quads) / max(min(quads), floor)
    assert ratio < _RIM_QUADRANT_ASYMMETRY_MAX, (
        f"PSF rim quadrant ratio {ratio:.2f} exceeds {_RIM_QUADRANT_ASYMMETRY_MAX:.1f}× "
        f"— possible rectangular FFT artefact.  Quadrant sums: {quads}."
    )


# ---------------------------------------------------------------------------
# Encircled-energy drift vs larger-support analytic reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_fov", [32, 48, 64])
def test_encircled_energy_matches_reference(candidate_fov: int) -> None:
    """Candidate and 2× reference PSFs match in encircled energy to within 5%.

    The reference is twice the candidate FOV so any flux clipped by the
    candidate's smaller boundary is captured by the reference.  We compare
    encircled energy at a radius safely inside the candidate's support so
    both arrays sample the same physical region.

    A failure here is the most direct evidence that fov_native_pixels has
    been pushed too small — the candidate is silently losing real PSF wings.
    """
    cand_prov = _candidate_provider(fov=candidate_fov, taper=4.0)
    ref_prov = _reference_provider(candidate_fov)
    _require_analytic_backend(cand_prov)
    _require_analytic_backend(ref_prov)

    cand_arr = cand_prov.get_psf_at_wavelength(_SCA_ID, _FIELD_POS, _WAVELENGTH_UM)
    ref_arr = ref_prov.get_psf_at_wavelength(_SCA_ID, _FIELD_POS, _WAVELENGTH_UM)

    # Both arrays use the same oversample → same per-pixel scale.  Pick a
    # radius well inside the candidate so we measure where both PSFs are
    # densely sampled.
    cand_half = cand_arr.shape[0] / 2.0
    radius = max(cand_half - _EE_MARGIN_PIXELS, 1.0)

    ee_cand = _encircled_energy(cand_arr, radius_pixels=radius)
    ee_ref = _encircled_energy(ref_arr, radius_pixels=radius)

    rel_drift = abs(ee_cand - ee_ref) / max(ee_ref, 1e-30)
    assert rel_drift < _EE_DRIFT_TOLERANCE, (
        f"Encircled-energy drift {rel_drift:.4f} at radius={radius:.1f} px "
        f"exceeds tolerance {_EE_DRIFT_TOLERANCE:.4f} for candidate_fov={candidate_fov} "
        f"(EE candidate={ee_cand:.6f}, EE reference={ee_ref:.6f})."
    )
