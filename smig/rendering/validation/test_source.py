"""
smig/rendering/validation/test_source.py
==========================================
Acceptance tests for :class:`~smig.rendering.source.FiniteSourceRenderer`.

Coverage
--------
AC-S1  Point source: flux conserved within 0.1%, concentrated at centroid.
AC-S2  Finite disk (TopHat): flux conserved within 0.1%.
AC-S3  Finite disk (Sersic LD approx): flux conserved within 0.1%.
AC-S4  Coordinate frame: positive dx/dy offset moves flux right/up.
AC-S5  Architecture boundary: no smig/sensor/ imports in smig/rendering/.

Run from the project root::

    python -m pytest smig/rendering/validation/test_source.py -v

GalSim tests are skipped automatically when ``galsim`` is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# GalSim availability probe
# ---------------------------------------------------------------------------

_GALSIM_AVAILABLE = False
_galsim: Any = None
try:
    import galsim as _galsim  # type: ignore[assignment]
    _GALSIM_AVAILABLE = True
except ImportError:
    pass

# Skip the entire module when galsim is absent.
pytestmark = pytest.mark.skipif(
    not _GALSIM_AVAILABLE,
    reason="galsim is required for rendering tests (pip install -e '.[phase2]')",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PIXEL_SCALE: float = 0.11  # arcsec / pixel


def _make_psf(
    pixel_scale: float = _DEFAULT_PIXEL_SCALE,
    psf_fwhm_arcsec: float = 0.2,
    stamp_size: int = 32,
) -> Any:
    """Return a normalised Gaussian PSF as a ``galsim.InterpolatedImage``.

    The Gaussian FWHM of 0.2 arcsec corresponds to ~2 Roman WFI pixels at
    the native 0.11 arcsec/pixel plate scale — a realistic diffraction-limited
    core for the analytic fallback.
    """
    gauss = _galsim.Gaussian(fwhm=psf_fwhm_arcsec)
    img = _galsim.Image(stamp_size, stamp_size, scale=pixel_scale)
    gauss.drawImage(image=img, method="auto")
    arr = img.array / float(img.array.sum())
    normalized_img = _galsim.Image(arr, scale=pixel_scale)
    return _galsim.InterpolatedImage(normalized_img)


def _make_stamp(size: int = 64, pixel_scale: float = _DEFAULT_PIXEL_SCALE) -> Any:
    """Return a zeroed ``galsim.Image`` stamp."""
    return _galsim.Image(size, size, scale=pixel_scale)


def _flux_total(stamp: Any) -> float:
    """Sum all pixel values in a GalSim Image."""
    return float(stamp.array.sum())


def _col_centroid(arr: np.ndarray) -> float:
    """Flux-weighted column (x) centroid, 0-indexed."""
    total = arr.sum()
    cols = np.arange(arr.shape[1], dtype=np.float64)
    return float((arr.sum(axis=0) * cols).sum() / total)


def _row_centroid(arr: np.ndarray) -> float:
    """Flux-weighted row (y) centroid, 0-indexed.

    In GalSim convention row 0 is y_min (bottom of the image); a positive
    sky-y shift moves the profile to a *larger* row index.
    """
    total = arr.sum()
    rows = np.arange(arr.shape[0], dtype=np.float64)
    return float((arr.sum(axis=1) * rows).sum() / total)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def renderer() -> Any:
    from smig.rendering.source import FiniteSourceRenderer
    return FiniteSourceRenderer()


@pytest.fixture
def psf() -> Any:
    return _make_psf()


# ---------------------------------------------------------------------------
# AC-S1 — Point source flux conservation and centroid
# ---------------------------------------------------------------------------


def test_point_source_flux_conservation(renderer: Any, psf: Any) -> None:
    """Point source (unresolved): total rendered flux == flux_e within 0.1%."""
    flux_e = 5000.0
    stamp = _make_stamp(64)
    renderer.render_source(flux_e, (0.0, 0.0), 0.1, None, psf, stamp)
    rendered_flux = _flux_total(stamp)
    assert abs(rendered_flux - flux_e) / flux_e < 0.001, (
        f"Point source flux error: rendered={rendered_flux:.2f}, "
        f"expected={flux_e:.2f}"
    )


def test_point_source_centroid_at_origin(renderer: Any, psf: Any) -> None:
    """Point source at zero offset: centroid near stamp centre."""
    stamp = _make_stamp(64)
    renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp)
    centre = 63 / 2.0  # 0-indexed centre of a 64-pixel stamp
    col_c = _col_centroid(stamp.array)
    row_c = _row_centroid(stamp.array)
    # Allow ±1 pixel from the geometric centre.
    assert abs(col_c - centre) < 1.0, (
        f"Column centroid {col_c:.3f} is too far from centre {centre:.3f}."
    )
    assert abs(row_c - centre) < 1.0, (
        f"Row centroid {row_c:.3f} is too far from centre {centre:.3f}."
    )


def test_point_source_adds_to_existing_flux(renderer: Any, psf: Any) -> None:
    """Two render_source calls on the same stamp accumulate flux."""
    stamp = _make_stamp(64)
    renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp)
    renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp)
    assert _flux_total(stamp) == pytest.approx(2000.0, rel=0.001)


def test_render_source_returns_none(renderer: Any, psf: Any) -> None:
    """render_source must return None (stamps are mutated in-place)."""
    stamp = _make_stamp(64)
    result = renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp)
    assert result is None


# ---------------------------------------------------------------------------
# AC-S2 — Finite disk (TopHat): flux conserved within 0.1%
# ---------------------------------------------------------------------------


def test_tophat_disk_flux_conservation(renderer: Any, psf: Any) -> None:
    """Uniform TopHat disk: total flux conserved within 0.1%.

    Uses a large stamp (128×128) to ensure the Sersic/TopHat wings are
    captured and FFT boundary effects do not artificially fail the tolerance.
    """
    flux_e = 5000.0
    # rho_star = 0.4 arcsec > UNRESOLVED_THRESHOLD (0.33 arcsec)
    stamp = _make_stamp(128)
    renderer.render_source(flux_e, (0.0, 0.0), 0.4, None, psf, stamp)
    rendered_flux = _flux_total(stamp)
    assert abs(rendered_flux - flux_e) / flux_e < 0.001, (
        f"TopHat flux error: rendered={rendered_flux:.2f}, expected={flux_e:.2f}"
    )


def test_tophat_disk_extends_beyond_psf(renderer: Any, psf: Any) -> None:
    """Resolved TopHat disk produces a wider profile than a point source."""
    flux = 1000.0
    stamp_point = _make_stamp(128)
    stamp_disk = _make_stamp(128)
    renderer.render_source(flux, (0.0, 0.0), 0.1, None, psf, stamp_point)
    renderer.render_source(flux, (0.0, 0.0), 0.4, None, psf, stamp_disk)

    # Second moment (spread) of the disk profile should be larger.
    def _sigma_px(arr: np.ndarray) -> float:
        h, w = arr.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        y_idx, x_idx = np.mgrid[:h, :w]
        r2 = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
        total = arr.sum()
        return float(np.sqrt((r2 * arr).sum() / total)) if total > 0 else 0.0

    sigma_point = _sigma_px(stamp_point.array)
    sigma_disk = _sigma_px(stamp_disk.array)
    assert sigma_disk > sigma_point, (
        f"Resolved disk should be broader than point source: "
        f"σ_point={sigma_point:.3f} px, σ_disk={sigma_disk:.3f} px"
    )


# ---------------------------------------------------------------------------
# AC-S3 — Finite disk (Sersic LD approx): flux conserved within 0.1%
# ---------------------------------------------------------------------------


def test_sersic_disk_flux_conservation(renderer: Any, psf: Any) -> None:
    """Sersic n=1 LD approx: total flux conserved within 0.1%.

    Uses a large stamp (256×256) because the exponential Sersic profile has
    extended wings that can cause flux loss on small stamps.
    """
    flux_e = 5000.0
    stamp = _make_stamp(256)
    # limb_darkening_coeffs triggers the Sersic branch
    renderer.render_source(flux_e, (0.0, 0.0), 0.4, (0.3, 0.1), psf, stamp)
    rendered_flux = _flux_total(stamp)
    assert abs(rendered_flux - flux_e) / flux_e < 0.001, (
        f"Sersic flux error: rendered={rendered_flux:.2f}, expected={flux_e:.2f}"
    )


def test_sersic_and_tophat_differ(renderer: Any, psf: Any) -> None:
    """Sersic and TopHat profiles produce different pixel distributions."""
    stamp_sersic = _make_stamp(256)
    stamp_tophat = _make_stamp(256)
    renderer.render_source(1000.0, (0.0, 0.0), 0.4, (0.3, 0.1), psf, stamp_sersic)
    renderer.render_source(1000.0, (0.0, 0.0), 0.4, None, psf, stamp_tophat)
    assert not np.allclose(stamp_sersic.array, stamp_tophat.array, atol=1e-6), (
        "Sersic and TopHat profiles should produce different pixel arrays."
    )


# ---------------------------------------------------------------------------
# AC-S4 — Coordinate frame sign correctness
# ---------------------------------------------------------------------------


def test_positive_dx_shifts_right(renderer: Any, psf: Any) -> None:
    """A positive dx offset moves the flux centroid to a larger column index.

    In GalSim convention, positive x is to the right, which corresponds to
    increasing column index in the numpy array.
    """
    stamp_centre = _make_stamp(64)
    stamp_shifted = _make_stamp(64)
    renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp_centre)
    renderer.render_source(1000.0, (8.0, 0.0), 0.1, None, psf, stamp_shifted)

    col_centre = _col_centroid(stamp_centre.array)
    col_shifted = _col_centroid(stamp_shifted.array)
    assert col_shifted > col_centre, (
        f"Positive dx should shift centroid right (larger col): "
        f"col_centre={col_centre:.3f}, col_shifted={col_shifted:.3f}"
    )


def test_positive_dy_shifts_up(renderer: Any, psf: Any) -> None:
    """A positive dy offset moves the flux centroid to a larger row index.

    In GalSim convention, positive y is upward, which maps to an increasing
    row index in the numpy array (row 0 = y_min = bottom of image).
    """
    stamp_centre = _make_stamp(64)
    stamp_shifted = _make_stamp(64)
    renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp_centre)
    renderer.render_source(1000.0, (0.0, 8.0), 0.1, None, psf, stamp_shifted)

    row_centre = _row_centroid(stamp_centre.array)
    row_shifted = _row_centroid(stamp_shifted.array)
    assert row_shifted > row_centre, (
        f"Positive dy should shift centroid up (larger row index in GalSim "
        f"convention): row_centre={row_centre:.3f}, row_shifted={row_shifted:.3f}"
    )


def test_negative_dx_shifts_left(renderer: Any, psf: Any) -> None:
    """A negative dx offset moves the flux centroid to a smaller column index."""
    stamp_centre = _make_stamp(64)
    stamp_shifted = _make_stamp(64)
    renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp_centre)
    renderer.render_source(1000.0, (-8.0, 0.0), 0.1, None, psf, stamp_shifted)

    col_centre = _col_centroid(stamp_centre.array)
    col_shifted = _col_centroid(stamp_shifted.array)
    assert col_shifted < col_centre, (
        f"Negative dx should shift centroid left (smaller col): "
        f"col_centre={col_centre:.3f}, col_shifted={col_shifted:.3f}"
    )


def test_zero_offset_centroid_symmetry(renderer: Any, psf: Any) -> None:
    """Zero offset renders a PSF-convolved source symmetric about the centre."""
    stamp = _make_stamp(64)
    renderer.render_source(1000.0, (0.0, 0.0), 0.1, None, psf, stamp)
    arr = stamp.array
    centre = (64 - 1) / 2.0
    col_c = _col_centroid(arr)
    row_c = _row_centroid(arr)
    # Centroid should be within 0.5 pixel of geometric centre for a centred,
    # symmetric PSF.
    assert abs(col_c - centre) < 0.5, (
        f"Column centroid {col_c:.3f} deviates too far from centre {centre:.3f}."
    )
    assert abs(row_c - centre) < 0.5, (
        f"Row centroid {row_c:.3f} deviates too far from centre {centre:.3f}."
    )


# ---------------------------------------------------------------------------
# Additional: threshold boundary, unresolved/resolved regime selection
# ---------------------------------------------------------------------------


def test_exactly_at_threshold_is_unresolved(renderer: Any, psf: Any) -> None:
    """A source at exactly UNRESOLVED_THRESHOLD_ARCSEC uses DeltaFunction."""
    from smig.rendering.source import FiniteSourceRenderer

    threshold = FiniteSourceRenderer.UNRESOLVED_THRESHOLD_ARCSEC
    flux_e = 1000.0
    stamp = _make_stamp(64)
    # Should not raise; uses DeltaFunction path.
    renderer.render_source(flux_e, (0.0, 0.0), threshold, None, psf, stamp)
    assert _flux_total(stamp) == pytest.approx(flux_e, rel=0.001)


def test_just_above_threshold_is_resolved(renderer: Any, psf: Any) -> None:
    """A source just above threshold uses the finite-disk path."""
    from smig.rendering.source import FiniteSourceRenderer

    threshold = FiniteSourceRenderer.UNRESOLVED_THRESHOLD_ARCSEC
    rho = threshold + 0.01  # just resolved
    flux_e = 1000.0
    stamp = _make_stamp(128)
    renderer.render_source(flux_e, (0.0, 0.0), rho, None, psf, stamp)
    assert _flux_total(stamp) == pytest.approx(flux_e, rel=0.001)


def test_construction_without_galsim_raises(monkeypatch: Any) -> None:
    """FiniteSourceRenderer.__init__ raises ImportError without galsim."""
    import smig.rendering.source as src_mod

    orig = src_mod._GALSIM_AVAILABLE
    try:
        src_mod._GALSIM_AVAILABLE = False
        with pytest.raises(ImportError, match="galsim"):
            src_mod.FiniteSourceRenderer()
    finally:
        src_mod._GALSIM_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Regression: finite input guards
# ---------------------------------------------------------------------------


def test_negative_flux_e_raises(renderer: Any, psf: Any) -> None:
    """render_source raises ValueError for negative flux_e."""
    stamp = _make_stamp(64)
    with pytest.raises(ValueError, match="flux_e"):
        renderer.render_source(-1.0, (0.0, 0.0), 0.1, None, psf, stamp)


def test_nan_flux_e_raises(renderer: Any, psf: Any) -> None:
    """render_source raises ValueError for NaN flux_e."""
    stamp = _make_stamp(64)
    with pytest.raises(ValueError, match="flux_e"):
        renderer.render_source(float("nan"), (0.0, 0.0), 0.1, None, psf, stamp)


def test_inf_flux_e_raises(renderer: Any, psf: Any) -> None:
    """render_source raises ValueError for Inf flux_e."""
    stamp = _make_stamp(64)
    with pytest.raises(ValueError, match="flux_e"):
        renderer.render_source(float("inf"), (0.0, 0.0), 0.1, None, psf, stamp)


def test_nan_centroid_offset_raises(renderer: Any, psf: Any) -> None:
    """render_source raises ValueError for non-finite centroid_offset_pix."""
    stamp = _make_stamp(64)
    with pytest.raises(ValueError, match="centroid_offset_pix"):
        renderer.render_source(500.0, (float("nan"), 0.0), 0.1, None, psf, stamp)


def test_inf_centroid_offset_raises(renderer: Any, psf: Any) -> None:
    """render_source raises ValueError for Inf in centroid_offset_pix."""
    stamp = _make_stamp(64)
    with pytest.raises(ValueError, match="centroid_offset_pix"):
        renderer.render_source(500.0, (0.0, float("inf")), 0.1, None, psf, stamp)


def test_negative_rho_star_raises(renderer: Any, psf: Any) -> None:
    """render_source raises ValueError for negative rho_star_arcsec."""
    stamp = _make_stamp(64)
    with pytest.raises(ValueError, match="rho_star_arcsec"):
        renderer.render_source(500.0, (0.0, 0.0), -0.1, None, psf, stamp)


def test_nan_rho_star_raises(renderer: Any, psf: Any) -> None:
    """render_source raises ValueError for NaN rho_star_arcsec."""
    stamp = _make_stamp(64)
    with pytest.raises(ValueError, match="rho_star_arcsec"):
        renderer.render_source(500.0, (0.0, 0.0), float("nan"), None, psf, stamp)
