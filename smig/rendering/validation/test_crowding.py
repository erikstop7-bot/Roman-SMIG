"""
smig/rendering/validation/test_crowding.py
============================================
Acceptance tests for :class:`~smig.rendering.crowding.CrowdedFieldRenderer`.

Coverage
--------
AC-C1  Returns correct numpy array shape ``(stamp_size, stamp_size)``.
AC-C2  Edge neighbors: peak pixel shifts towards the expected edge.
AC-C3  Static field cached: second call returns the same cached array.
AC-C4  Performance: 200-star stamp renders in < 2 s on CPU.
AC-C5  Catalog validation: missing columns, NaN values, wrong position dtype.
AC-C6  Brightness cap: stars fainter than cap are excluded.

Run from the project root::

    python -m pytest smig/rendering/validation/test_crowding.py -v

GalSim/pandas tests are skipped when Phase 2 extras are absent.
The slow performance test is skipped in CI environments where
``os.environ["CI"]`` is set.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# GalSim / pandas availability probes
# ---------------------------------------------------------------------------

_GALSIM_AVAILABLE = False
_galsim: Any = None
try:
    import galsim as _galsim  # type: ignore[assignment]
    _GALSIM_AVAILABLE = True
except ImportError:
    pass

_PANDAS_AVAILABLE = False
_pd: Any = None
try:
    import pandas as _pd  # type: ignore[assignment]
    _PANDAS_AVAILABLE = True
except ImportError:
    pass

_PHASE2_AVAILABLE = _GALSIM_AVAILABLE and _PANDAS_AVAILABLE

# Skip the entire module when Phase 2 extras are absent.
pytestmark = pytest.mark.skipif(
    not _PHASE2_AVAILABLE,
    reason=(
        "galsim and pandas are required for crowding tests "
        "(pip install -e '.[phase2]')"
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PIXEL_SCALE: float = 0.11  # arcsec / pixel
_DEFAULT_STAMP_SIZE: int = 64


def _make_psf(
    pixel_scale: float = _DEFAULT_PIXEL_SCALE,
    psf_fwhm_arcsec: float = 0.2,
    psf_stamp: int = 32,
) -> Any:
    """Return a normalised Gaussian PSF as a ``galsim.InterpolatedImage``."""
    gauss = _galsim.Gaussian(fwhm=psf_fwhm_arcsec)
    img = _galsim.Image(psf_stamp, psf_stamp, scale=pixel_scale)
    gauss.drawImage(image=img, method="auto")
    arr = img.array / float(img.array.sum())
    return _galsim.InterpolatedImage(_galsim.Image(arr, scale=pixel_scale))


def _make_catalog(
    n_stars: int = 10,
    stamp_center: tuple[float, float] = (512.0, 512.0),
    stamp_size: int = _DEFAULT_STAMP_SIZE,
    seed: int = 42,
    flux_range: tuple[float, float] = (100.0, 2000.0),
    mag_range: tuple[float, float] = (15.0, 25.0),
) -> Any:
    """Build a synthetic uniform-random neighbor catalog.

    Stars are placed within ±(stamp_size/2) pixels of *stamp_center* to
    ensure they all fall within or near the rendered stamp.

    Phase 3 deliverable: replace with Galaxia file-based catalogs.
    """
    rng = np.random.default_rng(seed)
    cx, cy = stamp_center
    half = stamp_size / 2.0
    x_pix = rng.uniform(cx - half, cx + half, size=n_stars).astype(np.float64)
    y_pix = rng.uniform(cy - half, cy + half, size=n_stars).astype(np.float64)
    flux_e = rng.uniform(*flux_range, size=n_stars).astype(np.float64)
    mag_w146 = rng.uniform(*mag_range, size=n_stars).astype(np.float64)
    return _pd.DataFrame(
        {"x_pix": x_pix, "y_pix": y_pix, "flux_e": flux_e, "mag_w146": mag_w146}
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def psf() -> Any:
    return _make_psf()


@pytest.fixture
def small_catalog() -> Any:
    return _make_catalog(n_stars=10)


@pytest.fixture
def renderer(small_catalog: Any) -> Any:
    from smig.rendering.crowding import CrowdedFieldRenderer
    return CrowdedFieldRenderer(small_catalog)


# ---------------------------------------------------------------------------
# AC-C1 — Correct output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_default(renderer: Any, psf: Any) -> None:
    """render_static_field returns ndarray of shape (stamp_size, stamp_size)."""
    result = renderer.render_static_field(psf, (512.0, 512.0))
    assert isinstance(result, np.ndarray), "Result must be a numpy ndarray."
    assert result.shape == (_DEFAULT_STAMP_SIZE, _DEFAULT_STAMP_SIZE), (
        f"Expected shape ({_DEFAULT_STAMP_SIZE}, {_DEFAULT_STAMP_SIZE}), "
        f"got {result.shape}."
    )


def test_output_dtype_float64(renderer: Any, psf: Any) -> None:
    """render_static_field returns float64 array."""
    result = renderer.render_static_field(psf, (512.0, 512.0))
    assert result.dtype == np.float64, (
        f"Expected float64, got {result.dtype}."
    )


def test_output_is_not_galsim_image(renderer: Any, psf: Any) -> None:
    """render_static_field must not return a GalSim Image object."""
    result = renderer.render_static_field(psf, (512.0, 512.0))
    assert type(result).__module__.split(".")[0] != "galsim", (
        "render_static_field must return np.ndarray, not a GalSim Image."
    )


def test_custom_stamp_size() -> None:
    """Output shape matches a non-default stamp_size."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    cat = _make_catalog(n_stars=5, stamp_size=32)
    renderer = CrowdedFieldRenderer(cat, stamp_size=32)
    result = renderer.render_static_field(_make_psf(), (512.0, 512.0))
    assert result.shape == (32, 32)


def test_empty_catalog_after_brightness_cap() -> None:
    """All stars fainter than cap → result is all-zeros array."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    cat = _make_catalog(n_stars=5, mag_range=(22.0, 25.0))
    # Cap at mag 20 — all stars are fainter than 20 → excluded.
    renderer = CrowdedFieldRenderer(cat, brightness_cap_mag=20.0)
    result = renderer.render_static_field(_make_psf(), (512.0, 512.0))
    assert result.shape == (_DEFAULT_STAMP_SIZE, _DEFAULT_STAMP_SIZE)
    assert result.sum() == pytest.approx(0.0, abs=1e-10), (
        "Expected all-zero array when all stars are fainter than cap."
    )


# ---------------------------------------------------------------------------
# AC-C2 — Edge neighbors: peak shifts towards expected edge
# ---------------------------------------------------------------------------


def test_single_star_right_of_centre_peak_shifts_right() -> None:
    """A star placed to the right of the stamp centre shifts the peak right.

    Strict 0.1% total flux conservation is *not* required for edge stars
    because the stamp may truncate flux at the boundary.
    """
    from smig.rendering.crowding import CrowdedFieldRenderer

    stamp_size = 64
    pixel_scale = _DEFAULT_PIXEL_SCALE
    cx, cy = 512.0, 512.0

    # Star placed 20 pixels to the right of the stamp centre.
    dx = 20.0
    cat = _pd.DataFrame({
        "x_pix": [cx + dx],
        "y_pix": [cy],
        "flux_e": [1000.0],
        "mag_w146": [20.0],
    })
    renderer = CrowdedFieldRenderer(cat, stamp_size=stamp_size, pixel_scale=pixel_scale)
    result = renderer.render_static_field(_make_psf(), (cx, cy))

    # Peak column should be in the right half of the stamp.
    peak_col = int(np.argmax(result.sum(axis=0)))
    assert peak_col > stamp_size // 2, (
        f"Star at +20 px from centre: peak column {peak_col} should be "
        f"> {stamp_size // 2} (right half of stamp)."
    )


def test_single_star_above_centre_peak_shifts_up() -> None:
    """A star placed above the stamp centre shifts the peak to a higher row.

    In GalSim convention, positive y is up = larger row index.
    """
    from smig.rendering.crowding import CrowdedFieldRenderer

    stamp_size = 64
    cx, cy = 512.0, 512.0
    dy = 20.0

    cat = _pd.DataFrame({
        "x_pix": [cx],
        "y_pix": [cy + dy],
        "flux_e": [1000.0],
        "mag_w146": [20.0],
    })
    renderer = CrowdedFieldRenderer(cat, stamp_size=stamp_size)
    result = renderer.render_static_field(_make_psf(), (cx, cy))

    peak_row = int(np.argmax(result.sum(axis=1)))
    assert peak_row > stamp_size // 2, (
        f"Star at +20 px from centre: peak row {peak_row} should be "
        f"> {stamp_size // 2} (upper half of stamp in GalSim convention)."
    )


def test_single_star_at_edge_does_not_raise() -> None:
    """A star placed far outside the stamp renders without raising."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    cx, cy = 512.0, 512.0
    # Place star 200 pixels outside the 64-pixel stamp.
    cat = _pd.DataFrame({
        "x_pix": [cx + 200.0],
        "y_pix": [cy],
        "flux_e": [1000.0],
        "mag_w146": [20.0],
    })
    renderer = CrowdedFieldRenderer(cat)
    result = renderer.render_static_field(_make_psf(), (cx, cy))
    # Just assert shape is correct; flux conservation is not required here.
    assert result.shape == (_DEFAULT_STAMP_SIZE, _DEFAULT_STAMP_SIZE)


# ---------------------------------------------------------------------------
# AC-C3 — Static field caching
# ---------------------------------------------------------------------------


def test_cache_hit_returns_copy_with_equal_values(renderer: Any, psf: Any) -> None:
    """Second call with identical args returns an equal array copy, not same object.

    The cache returns copies so callers cannot corrupt cached results in place.
    """
    result1 = renderer.render_static_field(psf, (512.0, 512.0))
    result2 = renderer.render_static_field(psf, (512.0, 512.0))
    assert result1 is not result2, (
        "Cache should return independent copies, not the same object."
    )
    np.testing.assert_array_equal(result1, result2)
    assert len(renderer._static_field_cache) == 1, "Cache should have exactly one entry."


def test_cache_miss_on_different_centre(renderer: Any, psf: Any) -> None:
    """Different stamp centres produce distinct cache entries."""
    result_a = renderer.render_static_field(psf, (512.0, 512.0))
    result_b = renderer.render_static_field(psf, (600.0, 512.0))
    assert result_a is not result_b
    assert len(renderer._static_field_cache) == 2


def test_cache_populates_on_first_call(renderer: Any, psf: Any) -> None:
    """Cache is empty before the first call and has one entry after."""
    assert len(renderer._static_field_cache) == 0
    renderer.render_static_field(psf, (512.0, 512.0))
    assert len(renderer._static_field_cache) == 1


def test_cache_key_includes_stamp_size() -> None:
    """Different stamp sizes produce separate cache entries."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    cat = _make_catalog(n_stars=5)
    r32 = CrowdedFieldRenderer(cat, stamp_size=32)
    r64 = CrowdedFieldRenderer(cat, stamp_size=64)
    psf = _make_psf()
    res32 = r32.render_static_field(psf, (512.0, 512.0))
    res64 = r64.render_static_field(psf, (512.0, 512.0))
    assert res32.shape == (32, 32)
    assert res64.shape == (64, 64)


# ---------------------------------------------------------------------------
# AC-C4 — Performance: 200-star stamp in < 2 s
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_200_star_render_performance() -> None:
    """A 200-star crowded-field stamp renders in < 2 s on CPU.

    The strict timing assertion is skipped in CI environments to prevent
    shared-runner flakes, but the render is always executed to verify
    correctness.
    """
    from smig.rendering.crowding import CrowdedFieldRenderer

    cat = _make_catalog(n_stars=200, seed=0)
    renderer = CrowdedFieldRenderer(cat, stamp_size=64)
    psf = _make_psf()

    t0 = time.perf_counter()
    result = renderer.render_static_field(psf, (512.0, 512.0))
    elapsed = time.perf_counter() - t0

    # Always verify correctness.
    assert result.shape == (64, 64)
    assert result.dtype == np.float64

    # Timing gate: skip in CI to avoid shared-runner flakes.
    in_ci = bool(os.environ.get("CI"))
    if not in_ci:
        assert elapsed < 2.0, (
            f"200-star render took {elapsed:.2f} s (limit: 2.0 s). "
            "Consider profiling galsim.Sum + Convolve path."
        )


# ---------------------------------------------------------------------------
# AC-C5 — Catalog validation
# ---------------------------------------------------------------------------


def test_missing_column_raises() -> None:
    """CrowdedFieldRenderer raises ValueError for a catalog missing a column."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    bad_cat = _pd.DataFrame({
        "x_pix": [512.0],
        "y_pix": [512.0],
        "flux_e": [500.0],
        # 'mag_w146' is absent
    })
    with pytest.raises(ValueError, match="missing required columns"):
        CrowdedFieldRenderer(bad_cat)


def test_nan_value_raises() -> None:
    """CrowdedFieldRenderer raises ValueError when a required column has NaN."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    bad_cat = _pd.DataFrame({
        "x_pix": [512.0, float("nan")],
        "y_pix": [512.0, 512.0],
        "flux_e": [500.0, 500.0],
        "mag_w146": [20.0, 20.0],
    })
    with pytest.raises(ValueError, match="NaN"):
        CrowdedFieldRenderer(bad_cat)


def test_non_float_position_raises() -> None:
    """CrowdedFieldRenderer raises ValueError for integer position columns."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    bad_cat = _pd.DataFrame({
        "x_pix": _pd.array([512, 513], dtype="int64"),  # should be float
        "y_pix": _pd.array([512, 512], dtype="int64"),
        "flux_e": [500.0, 600.0],
        "mag_w146": [20.0, 21.0],
    })
    with pytest.raises(ValueError, match="float dtype"):
        CrowdedFieldRenderer(bad_cat)


def test_valid_catalog_constructs() -> None:
    """A well-formed catalog constructs without error."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    cat = _make_catalog(n_stars=5)
    renderer = CrowdedFieldRenderer(cat)
    assert renderer is not None


def test_construction_without_galsim_raises(monkeypatch: Any) -> None:
    """CrowdedFieldRenderer raises ImportError without galsim."""
    import smig.rendering.crowding as crowd_mod

    orig = crowd_mod._GALSIM_AVAILABLE
    try:
        crowd_mod._GALSIM_AVAILABLE = False
        with pytest.raises(ImportError, match="galsim"):
            crowd_mod.CrowdedFieldRenderer(_make_catalog())
    finally:
        crowd_mod._GALSIM_AVAILABLE = orig


def test_construction_without_pandas_raises(monkeypatch: Any) -> None:
    """CrowdedFieldRenderer raises ImportError without pandas."""
    import smig.rendering.crowding as crowd_mod

    orig = crowd_mod._PANDAS_AVAILABLE
    try:
        crowd_mod._PANDAS_AVAILABLE = False
        with pytest.raises(ImportError, match="pandas"):
            crowd_mod.CrowdedFieldRenderer(_make_catalog())
    finally:
        crowd_mod._PANDAS_AVAILABLE = orig


# ---------------------------------------------------------------------------
# AC-C6 — Brightness cap
# ---------------------------------------------------------------------------


def test_brightness_cap_excludes_faint_stars() -> None:
    """Stars with mag_w146 > brightness_cap_mag contribute zero flux."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    cx, cy = 512.0, 512.0
    # One bright star (mag 18) + one faint star (mag 24).
    cat = _pd.DataFrame({
        "x_pix": [cx, cx],
        "y_pix": [cy, cy],
        "flux_e": [2000.0, 500.0],
        "mag_w146": [18.0, 24.0],
    })
    psf = _make_psf()
    renderer_no_cap = CrowdedFieldRenderer(cat)
    renderer_capped = CrowdedFieldRenderer(cat, brightness_cap_mag=20.0)

    result_no_cap = renderer_no_cap.render_static_field(psf, (cx, cy))
    result_capped = renderer_capped.render_static_field(psf, (cx, cy))

    # Capped result should have less total flux (faint star excluded).
    assert result_capped.sum() < result_no_cap.sum(), (
        "Brightness cap should reduce total flux by excluding faint stars."
    )


def test_brightness_cap_mag_metadata_not_flux() -> None:
    """brightness_cap_mag uses mag_w146 for filtering; flux_e is authoritative."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    cx, cy = 512.0, 512.0
    # High flux_e but high (faint) mag_w146 → should be excluded at cap=20.
    cat = _pd.DataFrame({
        "x_pix": [cx],
        "y_pix": [cy],
        "flux_e": [99999.0],  # very high flux
        "mag_w146": [25.0],   # but faint magnitude
    })
    renderer = CrowdedFieldRenderer(cat, brightness_cap_mag=20.0)
    result = renderer.render_static_field(_make_psf(), (cx, cy))
    # Star excluded by mag cap → zero flux despite large flux_e.
    assert result.sum() == pytest.approx(0.0, abs=1e-10), (
        "Star with mag_w146=25 should be excluded by brightness_cap_mag=20."
    )


# ---------------------------------------------------------------------------
# Additional: total flux scales with number of stars
# ---------------------------------------------------------------------------


def test_total_flux_increases_with_more_stars() -> None:
    """More stars in the stamp → higher total rendered flux."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    psf = _make_psf()
    cx, cy = 512.0, 512.0

    cat5 = _make_catalog(n_stars=5, seed=1)
    cat20 = _make_catalog(n_stars=20, seed=1)

    r5 = CrowdedFieldRenderer(cat5)
    r20 = CrowdedFieldRenderer(cat20)

    flux5 = r5.render_static_field(psf, (cx, cy)).sum()
    flux20 = r20.render_static_field(psf, (cx, cy)).sum()

    assert flux20 > flux5, (
        f"20 stars should produce more flux than 5 stars: "
        f"flux5={flux5:.1f}, flux20={flux20:.1f}"
    )


# ---------------------------------------------------------------------------
# Regression: non-finite and negative catalog value rejection
# ---------------------------------------------------------------------------


def test_nan_in_flux_e_raises() -> None:
    """CrowdedFieldRenderer raises ValueError when flux_e contains NaN."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    bad_cat = _pd.DataFrame({
        "x_pix": [512.0, 513.0],
        "y_pix": [512.0, 512.0],
        "flux_e": [500.0, float("nan")],
        "mag_w146": [20.0, 21.0],
    })
    with pytest.raises(ValueError, match="NaN"):
        CrowdedFieldRenderer(bad_cat)


def test_inf_in_flux_e_raises() -> None:
    """CrowdedFieldRenderer raises ValueError when flux_e contains Inf."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    bad_cat = _pd.DataFrame({
        "x_pix": [512.0],
        "y_pix": [512.0],
        "flux_e": [float("inf")],
        "mag_w146": [20.0],
    })
    with pytest.raises(ValueError, match="Inf"):
        CrowdedFieldRenderer(bad_cat)


def test_inf_in_position_raises() -> None:
    """CrowdedFieldRenderer raises ValueError when x_pix contains Inf."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    bad_cat = _pd.DataFrame({
        "x_pix": [float("inf")],
        "y_pix": [512.0],
        "flux_e": [500.0],
        "mag_w146": [20.0],
    })
    with pytest.raises(ValueError, match="Inf"):
        CrowdedFieldRenderer(bad_cat)


def test_negative_flux_e_raises() -> None:
    """CrowdedFieldRenderer raises ValueError when flux_e is negative."""
    from smig.rendering.crowding import CrowdedFieldRenderer

    bad_cat = _pd.DataFrame({
        "x_pix": [512.0],
        "y_pix": [512.0],
        "flux_e": [-100.0],
        "mag_w146": [20.0],
    })
    with pytest.raises(ValueError, match="negative"):
        CrowdedFieldRenderer(bad_cat)
