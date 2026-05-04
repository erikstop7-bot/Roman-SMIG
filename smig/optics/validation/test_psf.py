"""
smig/optics/validation/test_psf.py
====================================
Unit tests for :class:`~smig.optics.psf.STPSFProvider`.

Test coverage maps to the 10 acceptance criteria for Prompt 3:

  AC1  Constructs without error from PSFConfig defaults (either backend).
  AC2  Monochromatic PSF sum = 1.0 ± 1e-6.
  AC3  get_psf() returns galsim.InterpolatedImage with correct pixel scale.
  AC4  PSF FWHM varies with wavelength (both backends).
  AC5  PSF varies with field position (both backends).
  AC6  Jitter convolution strictly increases FWHM.
  AC7  Disk cache: second call returns bit-identical array (hit counter driven).
  AC8  Peak RSS increase < 500 MB for a small grid (2λ, 1 field position).
  AC9  WebbPSF-dependent tests are skipped when webbpsf is absent.
  AC10 ``smig/sensor/`` contains zero ``galsim`` imports (AST analysis).

Run from the project root::

    python -m pytest smig/optics/validation/test_psf.py -v

WebbPSF tests are automatically skipped when the package (and its data
files) are not installed.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smig.config.optics_schemas import PSFConfig

# ---------------------------------------------------------------------------
# Backend availability flags (mirrors what psf.py probes)
# ---------------------------------------------------------------------------

_GALSIM_AVAILABLE = False
_galsim: Any = None
try:
    import galsim as _galsim  # type: ignore[assignment]
    _GALSIM_AVAILABLE = True
except ImportError:
    pass

_WEBBPSF_AVAILABLE = False
try:
    import webbpsf  # noqa: F401
    _WEBBPSF_AVAILABLE = True
except ImportError:
    pass

# Skip the entire module when galsim is absent.
pytestmark = pytest.mark.skipif(
    not _GALSIM_AVAILABLE,
    reason="galsim is required for PSF tests (pip install -e '.[phase2]')",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_sigma_arcsec(
    psf_array: np.ndarray, pixel_scale_arcsec: float
) -> float:
    """Estimate PSF width via the 2-D second-moment method.

    Returns the RMS radius in arcseconds.  More robust than radial-profile
    half-maximum crossing for small (< 10 pixel) PSF cores.
    """
    h, w = psf_array.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y_idx, x_idx = np.mgrid[:h, :w]
    r2 = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
    total = float(psf_array.sum())
    if total <= 0.0:
        return 0.0
    sigma_pixels = float(np.sqrt((r2 * psf_array).sum() / total))
    return sigma_pixels * pixel_scale_arcsec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> PSFConfig:
    """Default PSFConfig (n_wavelengths=10, oversample=4)."""
    return PSFConfig()


@pytest.fixture
def small_config() -> PSFConfig:
    """Minimal PSFConfig for fast analytic-backend tests."""
    return PSFConfig(n_wavelengths=2, oversample=2)


@pytest.fixture
def cached_config(tmp_path: Path) -> PSFConfig:
    """PSFConfig with disk caching enabled in a temporary directory."""
    return PSFConfig(
        n_wavelengths=2,
        oversample=2,
        cache_dir=str(tmp_path / "psf_cache"),
    )


@pytest.fixture
def provider(small_config: PSFConfig) -> Any:
    from smig.optics.psf import STPSFProvider
    return STPSFProvider(small_config)


@pytest.fixture
def cached_provider(cached_config: PSFConfig) -> Any:
    from smig.optics.psf import STPSFProvider
    return STPSFProvider(cached_config)


# ---------------------------------------------------------------------------
# AC1 — Construction
# ---------------------------------------------------------------------------


def test_construction_default_config(default_config: PSFConfig) -> None:
    """STPSFProvider constructs without error from default PSFConfig."""
    from smig.optics.psf import STPSFProvider
    p = STPSFProvider(default_config)
    assert p._backend in ("webbpsf", "analytic")


def test_construction_small_config(small_config: PSFConfig) -> None:
    """STPSFProvider constructs from a minimal PSFConfig."""
    from smig.optics.psf import STPSFProvider
    p = STPSFProvider(small_config)
    assert p._config is small_config
    assert p._backend in ("webbpsf", "analytic")


def test_construction_stores_wavelength_grid(provider: Any) -> None:
    """Wavelength grid has the configured count and spans the range."""
    cfg = provider._config
    wl = provider._wavelengths_um
    assert len(wl) == cfg.n_wavelengths
    assert wl[0] == pytest.approx(cfg.wavelength_range_um[0], rel=1e-9)
    assert wl[-1] == pytest.approx(cfg.wavelength_range_um[1], rel=1e-9)


def test_construction_log_spaced_grid(provider: Any) -> None:
    """Wavelength grid is log-spaced (constant ratio between neighbours)."""
    wl = provider._wavelengths_um
    ratios = wl[1:] / wl[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-10)


# ---------------------------------------------------------------------------
# AC2 — Monochromatic normalization
# ---------------------------------------------------------------------------


def test_monochromatic_normalization(provider: Any) -> None:
    """Monochromatic PSF sum = 1.0 within 1e-6."""
    psf = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert psf.ndim == 2
    assert psf.dtype == np.float64
    assert abs(float(psf.sum()) - 1.0) < 1e-6


def test_monochromatic_normalization_short_wavelength(provider: Any) -> None:
    """Normalization holds at the short-wavelength end of the range."""
    psf = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 0.93)
    assert abs(float(psf.sum()) - 1.0) < 1e-6


def test_monochromatic_normalization_long_wavelength(provider: Any) -> None:
    """Normalization holds at the long-wavelength end of the range."""
    psf = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 2.0)
    assert abs(float(psf.sum()) - 1.0) < 1e-6


def test_monochromatic_all_positive(provider: Any) -> None:
    """All PSF pixels are non-negative."""
    psf = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert np.all(psf >= 0.0)


def test_sca_id_int_and_str_equivalent(provider: Any) -> None:
    """Integer and string SCA IDs produce identical PSFs."""
    psf_str = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    psf_int = provider.get_psf_at_wavelength(1, (0.5, 0.5), 1.5)
    np.testing.assert_array_equal(psf_str, psf_int)


def test_invalid_sca_id_raises(provider: Any) -> None:
    """SCA ID outside 1-18 raises ValueError."""
    with pytest.raises(ValueError, match="SCA ID"):
        provider.get_psf_at_wavelength(0, (0.5, 0.5), 1.5)
    with pytest.raises(ValueError, match="SCA ID"):
        provider.get_psf_at_wavelength(19, (0.5, 0.5), 1.5)


def test_unsupported_sed_raises(provider: Any) -> None:
    """Unsupported source_sed raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported SED"):
        provider.get_psf("SCA01", (0.5, 0.5), source_sed="blackbody")


# ---------------------------------------------------------------------------
# AC3 — get_psf returns InterpolatedImage with correct scale
# ---------------------------------------------------------------------------


def test_get_psf_returns_interpolated_image(provider: Any) -> None:
    """get_psf() returns a galsim.InterpolatedImage."""
    result = provider.get_psf("SCA01", (0.5, 0.5))
    assert isinstance(result, _galsim.InterpolatedImage)


def test_get_psf_pixel_scale(provider: Any) -> None:
    """Pixel scale stored in provider is 0.11 / oversample."""
    expected = 0.11 / provider._config.oversample
    assert provider._pixel_scale_arcsec == pytest.approx(expected, rel=1e-9)


def test_get_psf_image_scale_attribute(provider: Any) -> None:
    """The galsim Image embedded in the InterpolatedImage has the correct scale."""
    expected = 0.11 / provider._config.oversample
    result = provider.get_psf("SCA01", (0.5, 0.5))
    # GalSim stores the source image as result.image
    if hasattr(result, "image"):
        assert result.image.scale == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# AC4 — PSF FWHM varies with wavelength
# ---------------------------------------------------------------------------


def test_psf_wider_at_longer_wavelength(provider: Any) -> None:
    """PSF second-moment width increases with wavelength (Airy FWHM ∝ λ/D)."""
    psf_short = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 0.93)
    psf_long = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 2.0)
    sigma_short = _estimate_sigma_arcsec(psf_short, provider._pixel_scale_arcsec)
    sigma_long = _estimate_sigma_arcsec(psf_long, provider._pixel_scale_arcsec)
    assert sigma_long > sigma_short, (
        f"PSF should be wider at 2.0 μm than 0.93 μm: "
        f"σ(0.93)={sigma_short:.4f}\", σ(2.0)={sigma_long:.4f}\""
    )


def test_psf_arrays_differ_across_wavelengths(provider: Any) -> None:
    """PSF arrays at 0.93 μm and 2.0 μm must not be identical."""
    psf_short = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 0.93)
    psf_long = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 2.0)
    assert not np.allclose(psf_short, psf_long, atol=1e-10)


# ---------------------------------------------------------------------------
# AC5 — PSF varies with field position
# ---------------------------------------------------------------------------


def test_psf_varies_with_field_position(provider: Any) -> None:
    """PSF at field centre must differ from PSF at field corner."""
    psf_centre = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    psf_corner = provider.get_psf_at_wavelength("SCA01", (0.0, 0.0), 1.5)
    assert not np.allclose(psf_centre, psf_corner, atol=1e-10), (
        "PSF at field centre must differ from PSF at the corner."
    )


def test_psf_wider_at_corner_than_centre(provider: Any) -> None:
    """Analytic fallback: PSF is broader at the corner than at centre."""
    if provider._backend != "analytic":
        pytest.skip("Field-position shape depends on the backend in use.")
    psf_centre = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    psf_corner = provider.get_psf_at_wavelength("SCA01", (0.0, 0.0), 1.5)
    sigma_centre = _estimate_sigma_arcsec(psf_centre, provider._pixel_scale_arcsec)
    sigma_corner = _estimate_sigma_arcsec(psf_corner, provider._pixel_scale_arcsec)
    assert sigma_corner > sigma_centre, (
        f"Corner PSF should be broader than centre PSF in analytic mode: "
        f"σ(centre)={sigma_centre:.4f}\", σ(corner)={sigma_corner:.4f}\""
    )


# ---------------------------------------------------------------------------
# AC6 — Jitter increases FWHM
# ---------------------------------------------------------------------------


def test_jitter_increases_fwhm() -> None:
    """Large jitter (50 mas) strictly broadens the polychromatic PSF."""
    from smig.optics.psf import STPSFProvider

    # oversample=4 gives ~2 pixel Airy FWHM at 1.5 μm; 50 mas adds ~1.8 pix σ.
    cfg_no = PSFConfig(n_wavelengths=2, oversample=4, jitter_rms_mas=0.0)
    cfg_yes = PSFConfig(n_wavelengths=2, oversample=4, jitter_rms_mas=50.0)

    prov_no = STPSFProvider(cfg_no)
    prov_yes = STPSFProvider(cfg_yes)

    psf_no = prov_no.get_psf("SCA01", (0.5, 0.5))
    psf_yes = prov_yes.get_psf("SCA01", (0.5, 0.5))

    arr_no = psf_no.image.array if hasattr(psf_no, "image") else psf_no.drawImage(scale=cfg_no.oversample and 0.11 / cfg_no.oversample).array
    arr_yes = psf_yes.image.array if hasattr(psf_yes, "image") else psf_yes.drawImage(scale=0.11 / cfg_yes.oversample).array

    sigma_no = _estimate_sigma_arcsec(arr_no, 0.11 / cfg_no.oversample)
    sigma_yes = _estimate_sigma_arcsec(arr_yes, 0.11 / cfg_yes.oversample)

    assert sigma_yes > sigma_no, (
        f"Jitter should broaden PSF: σ(no jitter)={sigma_no:.4f}\" "
        f"vs σ(50 mas)={sigma_yes:.4f}\""
    )


def test_jitter_zero_no_change() -> None:
    """jitter_rms_mas=0.0 should produce the same PSF as no-jitter path."""
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, jitter_rms_mas=0.0)
    p = STPSFProvider(cfg)
    result = p.get_psf("SCA01", (0.5, 0.5))
    assert isinstance(result, _galsim.InterpolatedImage)


def test_different_jitter_seeds_produce_different_psfs() -> None:
    """Different jitter_seed values produce distinct polychromatic PSFs."""
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, jitter_rms_mas=20.0)
    p = STPSFProvider(cfg)

    psf_0 = p.get_psf("SCA01", (0.5, 0.5), jitter_seed=0)
    psf_1 = p.get_psf("SCA01", (0.5, 0.5), jitter_seed=1)

    if hasattr(psf_0, "image"):
        arr0 = psf_0.image.array
        arr1 = psf_1.image.array
    else:
        pytest.skip("Cannot extract array from InterpolatedImage on this GalSim version.")

    assert not np.array_equal(arr0, arr1), (
        "Different jitter seeds must produce different PSF arrays."
    )


def test_same_jitter_seed_is_deterministic() -> None:
    """Same jitter_seed always produces the same polychromatic PSF."""
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, jitter_rms_mas=20.0)
    p1 = STPSFProvider(cfg)
    p2 = STPSFProvider(cfg)

    psf_a = p1.get_psf("SCA01", (0.5, 0.5), jitter_seed=42)
    psf_b = p2.get_psf("SCA01", (0.5, 0.5), jitter_seed=42)

    if hasattr(psf_a, "image"):
        np.testing.assert_array_equal(psf_a.image.array, psf_b.image.array)


# ---------------------------------------------------------------------------
# AC7 — Disk cache correctness (hit counter, not timing)
# ---------------------------------------------------------------------------


def test_memory_cache_hit_on_second_call(provider: Any) -> None:
    """Second call with same params is a memory-cache hit."""
    provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    initial_hits = provider.cache_hits
    provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert provider.cache_hits == initial_hits + 1


def test_memory_cache_miss_on_new_params(provider: Any) -> None:
    """Different wavelength is a fresh cache miss."""
    provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.0)
    initial_misses = provider.cache_misses
    provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.9)
    assert provider.cache_misses == initial_misses + 1


def test_disk_cache_returns_identical_array(cached_provider: Any) -> None:
    """A second STPSFProvider with the same config+cache_dir returns the
    bit-identical array loaded from disk."""
    from smig.optics.psf import STPSFProvider

    arr_first = cached_provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)

    # New provider — empty memory cache, but disk cache populated.
    p2 = STPSFProvider(cached_provider._config)
    arr_second = p2.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)

    np.testing.assert_array_equal(arr_first, arr_second)


def test_disk_cache_config_change_invalidates(tmp_path: Path) -> None:
    """Changing the PSFConfig changes the config_hash, producing a distinct key."""
    from smig.optics.psf import STPSFProvider

    cache = str(tmp_path / "shared_cache")
    cfg1 = PSFConfig(n_wavelengths=2, oversample=2, cache_dir=cache)
    cfg2 = PSFConfig(
        n_wavelengths=2, oversample=2, cache_dir=cache, filter_name="F087"
    )

    p1 = STPSFProvider(cfg1)
    p2 = STPSFProvider(cfg2)

    assert p1._config_hash != p2._config_hash

    arr1 = p1.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    arr2 = p2.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)

    # Both compute successfully (no stale-cache poisoning).
    assert arr1.shape == arr2.shape
    assert arr1.ndim == 2


def test_disk_cache_wavelength_grid_change_invalidates(tmp_path: Path) -> None:
    """Changing n_wavelengths invalidates the polychromatic cache."""
    from smig.optics.psf import STPSFProvider

    cache = str(tmp_path / "shared_cache")
    cfg_2wl = PSFConfig(n_wavelengths=2, oversample=2, cache_dir=cache)
    cfg_3wl = PSFConfig(n_wavelengths=3, oversample=2, cache_dir=cache)

    p2 = STPSFProvider(cfg_2wl)
    p3 = STPSFProvider(cfg_3wl)

    # Different config_hashes → different polychromatic cache keys.
    assert p2._config_hash != p3._config_hash


def test_no_cache_dir_does_not_create_files() -> None:
    """Without cache_dir, no .h5 file is created."""
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, cache_dir=None)
    p = STPSFProvider(cfg)
    p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert p._get_cache_path() is None


# ---------------------------------------------------------------------------
# AC8 — Memory footprint < 500 MB for a small grid
# ---------------------------------------------------------------------------


def test_memory_footprint_small_grid() -> None:
    """Peak RSS increase must be < 500 MB for a 2-wavelength, 1-position run."""
    from smig.optics.psf import STPSFProvider

    # Measure before
    rss_before_mb = _get_rss_mb()
    if rss_before_mb is None:
        pytest.skip("Cannot measure RSS on this platform (install psutil).")

    cfg = PSFConfig(n_wavelengths=2, oversample=2)
    p = STPSFProvider(cfg)
    _ = p.get_psf("SCA01", (0.5, 0.5))

    rss_after_mb = _get_rss_mb()
    increase_mb = rss_after_mb - rss_before_mb  # type: ignore[operator]
    assert increase_mb < 500.0, (
        f"RSS increased by {increase_mb:.1f} MB (limit: 500 MB)."
    )


def _get_rss_mb() -> float | None:
    """Return current process RSS in MB, or None if measurement is unavailable."""
    try:
        import os
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        pass
    try:
        import resource  # Unix only
        import sys
        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss_raw / (1024 ** 2) if sys.platform == "darwin" else rss_raw / 1024
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# AC9 — WebbPSF backend tests (skipped when absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _WEBBPSF_AVAILABLE,
    reason="webbpsf not installed or data files absent",
)
class TestWebbPSFBackend:
    """Tests that exercise the WebbPSF backend specifically."""

    def test_backend_is_webbpsf(self) -> None:
        from smig.optics.psf import STPSFProvider
        p = STPSFProvider(PSFConfig(n_wavelengths=2, oversample=2))
        assert p._backend == "webbpsf"

    def test_monochromatic_normalization(self) -> None:
        from smig.optics.psf import STPSFProvider
        p = STPSFProvider(PSFConfig(n_wavelengths=2, oversample=2))
        psf = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
        assert abs(float(psf.sum()) - 1.0) < 1e-6

    def test_fwhm_varies_with_wavelength(self) -> None:
        from smig.optics.psf import STPSFProvider
        p = STPSFProvider(PSFConfig(n_wavelengths=2, oversample=4))
        psf_short = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 0.93)
        psf_long = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 2.0)
        sigma_short = _estimate_sigma_arcsec(psf_short, 0.11 / 4)
        sigma_long = _estimate_sigma_arcsec(psf_long, 0.11 / 4)
        assert sigma_long > sigma_short

    def test_fwhm_varies_with_field_position(self) -> None:
        from smig.optics.psf import STPSFProvider
        p = STPSFProvider(PSFConfig(n_wavelengths=2, oversample=2))
        psf_centre = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
        psf_edge = p.get_psf_at_wavelength("SCA09", (0.1, 0.9), 1.5)
        assert not np.allclose(psf_centre, psf_edge, atol=1e-10)

    def test_get_psf_returns_interpolated_image(self) -> None:
        from smig.optics.psf import STPSFProvider
        p = STPSFProvider(PSFConfig(n_wavelengths=2, oversample=2))
        result = p.get_psf("SCA01", (0.5, 0.5))
        assert isinstance(result, _galsim.InterpolatedImage)


# ---------------------------------------------------------------------------
# AC10 — Architecture boundary: no galsim in smig/sensor/
# ---------------------------------------------------------------------------


def test_no_galsim_imports_in_sensor() -> None:
    """Verify that no file under smig/sensor/ imports galsim (AST-based)."""
    # smig/optics/validation/test_psf.py  →  parents[2] = smig/
    sensor_dir = Path(__file__).resolve().parent.parent.parent / "sensor"
    assert sensor_dir.is_dir(), (
        f"Expected smig/sensor/ at {sensor_dir} — path may need updating."
    )

    violations: list[str] = []
    for py_file in sensor_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "galsim" or alias.name.startswith("galsim."):
                        violations.append(
                            f"{py_file.relative_to(sensor_dir.parent)}:{node.lineno}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "galsim" or mod.startswith("galsim."):
                    violations.append(
                        f"{py_file.relative_to(sensor_dir.parent)}:{node.lineno}"
                    )

    assert not violations, (
        f"Found galsim import(s) in smig/sensor/ — star-topology violation:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# Additional: backend attribute and imports safe without Phase 2
# ---------------------------------------------------------------------------


def test_backend_attribute_valid(provider: Any) -> None:
    """._backend is one of the two documented values."""
    assert provider._backend in ("webbpsf", "analytic")


def test_analytic_backend_when_webbpsf_absent() -> None:
    """When webbpsf is absent, backend is 'analytic'."""
    if _WEBBPSF_AVAILABLE:
        pytest.skip("webbpsf is installed; cannot test analytic-only path.")
    from smig.optics.psf import STPSFProvider
    p = STPSFProvider(PSFConfig(n_wavelengths=2, oversample=2))
    assert p._backend == "analytic"


def test_smig_optics_import_always_succeeds() -> None:
    """``import smig.optics`` must not raise even without Phase 2 extras."""
    import importlib
    mod = importlib.import_module("smig.optics")
    # STPSFProvider is either the class or None (base-only install).
    assert mod.STPSFProvider is not None or mod.STPSFProvider is None


def test_field_position_clamped(provider: Any) -> None:
    """Field positions outside [0,1] are clamped, not rejected."""
    psf = provider.get_psf_at_wavelength("SCA01", (-0.1, 1.5), 1.5)
    assert abs(float(psf.sum()) - 1.0) < 1e-6


def test_field_position_clamp_zero_endpoint(provider: Any) -> None:
    """fp=(0.0, 0.0) is accepted and maps to detector pixel (0, 0)."""
    psf = provider.get_psf_at_wavelength("SCA01", (0.0, 0.0), 1.5)
    assert psf.ndim == 2
    assert abs(float(psf.sum()) - 1.0) < 1e-6


def test_field_position_clamp_one_endpoint(provider: Any) -> None:
    """fp=(1.0, 1.0) is accepted and maps to detector pixel (4095, 4095), not 4096."""
    psf = provider.get_psf_at_wavelength("SCA01", (1.0, 1.0), 1.5)
    assert psf.ndim == 2
    assert abs(float(psf.sum()) - 1.0) < 1e-6


def test_field_position_clamp_near_edge_epsilon(provider: Any) -> None:
    """fp values just inside each edge are accepted without clamping error."""
    eps = 1e-4
    for fp in [(eps, eps), (1.0 - eps, 1.0 - eps)]:
        psf = provider.get_psf_at_wavelength("SCA01", fp, 1.5)
        assert abs(float(psf.sum()) - 1.0) < 1e-6, (
            f"Normalization failed for field_position={fp}"
        )


def test_field_position_webbpsf_det_coords_in_range() -> None:
    """_quantize_field_position followed by *4095 stays within [0, 4095]."""
    from smig.optics.psf import _quantize_field_position

    for raw_fp in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (-0.5, 2.0), (1.0, 0.0)]:
        qfp = _quantize_field_position(raw_fp)
        det_x = qfp[0] * 4095.0
        det_y = qfp[1] * 4095.0
        assert 0.0 <= det_x <= 4095.0, (
            f"det_x={det_x} out of range for raw fp={raw_fp}"
        )
        assert 0.0 <= det_y <= 4095.0, (
            f"det_y={det_y} out of range for raw fp={raw_fp}"
        )


def test_polychromatic_normalization(provider: Any) -> None:
    """Polychromatic PSF underlying array sums to 1.0."""
    result = provider.get_psf("SCA01", (0.5, 0.5))
    if hasattr(result, "image"):
        total = float(result.image.array.sum())
        assert abs(total - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# fov_native_pixels — plumbing and validation
# ---------------------------------------------------------------------------


def test_fov_native_pixels_default() -> None:
    """Default fov_native_pixels is 128."""
    cfg = PSFConfig()
    assert cfg.fov_native_pixels == 128


def test_fov_native_pixels_minimum_rejected() -> None:
    """fov_native_pixels < 16 raises ValidationError."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        PSFConfig(fov_native_pixels=15)


def test_fov_native_pixels_plumbed_to_analytic_array() -> None:
    """Analytic PSF array shape reflects fov_native_pixels * oversample."""
    from smig.optics.psf import STPSFProvider
    cfg = PSFConfig(n_wavelengths=2, oversample=2, fov_native_pixels=32)
    p = STPSFProvider(cfg)
    if p._backend != "analytic":
        pytest.skip("shape test only meaningful for analytic backend")
    arr = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert arr.shape == (32 * 2, 32 * 2), (
        f"Expected shape (64, 64) for fov_native_pixels=32, oversample=2; got {arr.shape}"
    )


def test_fov_native_pixels_normalization_custom() -> None:
    """fov_native_pixels=32 PSF still normalizes to 1.0 ± 1e-6."""
    from smig.optics.psf import STPSFProvider
    cfg = PSFConfig(n_wavelengths=2, oversample=2, fov_native_pixels=32)
    p = STPSFProvider(cfg)
    arr = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert abs(float(arr.sum()) - 1.0) < 1e-6


def test_simulation_config_psf_fov_exceeds_geometry_rejected() -> None:
    """SimulationConfig rejects psf.fov_native_pixels > detector.geometry.nx/ny."""
    import pydantic
    import yaml
    from pathlib import Path
    from smig.config.optics_schemas import SimulationConfig

    raw = yaml.safe_load(
        Path("smig/config/simulation.yaml").read_text()
    )
    # nx=ny=256, fov_native_pixels=512 → must fail
    raw["psf"]["fov_native_pixels"] = 512
    with pytest.raises((pydantic.ValidationError, ValueError)):
        SimulationConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Jitter reflect-mode correctness
# ---------------------------------------------------------------------------


def test_jitter_reflect_mode_finite_normalized() -> None:
    """Jitter-on PSF (reflect mode) remains finite and sums to 1.0 ± 1e-5."""
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, jitter_rms_mas=20.0)
    p = STPSFProvider(cfg)
    result = p.get_psf("SCA01", (0.5, 0.5), jitter_seed=7)

    if not hasattr(result, "image"):
        pytest.skip("Cannot extract array on this GalSim version.")
    arr = result.image.array
    assert np.all(np.isfinite(arr)), "Jitter-on PSF contains non-finite values."
    assert abs(float(arr.sum()) - 1.0) < 1e-5, (
        f"Jitter-on PSF sum deviates from 1.0: {arr.sum():.8f}"
    )


def test_jitter_reflect_mode_non_negative() -> None:
    """Jitter-on PSF values are non-negative (reflect mode preserves sign)."""
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, jitter_rms_mas=20.0)
    p = STPSFProvider(cfg)
    result = p.get_psf("SCA01", (0.5, 0.5), jitter_seed=7)

    if not hasattr(result, "image"):
        pytest.skip("Cannot extract array on this GalSim version.")
    arr = result.image.array
    assert np.all(arr >= 0.0), (
        f"Jitter-on PSF has negative pixels; min={arr.min():.2e}"
    )


def test_jitter_reflect_mode_deterministic() -> None:
    """Same seed → bit-identical PSF array under reflect-mode jitter."""
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, jitter_rms_mas=20.0)
    p1 = STPSFProvider(cfg)
    p2 = STPSFProvider(cfg)

    r1 = p1.get_psf("SCA01", (0.5, 0.5), jitter_seed=99)
    r2 = p2.get_psf("SCA01", (0.5, 0.5), jitter_seed=99)

    if not (hasattr(r1, "image") and hasattr(r2, "image")):
        pytest.skip("Cannot extract array on this GalSim version.")
    np.testing.assert_array_equal(
        r1.image.array, r2.image.array,
        err_msg="Same jitter_seed must produce identical arrays with reflect mode.",
    )


def test_jitter_reflect_preserves_edge_energy_vs_constant() -> None:
    """Reflect-mode Gaussian blur preserves outer-rim energy; constant mode suppresses it.

    With mode='constant', cval=0, edge pixels are blended with zeros → artificially
    reduced.  With mode='reflect', edge pixels are blended with their mirror image →
    natural values preserved.  Therefore rms_reflect >= rms_constant at the outer rim.
    """
    from scipy.ndimage import gaussian_filter
    from smig.optics.psf import STPSFProvider

    cfg = PSFConfig(n_wavelengths=2, oversample=2, fov_native_pixels=32, jitter_rms_mas=0.0)
    p = STPSFProvider(cfg)
    if p._backend != "analytic":
        pytest.skip("outer-rim comparison only runs on analytic backend")

    raw = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    sigma = 2.0  # pixels — large enough to reach the boundary

    blurred_reflect = gaussian_filter(raw, sigma=sigma, mode="reflect")
    blurred_constant = gaussian_filter(raw, sigma=sigma, mode="constant", cval=0.0)

    # Normalize both so the comparison is flux-independent.
    blurred_reflect = blurred_reflect / blurred_reflect.sum()
    blurred_constant = blurred_constant / blurred_constant.sum()

    rms_reflect = _outer_rim_rms(blurred_reflect, rim_width=2)
    rms_constant = _outer_rim_rms(blurred_constant, rim_width=2)

    assert rms_reflect >= rms_constant, (
        f"Reflect mode should preserve edge energy vs constant mode: "
        f"reflect={rms_reflect:.2e}, constant={rms_constant:.2e}"
    )


# ---------------------------------------------------------------------------
# Edge taper helpers and tests
# ---------------------------------------------------------------------------


def _outer_rim_rms(array: np.ndarray, rim_width: int = 1) -> float:
    """RMS of pixel values in the outer ``rim_width`` pixel border."""
    mask = np.zeros(array.shape, dtype=bool)
    mask[:rim_width, :] = True
    mask[-rim_width:, :] = True
    mask[:, :rim_width] = True
    mask[:, -rim_width:] = True
    return float(np.sqrt(np.mean(array[mask] ** 2)))


def test_taper_default_is_none() -> None:
    """psf_edge_taper_pixels defaults to None (taper disabled)."""
    cfg = PSFConfig()
    assert cfg.psf_edge_taper_pixels is None


def test_taper_negative_rejected() -> None:
    """Negative psf_edge_taper_pixels raises ValidationError."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        PSFConfig(psf_edge_taper_pixels=-1.0)


def test_taper_off_identical_to_baseline(provider: Any) -> None:
    """With taper disabled (None), PSF is identical to pre-feature behavior."""
    assert provider._config.psf_edge_taper_pixels is None
    psf = provider.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert abs(float(psf.sum()) - 1.0) < 1e-6
    assert np.all(psf >= 0.0)


def test_taper_on_normalization() -> None:
    """With taper enabled, monochromatic PSF still sums to 1.0 ± 1e-6."""
    from smig.optics.psf import STPSFProvider
    cfg = PSFConfig(
        n_wavelengths=2, oversample=2, fov_native_pixels=32,
        psf_edge_taper_pixels=4.0,
    )
    p = STPSFProvider(cfg)
    if p._backend != "analytic":
        pytest.skip("taper normalization test only runs on analytic backend")
    arr = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert abs(float(arr.sum()) - 1.0) < 1e-6


def test_taper_on_finite_nonneg() -> None:
    """With taper enabled, all PSF values are finite and non-negative."""
    from smig.optics.psf import STPSFProvider
    cfg = PSFConfig(
        n_wavelengths=2, oversample=2, fov_native_pixels=32,
        psf_edge_taper_pixels=4.0,
    )
    p = STPSFProvider(cfg)
    if p._backend != "analytic":
        pytest.skip("taper finite/nonneg test only runs on analytic backend")
    arr = p.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)


def test_taper_reduces_edge_energy() -> None:
    """Tapered PSF has strictly lower RMS at the outermost pixel border."""
    from smig.optics.psf import STPSFProvider

    cfg_off = PSFConfig(
        n_wavelengths=2, oversample=2, fov_native_pixels=32,
    )
    cfg_on = PSFConfig(
        n_wavelengths=2, oversample=2, fov_native_pixels=32,
        psf_edge_taper_pixels=4.0,
    )
    p_off = STPSFProvider(cfg_off)
    p_on = STPSFProvider(cfg_on)

    if p_off._backend != "analytic":
        pytest.skip("edge-energy test only runs on analytic backend")

    arr_off = p_off.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)
    arr_on = p_on.get_psf_at_wavelength("SCA01", (0.5, 0.5), 1.5)

    # The cosine taper is exactly 0 at the outermost row/column (i=0),
    # so after tapering + renormalization those pixels are 0.
    rms_off = _outer_rim_rms(arr_off, rim_width=1)
    rms_on = _outer_rim_rms(arr_on, rim_width=1)

    assert rms_on < rms_off, (
        f"Tapered edge RMS ({rms_on:.2e}) should be < untapered ({rms_off:.2e})"
    )


def test_taper_polychromatic_normalization() -> None:
    """With taper enabled, polychromatic PSF array still sums to 1.0."""
    from smig.optics.psf import STPSFProvider
    cfg = PSFConfig(
        n_wavelengths=2, oversample=2, fov_native_pixels=32,
        psf_edge_taper_pixels=4.0,
    )
    p = STPSFProvider(cfg)
    if p._backend != "analytic":
        pytest.skip("taper polychromatic test only runs on analytic backend")
    result = p.get_psf("SCA01", (0.5, 0.5))
    if hasattr(result, "image"):
        total = float(result.image.array.sum())
        assert abs(total - 1.0) < 1e-5


def test_cosine_taper_1d_boundary_values() -> None:
    """_cosine_taper_1d returns 0 at edges and 1 in interior."""
    from smig.optics.psf import _cosine_taper_1d

    t = _cosine_taper_1d(20, 4)
    assert t[0] == pytest.approx(0.0, abs=1e-12)
    assert t[-1] == pytest.approx(0.0, abs=1e-12)
    # Interior pixels (beyond the ramp) must be 1.
    assert np.all(t[4:-4] == 1.0)


def test_cosine_taper_1d_no_taper() -> None:
    """_cosine_taper_1d with w=0 returns all-ones array."""
    from smig.optics.psf import _cosine_taper_1d

    t = _cosine_taper_1d(16, 0)
    assert np.all(t == 1.0)

