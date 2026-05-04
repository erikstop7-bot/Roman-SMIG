"""
smig/rendering/validation/test_integration_phase2.py
=====================================================
Comprehensive Phase 2 end-to-end validation suite.

Coverage
--------
AC-1   PSF physics: monochromatic FWHM increases with wavelength (0.93–2.0 μm)
       on analytic fallback backend.
AC-2   Flux conservation (Rendering): sum of ideal_image_e before detector is
       within 0.1% of injected source flux.
AC-3   Flux conservation (Detector): aperture photometry on ``rate_image * t_exp_s``
       recovers injected flux within 5%.
AC-4   DIA null test: static-field difference residual mean is consistent with
       zero; RMS is within 2× the theoretical noise level.
AC-5   DIA recovery: injected point source recovered within 10% flux accuracy.
AC-6   Seed determinism: same ``(master_seed, event_id)`` → identical outputs.
       Float arrays: ``assert_allclose(rtol=1e-6, atol=1e-8)``.
       Integer/bool arrays: ``assert_array_equal``.
AC-7   Seed independence: different ``event_id`` → KS test ``p-value < 0.05``
       on pooled difference-stamp pixels across all 3 epochs.
AC-8   Memory: full pipeline on 32×32 / 3-epoch stays under 1 GB peak
       (``tracemalloc``). Marked ``@pytest.mark.slow``.
AC-9   Architecture boundary (Python AST): ``smig/rendering/**/*.py`` may only
       import ``smig.sensor.detector``; leaf modules (ipc, readout, …) are
       forbidden.  ``smig/sensor/**/*.py`` must not import ``galsim``.
AC-10  Leakage guard CLI: ``scripts/validate_splits.py`` correctly flags
       duplicate event IDs, shared starfield seeds, and param-similar events in
       different splits; returns exit-code 0 for a clean manifest.

Test configuration fixtures
---------------------------
* ``_force_analytic_psf_backend`` — ``autouse=True``; patches
  ``smig.optics.psf._WEBBPSF_AVAILABLE = False`` so every ``STPSFProvider``
  constructed during this module uses the analytic Airy+Gaussian backend.
* ``SimulationConfig``: ``oversample=2``, ``n_wavelengths=2``, no jitter,
  32×32 context, 16×16 science crop — keeps CI fast.

Run from the SMIG project root::

    python -m pytest smig/rendering/validation/test_integration_phase2.py -v
    python -m pytest smig/rendering/validation/test_integration_phase2.py -v -m slow
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Guard: skip entire module when Phase 2 extras are absent.
# ---------------------------------------------------------------------------

galsim = pytest.importorskip("galsim", reason="galsim required for Phase 2 integration tests")
pd = pytest.importorskip("pandas", reason="pandas required for Phase 2 integration tests")

from smig.config.optics_schemas import (
    CrowdedFieldConfig,
    DIAConfig,
    PSFConfig,
    RenderingConfig,
    SimulationConfig,
)
from smig.config.schemas import DetectorConfig
from smig.rendering.dia import DIAPipeline
from smig.rendering.pipeline import EventSceneOutput, SceneSimulator
from smig.rendering.source import FiniteSourceRenderer
from smig.sensor.detector import H4RG10Detector


# ---------------------------------------------------------------------------
# Module-level fixture: force the analytic PSF backend for every test.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _force_analytic_psf_backend():
    """Patch _WEBBPSF_AVAILABLE=False so STPSFProvider always uses analytic backend.

    Runs as a function-scoped autouse fixture so every ``STPSFProvider``
    construction that happens inside a test (or a fixture resolved within a
    test call) sees the patched value.
    """
    with patch("smig.optics.psf._WEBBPSF_AVAILABLE", False):
        yield


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_small_detector(nx: int = 32, ny: int = 32) -> DetectorConfig:
    """Return a DetectorConfig with overridden geometry (all other defaults)."""
    base = DetectorConfig()
    d = base.model_dump()
    d["geometry"]["nx"] = nx
    d["geometry"]["ny"] = ny
    return DetectorConfig.model_validate(d)


def _make_sim_config(
    ctx: int = 64,
    sci_sz: int = 16,
    n_ref: int = 5,
    brightness_cap_mag: float | None = None,
) -> SimulationConfig:
    """Build a minimal SimulationConfig suitable for fast CI tests.

    Forces the analytic PSF backend via ``oversample=2, n_wavelengths=2``,
    disables jitter (``jitter_rms_mas=0.0``), and uses small stamp geometries.
    """
    small_det = _make_small_detector(nx=ctx, ny=ctx)
    return SimulationConfig(
        detector=small_det,
        psf=PSFConfig(
            oversample=2,
            n_wavelengths=2,
            jitter_rms_mas=0.0,
            # PSF FOV must be <= detector geometry; tying it to ctx keeps the
            # configuration valid when callers shrink the test stamp size.
            fov_native_pixels=ctx,
        ),
        rendering=RenderingConfig(),
        crowded_field=CrowdedFieldConfig(
            stamp_size=sci_sz,
            pixel_scale_arcsec=0.11,
            brightness_cap_mag=brightness_cap_mag,
        ),
        dia=DIAConfig(
            n_reference_epochs=n_ref,
            context_stamp_size=ctx,
            science_stamp_size=sci_sz,
            subtraction_method="alard_lupton",
        ),
    )


def _make_source_params(n: int, flux_e: float = 5_000.0) -> list[dict]:
    """Return *n* identical point-source parameter dicts."""
    return [
        {
            "flux_e": flux_e,
            "centroid_offset_pix": (0.0, 0.0),
            "rho_star_arcsec": 0.0,
            "limb_darkening_coeffs": None,
        }
        for _ in range(n)
    ]


def _make_timestamps(
    n: int,
    t0_mjd: float = 60_000.0,
    dt_days: float = 1.0,
) -> np.ndarray:
    """Return an evenly-spaced MJD timestamp array of length *n*."""
    return np.array([t0_mjd + i * dt_days for i in range(n)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Aperture photometry helper (no photutils — spec requirement)
# ---------------------------------------------------------------------------

def _aperture_flux(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float = 3.0,
) -> float:
    """Sum pixels within *radius* px of *(cx, cy)*, subtract edge background.

    Implements the "3-pixel aperture photometry" defined in the spec:
    sum all pixels within a circle of the given radius centred on the known
    sub-pixel centroid, then subtract a per-pixel background estimated as the
    median of all edge pixels (within 1 px of the stamp border).

    Parameters
    ----------
    image:
        2-D float array (any units).
    cx, cy:
        Column and row coordinates of the centroid (0-indexed).
    radius:
        Aperture radius in pixels.  Default: 3.0.

    Returns
    -------
    float
        ``sum(aperture_pixels) - median(edge_pixels) * n_aperture_pixels``.
    """
    h, w = image.shape
    rows, cols = np.mgrid[:h, :w]
    dist = np.sqrt((cols.astype(float) - cx) ** 2 + (rows.astype(float) - cy) ** 2)
    aperture_mask = dist <= radius

    edge_mask = (rows <= 1) | (rows >= h - 2) | (cols <= 1) | (cols >= w - 2)
    bg = float(np.median(image[edge_mask])) if edge_mask.any() else 0.0

    n_ap = int(aperture_mask.sum())
    return float(image[aperture_mask].sum()) - bg * n_ap


# ---------------------------------------------------------------------------
# AC-1: PSF Physics
# ---------------------------------------------------------------------------

def _measure_psf_second_moment(psf_array: np.ndarray) -> float:
    """Measure PSF RMS radius via second moment — a robust proxy for FWHM.

    Returns the square root of the intensity-weighted second moment of the PSF
    about its centroid, in pixels.
    """
    h, w = psf_array.shape
    rows, cols = np.mgrid[:h, :w].astype(float)
    total = float(psf_array.sum())
    if total <= 0.0:
        return 0.0
    cx = float(np.sum(cols * psf_array)) / total
    cy = float(np.sum(rows * psf_array)) / total
    sigma2 = float(
        np.sum(((cols - cx) ** 2 + (rows - cy) ** 2) * psf_array)
    ) / total
    return float(np.sqrt(sigma2))


class TestPSFPhysics:
    """AC-1: Monochromatic FWHM must increase with wavelength (analytic backend)."""

    def test_fwhm_increases_from_blue_to_red(self) -> None:
        """PSF second-moment at 2.0 μm must exceed that at 0.93 μm.

        The analytic Airy disk has FWHM ∝ λ/D, so the PSF broadens at longer
        wavelengths across the W146 bandpass (0.93 – 2.00 μm).
        """
        from smig.optics.psf import STPSFProvider

        cfg = PSFConfig(oversample=2, n_wavelengths=2, jitter_rms_mas=0.0)
        provider = STPSFProvider(cfg)
        assert provider._backend == "analytic", (
            f"Expected analytic backend (WebbPSF patched out), got {provider._backend!r}."
        )

        psf_blue = provider.get_psf_at_wavelength(1, (0.5, 0.5), wavelength_um=0.93)
        psf_red = provider.get_psf_at_wavelength(1, (0.5, 0.5), wavelength_um=2.00)

        sigma_blue = _measure_psf_second_moment(psf_blue)
        sigma_red = _measure_psf_second_moment(psf_red)

        assert sigma_red > sigma_blue, (
            f"PSF FWHM trend violation: σ_red={sigma_red:.4f} px @ 2.00 μm must "
            f"exceed σ_blue={sigma_blue:.4f} px @ 0.93 μm.  "
            "Diffraction scaling (FWHM ∝ λ/D) not reproduced by analytic backend."
        )

    def test_monochromatic_psf_normalized_to_unity(self) -> None:
        """get_psf_at_wavelength must return sum=1.0 ± 1e-5."""
        from smig.optics.psf import STPSFProvider

        cfg = PSFConfig(oversample=2, n_wavelengths=2, jitter_rms_mas=0.0)
        provider = STPSFProvider(cfg)
        psf = provider.get_psf_at_wavelength(1, (0.5, 0.5), wavelength_um=1.5)
        assert abs(float(psf.sum()) - 1.0) < 1e-5, (
            f"PSF not normalized: sum={psf.sum():.8f}, expected 1.0 ± 1e-5."
        )

    def test_fwhm_trend_monotone_across_intermediate_wavelength(self) -> None:
        """FWHM at 1.5 μm must lie between 0.93 μm and 2.0 μm values."""
        from smig.optics.psf import STPSFProvider

        cfg = PSFConfig(oversample=2, n_wavelengths=2, jitter_rms_mas=0.0)
        provider = STPSFProvider(cfg)

        sigmas = {
            wl: _measure_psf_second_moment(
                provider.get_psf_at_wavelength(1, (0.5, 0.5), wavelength_um=wl)
            )
            for wl in (0.93, 1.50, 2.00)
        }
        assert sigmas[0.93] < sigmas[1.50] < sigmas[2.00], (
            f"FWHM not monotonically increasing: {sigmas}"
        )


# ---------------------------------------------------------------------------
# AC-2: Flux conservation (Rendering)
# ---------------------------------------------------------------------------

class TestRenderingFluxConservation:
    """AC-2: ideal_image_e sum before detector is within 0.1% of injected flux."""

    def test_point_source_flux_conserved_to_0pt1pct(self) -> None:
        """FiniteSourceRenderer.render_source must place flux_e e⁻ in the stamp.

        Creates a blank GalSim stamp, renders a single point source with a
        known total flux, clips to non-negative (mirroring the pipeline), and
        asserts the pixel sum matches the injected flux to within 0.1%.
        """
        from smig.optics.psf import STPSFProvider

        flux_e = 10_000.0
        cfg = PSFConfig(oversample=2, n_wavelengths=2, jitter_rms_mas=0.0)
        provider = STPSFProvider(cfg)
        # Use jitter_seed=1 to get a deterministic PSF realization.
        psf = provider.get_psf(sca_id=1, field_position=(0.5, 0.5), jitter_seed=1)

        stamp_size = 128
        stamp = galsim.Image(stamp_size, stamp_size, scale=0.11)

        renderer = FiniteSourceRenderer()
        renderer.render_source(
            flux_e=flux_e,
            centroid_offset_pix=(0.0, 0.0),
            rho_star_arcsec=0.0,
            limb_darkening_coeffs=None,
            psf=psf,
            stamp=stamp,
        )

        rendered = np.clip(stamp.array, 0.0, None)
        total_rendered = float(rendered.sum())
        rel_error = abs(total_rendered - flux_e) / flux_e

        assert rel_error < 0.001, (
            f"Rendering flux conservation failed: injected {flux_e:.1f} e⁻, "
            f"stamp sum = {total_rendered:.2f} e⁻ (rel. error = {rel_error:.4%}, "
            f"tolerance 0.1%)."
        )


# ---------------------------------------------------------------------------
# AC-3: Flux conservation (Detector)
# ---------------------------------------------------------------------------

class TestDetectorFluxConservation:
    """AC-3: Aperture photometry on rate_image * t_exp_s recovers flux within 5%."""

    def test_aperture_flux_within_5pct(self) -> None:
        """H4RG10Detector must conserve total photon flux through the signal chain.

        Injects ``flux_e`` electrons into a single central pixel of a 32×32
        ideal image, processes the epoch through the full detector chain, and
        measures the aperture-photometry flux in a 3-px radius circle centred on
        the known source position.  The dark-current contribution to the
        aperture is subtracted before comparing to the injected flux.
        """
        flux_e = 5_000.0
        det = _make_small_detector(nx=32, ny=32)
        rng = np.random.default_rng(1234)
        detector = H4RG10Detector(det, rng)

        # Delta-function ideal image: all flux at one pixel plus a baseline
        # background to prevent negative-lambda Poisson draws from float noise.
        ideal_image = np.zeros((32, 32), dtype=np.float64)
        cy, cx = 16, 16
        ideal_image[cy, cx] = flux_e
        ideal_image += 10.0

        det_output = detector.process_epoch(
            ideal_image_e=ideal_image,
            epoch_index=0,
            epoch_time_mjd=60_000.0,
            prev_epoch_time_mjd=None,
        )

        t_exp_s = det.readout.exposure_time_s
        electrons_image = det_output.rate_image * t_exp_s

        # Count aperture pixels for dark-current subtraction.
        rows, cols = np.mgrid[:32, :32]
        dist = np.sqrt((cols.astype(float) - cx) ** 2 + (rows.astype(float) - cy) ** 2)
        n_aperture_pixels = int((dist <= 3.0).sum())

        # Gross aperture sum (signal + dark current in aperture).
        gross = _aperture_flux(electrons_image, cx=float(cx), cy=float(cy), radius=3.0)

        # Remove dark-current contribution inside aperture.
        dark_e_per_s = det.electrical.dark_current_e_per_s
        dark_in_aperture = dark_e_per_s * t_exp_s * n_aperture_pixels

        source_recovered = gross - dark_in_aperture
        rel_error = abs(source_recovered - flux_e) / flux_e

        assert rel_error < 0.05, (
            f"Detector flux conservation failed: injected {flux_e:.1f} e⁻, "
            f"recovered {source_recovered:.2f} e⁻ (rel. error = {rel_error:.3%}, "
            f"tolerance 5%)."
        )


# ---------------------------------------------------------------------------
# AC-4: DIA null test
# ---------------------------------------------------------------------------

class TestDIANullTest:
    """AC-4: Static-field difference must be noise-consistent with zero signal."""

    # Use n_ref=20 for a well-averaged reference, reducing its noise contribution.
    _N_REF = 20
    _CTX = 64
    _SCI_SZ = 16
    _BG_E_PER_S = 2.0

    @pytest.fixture()
    def _dia_setup(
        self,
    ) -> tuple[DIAPipeline, DetectorConfig, np.ndarray, np.ndarray]:
        """Build DIAPipeline and return (pipeline, det, reference, science_rate)."""
        det = _make_small_detector(nx=self._CTX, ny=self._CTX)
        dia_cfg = DIAConfig(
            context_stamp_size=self._CTX,
            science_stamp_size=self._SCI_SZ,
            n_reference_epochs=self._N_REF,
        )
        rng_ref = np.random.default_rng(999)
        pipeline = DIAPipeline(dia_cfg, det, rng_ref)

        # Build reference from static flat fields (no source, zero ideal electrons).
        ideal_flat = np.zeros((self._CTX, self._CTX), dtype=np.float64)
        reference = pipeline.build_reference(
            [ideal_flat] * self._N_REF,
            [self._BG_E_PER_S] * self._N_REF,
        )

        # Science: same ideal flat + independent Gaussian noise.
        t = det.readout.exposure_time_s
        rn = det.electrical.read_noise_cds_electrons
        dk = det.electrical.dark_current_e_per_s
        var_e = rn ** 2 + (dk + self._BG_E_PER_S) * t
        variance_rate = var_e / (t ** 2)

        rng_sci = np.random.default_rng(777)
        noise_sci = rng_sci.normal(0.0, float(np.sqrt(variance_rate)), size=(self._CTX, self._CTX))
        science_rate = ideal_flat / t + self._BG_E_PER_S + noise_sci

        return pipeline, det, reference, science_rate

    def test_residual_mean_consistent_with_zero(self, _dia_setup: Any) -> None:
        """|mean(residuals)| < 3 * std(residuals) / sqrt(N).

        Tests the null hypothesis that the residual mean is statistically
        indistinguishable from zero (a 3σ single-sample z-test).
        """
        pipeline, det, reference, science_rate = _dia_setup
        residuals = pipeline.subtract(science_rate, reference)

        mean_r = float(np.mean(residuals))
        std_r = float(np.std(residuals))
        n = residuals.size
        threshold = 3.0 * std_r / float(np.sqrt(n))

        assert abs(mean_r) < threshold, (
            f"DIA null test: |mean| = {abs(mean_r):.4e} >= 3·std/√N = {threshold:.4e}."
        )

    def test_residual_rms_within_2x_theoretical(self, _dia_setup: Any) -> None:
        """RMS of residuals must be within 2× the expected noise level.

        Expected theoretical std of the difference image:
            σ_diff = sqrt(σ²_science + σ²_reference)
                   = sqrt(variance_rate · (1 + 1/n_ref))
        """
        pipeline, det, reference, science_rate = _dia_setup
        residuals = pipeline.subtract(science_rate, reference)

        t = det.readout.exposure_time_s
        rn = det.electrical.read_noise_cds_electrons
        dk = det.electrical.dark_current_e_per_s
        var_e = rn ** 2 + (dk + self._BG_E_PER_S) * t
        variance_rate = var_e / (t ** 2)
        std_theoretical = float(np.sqrt(variance_rate * (1.0 + 1.0 / self._N_REF)))

        rms_actual = float(np.std(residuals))
        assert rms_actual < 2.0 * std_theoretical, (
            f"DIA null test RMS = {rms_actual:.4e} > 2× theoretical = "
            f"{2.0 * std_theoretical:.4e}."
        )


# ---------------------------------------------------------------------------
# AC-5: DIA recovery
# ---------------------------------------------------------------------------

class TestDIARecovery:
    """AC-5: Injected point source recovered within 10% via aperture photometry."""

    def test_point_source_recovered_within_10pct(self) -> None:
        """AL subtraction must recover an injected point source to within 10%%.

        Procedure:
        1. Build reference from ``n_ref`` static flat epochs (no source).
        2. Construct science image = flat + delta-function source at centre.
        3. Run AL subtraction; extract science-size stamp.
        4. Apply aperture photometry at known centroid.
        5. Convert rate-space measurement back to electrons; compare to injected flux.
        """
        ctx = 64
        sci_sz = 16
        n_ref = 20
        bg_e_per_s = 2.0
        source_flux_e = 5_000.0

        det = _make_small_detector(nx=ctx, ny=ctx)
        dia_cfg = DIAConfig(
            context_stamp_size=ctx,
            science_stamp_size=sci_sz,
            n_reference_epochs=n_ref,
        )
        rng_ref = np.random.default_rng(42)
        pipeline = DIAPipeline(dia_cfg, det, rng_ref)

        # Derived noise parameters.
        t = det.readout.exposure_time_s
        rn = det.electrical.read_noise_cds_electrons
        dk = det.electrical.dark_current_e_per_s
        var_e = rn ** 2 + (dk + bg_e_per_s) * t
        variance_rate = var_e / (t ** 2)

        # Build reference: no source in any epoch.
        ideal_flat = np.zeros((ctx, ctx), dtype=np.float64)
        reference = pipeline.build_reference(
            [ideal_flat] * n_ref, [bg_e_per_s] * n_ref
        )

        # Science: flat background + noise + point source at centre.
        rng_sci = np.random.default_rng(123)
        noise_sci = rng_sci.normal(
            0.0, float(np.sqrt(variance_rate)), size=(ctx, ctx)
        )
        science_rate = ideal_flat / t + bg_e_per_s + noise_sci

        cy, cx = ctx // 2, ctx // 2
        science_rate[cy, cx] += source_flux_e / t  # add source in rate space

        # DIA subtract and crop.
        diff_full = pipeline.subtract(science_rate, reference)
        diff_stamp = pipeline.extract_stamp(diff_full)

        # Source centroid in the extracted stamp.
        sci_cx = float(sci_sz // 2)
        sci_cy = float(sci_sz // 2)

        # Aperture photometry (rate space) → electrons.
        recovered_rate = _aperture_flux(
            diff_stamp, cx=sci_cx, cy=sci_cy, radius=3.0
        )
        recovered_electrons = recovered_rate * t

        rel_error = abs(recovered_electrons - source_flux_e) / source_flux_e
        assert rel_error < 0.10, (
            f"DIA recovery failed: injected {source_flux_e:.1f} e⁻, "
            f"recovered {recovered_electrons:.2f} e⁻ "
            f"(rel. error = {rel_error:.3%}, tolerance 10%%)."
        )


# ---------------------------------------------------------------------------
# Shared fixtures for seed tests (AC-6 / AC-7)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sim_config() -> SimulationConfig:
    """Small 64×64 / 16×16 SimulationConfig for seed determinism tests."""
    return _make_sim_config(ctx=64, sci_sz=16, n_ref=5)


def _run_simulator(
    config: SimulationConfig,
    master_seed: int,
    event_id: str,
    n_sci: int = 3,
    flux_e: float = 5_000.0,
) -> EventSceneOutput:
    """Convenience wrapper: construct SceneSimulator and run simulate_event."""
    sim = SceneSimulator(config, master_seed=master_seed)
    return sim.simulate_event(
        event_id=event_id,
        source_params_sequence=_make_source_params(n_sci, flux_e=flux_e),
        timestamps_mjd=_make_timestamps(n_sci),
        backgrounds_e_per_s=[1.0] * n_sci,
    )


# ---------------------------------------------------------------------------
# AC-6: Seed determinism
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    """AC-6: Same (master_seed, event_id) → identical outputs."""

    def test_difference_stamps_allclose(self, sim_config: SimulationConfig) -> None:
        """Float difference stamps: assert_allclose(rtol=1e-6, atol=1e-8)."""
        out1 = _run_simulator(sim_config, master_seed=42, event_id="det_float_001")
        out2 = _run_simulator(sim_config, master_seed=42, event_id="det_float_001")
        np.testing.assert_allclose(
            out1.difference_stamps,
            out2.difference_stamps,
            rtol=1e-6,
            atol=1e-8,
            err_msg="difference_stamps differ across two runs with same seed.",
        )

    def test_saturation_masks_bit_identical(self, sim_config: SimulationConfig) -> None:
        """Bool saturation masks: assert_array_equal (exact)."""
        out1 = _run_simulator(sim_config, master_seed=42, event_id="det_bool_001")
        out2 = _run_simulator(sim_config, master_seed=42, event_id="det_bool_001")
        np.testing.assert_array_equal(
            out1.saturation_stamps,
            out2.saturation_stamps,
            err_msg="saturation_stamps differ across two runs with same seed.",
        )

    def test_cr_masks_bit_identical(self, sim_config: SimulationConfig) -> None:
        """Bool CR masks: assert_array_equal (exact)."""
        out1 = _run_simulator(sim_config, master_seed=42, event_id="det_bool_002")
        out2 = _run_simulator(sim_config, master_seed=42, event_id="det_bool_002")
        np.testing.assert_array_equal(
            out1.cr_stamps,
            out2.cr_stamps,
            err_msg="cr_stamps differ across two runs with same seed.",
        )

    def test_provenance_json_identical(self, sim_config: SimulationConfig) -> None:
        """model_dump(mode='json') must match for every provenance record."""
        out1 = _run_simulator(sim_config, master_seed=99, event_id="det_prov_001")
        out2 = _run_simulator(sim_config, master_seed=99, event_id="det_prov_001")
        assert len(out1.provenance) == len(out2.provenance)
        for i, (r1, r2) in enumerate(zip(out1.provenance, out2.provenance)):
            assert r1.model_dump(mode="json") == r2.model_dump(mode="json"), (
                f"Provenance record mismatch at epoch {i}: seed determinism broken."
            )


# ---------------------------------------------------------------------------
# AC-7: Seed independence
# ---------------------------------------------------------------------------

class TestSeedIndependence:
    """AC-7: Different event_id → statistically different pixel distributions."""

    def test_different_events_ks_pvalue_below_0pt05(
        self, sim_config: SimulationConfig
    ) -> None:
        """KS test on pooled difference-stamp pixels must yield p-value < 0.05.

        Pixels across all 3 science epochs are pooled to maximise test power
        when the individual 16×16 stamps are small.  A bright source
        (flux_e=10000) ensures the per-event pixel distributions carry a clear
        signal above the noise floor.
        """
        sim = SceneSimulator(sim_config, master_seed=42)
        n_sci = 3
        flux_e = 10_000.0

        out_a = sim.simulate_event(
            event_id="ks_alpha",
            source_params_sequence=_make_source_params(n_sci, flux_e=flux_e),
            timestamps_mjd=_make_timestamps(n_sci),
            backgrounds_e_per_s=[1.0] * n_sci,
        )
        out_b = sim.simulate_event(
            event_id="ks_beta",
            source_params_sequence=_make_source_params(n_sci, flux_e=flux_e),
            timestamps_mjd=_make_timestamps(n_sci),
            backgrounds_e_per_s=[1.0] * n_sci,
        )

        # Pool all epochs to increase KS test sample size.
        pixels_a = out_a.difference_stamps.ravel()
        pixels_b = out_b.difference_stamps.ravel()

        ks_stat, pvalue = ks_2samp(pixels_a, pixels_b)
        assert pvalue < 0.05, (
            f"KS test failed: p-value = {pvalue:.4f} >= 0.05 "
            f"(KS stat = {ks_stat:.4f}).  Different event IDs produced "
            "statistically indistinguishable pixel distributions."
        )


# ---------------------------------------------------------------------------
# AC-8: Memory budget
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMemoryBudget:
    """AC-8: Full pipeline on 64×64 / 3-epoch must stay under 1 GB peak memory."""

    def test_peak_tracemalloc_under_1gb(self) -> None:
        """Use tracemalloc to measure peak Python-layer memory during pipeline run.

        Limit: 1 GiB (1,073,741,824 bytes).
        Config: 64×64 context, 16×16 science, 3 science epochs, 5 reference epochs.
        """
        config = _make_sim_config(ctx=64, sci_sz=16, n_ref=5)
        sim = SceneSimulator(config, master_seed=1)

        tracemalloc.start()
        try:
            sim.simulate_event(
                event_id="mem_budget_event",
                source_params_sequence=_make_source_params(3, flux_e=5_000.0),
                timestamps_mjd=_make_timestamps(3),
                backgrounds_e_per_s=[1.0] * 3,
            )
            _current, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_gib = peak_bytes / (1024 ** 3)
        assert peak_gib < 1.0, (
            f"Peak tracemalloc memory {peak_gib:.4f} GiB exceeds 1 GiB limit."
        )


# ---------------------------------------------------------------------------
# AC-9: Architecture boundary (Python AST)
# ---------------------------------------------------------------------------

class TestArchitectureBoundary:
    """AC-9: Rendering code may only import smig.sensor.detector; sensor may not import galsim."""

    # The one permitted import from the sensor package.
    _ALLOWED_SENSOR_MODULE = "smig.sensor.detector"

    # Leaf modules that must NEVER appear in rendering imports.
    _EXPLICITLY_FORBIDDEN = {
        "smig.sensor.ipc",
        "smig.sensor.readout",
        "smig.sensor.persistence",
        "smig.sensor.nonlinearity",
        "smig.sensor.charge_diffusion",
        "smig.sensor.noise",
        "smig.sensor.calibration",
        "smig.sensor.memory_profiler",
    }

    @staticmethod
    def _collect_py_files(root: Path) -> list[Path]:
        return sorted(root.rglob("*.py"))

    @staticmethod
    def _extract_imports(filepath: Path) -> list[tuple[str, str]]:
        """Parse *filepath* with the AST and return (module_name, names_str) pairs.

        Each entry represents one import node.  For ``import X``, ``names_str``
        is ``""``; for ``from X import Y, Z``, it is ``"Y, Z"``.
        """
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        found: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.append((alias.name, ""))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(a.name for a in (node.names or []))
                found.append((module, names))
        return found

    def test_rendering_only_imports_detector_from_sensor(self) -> None:
        """All smig/rendering/**/*.py: only smig.sensor.detector is allowed.

        Uses Python's ast module exclusively (no grep / subprocess).
        """
        rendering_root = Path(__file__).parent.parent
        violations: list[str] = []

        for filepath in self._collect_py_files(rendering_root):
            rel = filepath.relative_to(rendering_root.parent.parent)
            for module, names in self._extract_imports(filepath):
                if not module.startswith("smig.sensor."):
                    continue
                if module == self._ALLOWED_SENSOR_MODULE:
                    continue
                violations.append(
                    f"{rel}: forbidden sensor import '{module}'"
                    + (f" (names: {names})" if names else "")
                )

        assert not violations, (
            "Architecture boundary violated — rendering code imports forbidden "
            "sensor leaf modules:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_explicitly_forbidden_sensor_leaves_absent(self) -> None:
        """Explicitly verify the known forbidden leaf modules are never imported."""
        rendering_root = Path(__file__).parent.parent

        for filepath in self._collect_py_files(rendering_root):
            for module, names in self._extract_imports(filepath):
                for forbidden in self._EXPLICITLY_FORBIDDEN:
                    assert not module.startswith(forbidden), (
                        f"{filepath.name}: explicitly forbidden import "
                        f"'{module}' (matches forbidden prefix '{forbidden}')"
                    )

    def test_sensor_modules_do_not_import_galsim(self) -> None:
        """No file in smig/sensor/ may import galsim.

        GalSim rendering happens upstream in the rendering layer; sensor modules
        receive and return plain numpy arrays and must remain GalSim-free.
        Uses Python's ast module exclusively (no grep / subprocess).
        """
        sensor_root = Path(__file__).parent.parent.parent / "sensor"
        assert sensor_root.exists(), f"smig/sensor/ not found at {sensor_root}"

        violations: list[str] = []
        for filepath in self._collect_py_files(sensor_root):
            for module, names in self._extract_imports(filepath):
                if "galsim" in module.lower():
                    violations.append(
                        f"{filepath.name}: import '{module}'"
                        + (f" names={names}" if names else "")
                    )

        assert not violations, (
            "Sensor modules must not import galsim.  Violations:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# AC-10: validate_splits.py CLI unit tests
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "scripts"
_VALIDATE_SCRIPT = _SCRIPTS_DIR / "validate_splits.py"


def _run_validate_script(manifest_data: dict, tmp_path: Path) -> int:
    """Write *manifest_data* to a temp file, run validate_splits.py, return exit code."""
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))
    result = subprocess.run(
        [sys.executable, str(_VALIDATE_SCRIPT), str(manifest_file)],
        capture_output=True,
    )
    return result.returncode


@pytest.mark.skipif(
    not _VALIDATE_SCRIPT.exists(),
    reason="scripts/validate_splits.py not yet created",
)
class TestValidateSplitsCLI:
    """AC-10: validate_splits.py returns correct exit codes for leaky/clean manifests."""

    def test_clean_manifest_exits_zero(self, tmp_path: Path) -> None:
        """No violations in a well-separated manifest → exit 0."""
        manifest = {
            "events": [
                {
                    "event_id": "ev001",
                    "split": "train",
                    "starfield_seed": 1,
                    "params": {"t_E": 30.0, "u_0": 0.10, "s": 1.0, "q": 0.001},
                },
                {
                    "event_id": "ev002",
                    "split": "val",
                    "starfield_seed": 2,
                    "params": {"t_E": 60.0, "u_0": 0.50, "s": 1.2, "q": 0.010},
                },
                {
                    "event_id": "ev003",
                    "split": "test",
                    "starfield_seed": 3,
                    "params": {"t_E": 10.0, "u_0": 0.80, "s": 0.8, "q": 0.100},
                },
            ]
        }
        assert _run_validate_script(manifest, tmp_path) == 0

    def test_duplicate_event_id_across_splits_exits_one(self, tmp_path: Path) -> None:
        """Same event_id in two different splits → exit 1."""
        manifest = {
            "events": [
                {
                    "event_id": "ev_dup",
                    "split": "train",
                    "starfield_seed": 10,
                    "params": {"t_E": 30.0, "u_0": 0.1, "s": 1.0, "q": 0.001},
                },
                {
                    "event_id": "ev_dup",
                    "split": "val",
                    "starfield_seed": 20,
                    "params": {"t_E": 60.0, "u_0": 0.5, "s": 1.2, "q": 0.010},
                },
            ]
        }
        assert _run_validate_script(manifest, tmp_path) == 1

    def test_shared_starfield_seed_across_splits_exits_one(
        self, tmp_path: Path
    ) -> None:
        """Same starfield_seed in two different splits → exit 1."""
        manifest = {
            "events": [
                {
                    "event_id": "ev_a",
                    "split": "train",
                    "starfield_seed": 42,
                    "params": {"t_E": 30.0, "u_0": 0.1, "s": 1.0, "q": 0.001},
                },
                {
                    "event_id": "ev_b",
                    "split": "test",
                    "starfield_seed": 42,
                    "params": {"t_E": 60.0, "u_0": 0.5, "s": 1.2, "q": 0.010},
                },
            ]
        }
        assert _run_validate_script(manifest, tmp_path) == 1

    def test_param_similar_events_in_different_splits_exits_one(
        self, tmp_path: Path
    ) -> None:
        """Events with all params within 5%% in different splits → exit 1."""
        manifest = {
            "events": [
                {
                    "event_id": "ev_c",
                    "split": "train",
                    "starfield_seed": 100,
                    # Baseline params.
                    "params": {"t_E": 30.0, "u_0": 0.100, "s": 1.000, "q": 0.001000},
                },
                {
                    "event_id": "ev_d",
                    "split": "val",
                    "starfield_seed": 200,
                    # All within ~3% of ev_c → similarity check must trigger.
                    "params": {"t_E": 30.5, "u_0": 0.102, "s": 1.010, "q": 0.001010},
                },
            ]
        }
        assert _run_validate_script(manifest, tmp_path) == 1

    def test_param_similar_events_in_same_split_exits_zero(
        self, tmp_path: Path
    ) -> None:
        """Similar params in the SAME split are allowed (no cross-split leakage) → exit 0."""
        manifest = {
            "events": [
                {
                    "event_id": "ev_e",
                    "split": "train",
                    "starfield_seed": 300,
                    "params": {"t_E": 30.0, "u_0": 0.100, "s": 1.000, "q": 0.001000},
                },
                {
                    "event_id": "ev_f",
                    "split": "train",
                    "starfield_seed": 400,
                    # All within ~2% of ev_e, but SAME split → OK.
                    "params": {"t_E": 30.3, "u_0": 0.101, "s": 1.005, "q": 0.001010},
                },
            ]
        }
        assert _run_validate_script(manifest, tmp_path) == 0
