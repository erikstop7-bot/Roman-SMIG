"""
smig/microlensing/validation/test_binding.py
============================================
Tests for :func:`~smig.microlensing.binding.bind_event_to_source`.

Coverage
--------
AC-B1  Contract: return type is list[dict] with "flux_e" float in every entry.
AC-B2  PSPL peak: flux_e[peak] / flux_e[baseline] matches analytic A_peak /
       A_baseline within 2%.
AC-B3  Null DIA: u0=100 event produces DIA difference residuals consistent with
       noise-only (per-epoch MAD normalization; median(r) < 0.8, P99(r) < 10.0).
       Requires galsim + pandas; skipped otherwise.

Run from the SMIG project root::

    python -m pytest smig/microlensing/validation/test_binding.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from smig.catalogs.base import StarRecord
from smig.catalogs.photometry import mag_ab_to_electrons  # noqa: F401 — used in AC-B3
from smig.microlensing.binding import bind_event_to_source
from smig.microlensing.event import EventClass, MicrolensingEvent

# These imports are safe even without galsim (pipeline.py guards galsim lazily).
from smig.config.optics_schemas import (
    CrowdedFieldConfig,
    DIAConfig,
    PSFConfig,
    RenderingConfig,
    SimulationConfig,
)
from smig.config.schemas import DetectorConfig
from smig.rendering.pipeline import SceneSimulator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_star(mag: float = 20.0) -> StarRecord:
    return StarRecord(
        galactic_l_deg=1.0,
        galactic_b_deg=-3.0,
        distance_kpc=8.0,
        mass_msun=1.0,
        teff_K=5800.0,
        log_g=4.5,
        metallicity_feh=0.0,
        mag_F146_ab=mag,
        source_id="test_star_0",
        catalog_tile_id="tile_0",
    )


def _make_pspl_event(
    u0: float,
    t0_mjd: float = 60_000.0,
    tE_days: float = 30.0,
) -> MicrolensingEvent:
    return MicrolensingEvent(
        event_id="test_event",
        t0_mjd=t0_mjd,
        tE_days=tE_days,
        u0=u0,
        rho=0.0,
        alpha_rad=0.0,
        event_class=EventClass.PSPL,
    )


def _make_sim_config(ctx: int = 64, sci_sz: int = 16, n_ref: int = 5) -> SimulationConfig:
    default_det = DetectorConfig()
    det_dict = default_det.model_dump()
    det_dict["geometry"]["nx"] = ctx
    det_dict["geometry"]["ny"] = ctx
    small_det = DetectorConfig.model_validate(det_dict)
    return SimulationConfig(
        detector=small_det,
        # fov_native_pixels must match ctx; PSFConfig default is 128, which violates
        # the SimulationConfig cross-field constraint psf.fov_native_pixels <= detector.geometry.nx
        # for the small (ctx=64) test profile and would raise validation failure.
        psf=PSFConfig(
            oversample=1,
            n_wavelengths=2,
            jitter_rms_mas=0.0,
            fov_native_pixels=ctx,
        ),
        rendering=RenderingConfig(),
        crowded_field=CrowdedFieldConfig(stamp_size=sci_sz, pixel_scale_arcsec=0.11),
        dia=DIAConfig(
            n_reference_epochs=n_ref,
            context_stamp_size=ctx,
            science_stamp_size=sci_sz,
        ),
    )


def _pspl_magnification(t: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
    return (u ** 2 + 2.0) / (u * np.sqrt(u ** 2 + 4.0))


# ---------------------------------------------------------------------------
# AC-B1  Contract test
# ---------------------------------------------------------------------------

def test_return_type_is_list_of_dict() -> None:
    """Return is list[dict] with "flux_e" float in every entry."""
    event = _make_pspl_event(u0=1.0)
    star = _make_star()
    t = np.linspace(59_990.0, 60_010.0, 5)
    result = bind_event_to_source(event, star, t, exposure_s=100.0)

    assert isinstance(result, list)
    assert len(result) == len(t)
    assert all(isinstance(x, dict) and "flux_e" in x for x in result)
    assert all(isinstance(x["flux_e"], float) for x in result)


# ---------------------------------------------------------------------------
# AC-B2  PSPL peak test (no SceneSimulator needed)
# ---------------------------------------------------------------------------

def test_pspl_peak_ratio() -> None:
    """flux_e[peak] / flux_e[baseline] matches analytic A_peak / A_baseline within 2%."""
    u0 = 1e-3
    t0_mjd = 60_000.0
    tE_days = 30.0
    exposure_s = 100.0

    event = _make_pspl_event(u0=u0, t0_mjd=t0_mjd, tE_days=tE_days)
    star = _make_star(mag=20.0)

    # 11 epochs: index 0 at peak (t=t0), indices 1-10 far from peak
    t_peak = np.array([t0_mjd])
    t_far = t0_mjd + np.linspace(5.0 * tE_days, 15.0 * tE_days, 10)
    epoch_times_mjd = np.concatenate([t_peak, t_far])
    assert len(epoch_times_mjd) == 11

    result = bind_event_to_source(event, star, epoch_times_mjd, exposure_s)
    flux_e = np.array([d["flux_e"] for d in result])

    peak_flux = flux_e[0]
    baseline_flux = np.mean(flux_e[1:])

    A_vals = _pspl_magnification(epoch_times_mjd, t0_mjd, tE_days, u0)
    analytic_ratio = A_vals[0] / np.mean(A_vals[1:])
    measured_ratio = peak_flux / baseline_flux

    assert abs(measured_ratio - analytic_ratio) / analytic_ratio < 0.02, (
        f"Peak ratio {measured_ratio:.6f} deviates from analytic {analytic_ratio:.6f} "
        f"by {abs(measured_ratio - analytic_ratio) / analytic_ratio * 100:.3f}% (limit 2%)."
    )


# ---------------------------------------------------------------------------
# AC-B3  Null DIA residuals (requires galsim + pandas)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_null_event_dia_residuals() -> None:
    """u0=100 → A≈1; DIA difference residuals are noise-only.

    Statistic choice: per-epoch MAD normalization, pooled across epochs.
    Computing MAD independently per epoch lets σ adapt to each frame's noise
    level before pooling r = |diff| / σ for the tail gate.

    Correctness requirements addressed in this test:

    1. ``baseline_source_flux_e`` is passed so reference epochs include the
       unlensed source at F₀.  Without it the full source signal leaks into the
       difference image (≈153 e⁻/s at the PSF peak, σ≈0.26 e⁻/s → r≈600)
       and P99 ≫ 10 regardless of AL quality.

    2. ``context_stamp_size=64`` satisfies the AL kernel size constraint
       (kernel_size=33 px < 64 px); the boundary-free interior covers the
       16-px science stamp.

    Marked @pytest.mark.integration: requires galsim + pandas and runs the
    full SceneSimulator pipeline (~seconds).
    Excluded from fast CI (-m "not integration").

    Gates: median(r) < 0.8 catches systematic bias; P99 < 10 is lenient for
    AL-correlation tails (non-i.i.d. by construction) while blocking
    catastrophic failures.
    """
    pytest.importorskip("galsim", reason="galsim required for null DIA test")
    pytest.importorskip("pandas", reason="pandas required for null DIA test")

    u0 = 100.0
    t0_mjd = 60_000.0
    exposure_s = 100.0
    n_epochs = 5

    event = _make_pspl_event(u0=u0, t0_mjd=t0_mjd)
    star = _make_star(mag=22.0)  # faint → modest flux, avoids saturation

    epoch_times_mjd = np.linspace(t0_mjd - 2.0, t0_mjd + 2.0, n_epochs)
    source_params = bind_event_to_source(event, star, epoch_times_mjd, exposure_s)

    cfg = _make_sim_config(ctx=64, sci_sz=16, n_ref=5)
    sim = SceneSimulator(cfg, master_seed=0)
    # Baseline F₀: unlensed source flux (A=1).  For u0=100 all source_params
    # have A≈1 already, so source_params[0]["flux_e"] ≈ F₀ to six significant
    # figures.  Passing this makes reference epochs physically correct: they
    # contain the source at F₀ just like a real pre-event template.
    baseline_flux_e: float = source_params[0]["flux_e"]
    out = sim.simulate_event(
        event_id="null_test",
        source_params_sequence=source_params,
        timestamps_mjd=epoch_times_mjd,
        backgrounds_e_per_s=[0.1] * n_epochs,
        baseline_source_flux_e=baseline_flux_e,
    )

    dia_cube = out.difference_stamps  # (n_epochs, sci_sz, sci_sz)

    # Per-epoch MAD: normalize each frame independently so that epoch-to-epoch
    # noise variations don't inflate the global σ estimate.
    r_parts: list[np.ndarray] = []
    for frame in dia_cube:
        sigma_e = scipy.stats.median_abs_deviation(frame, axis=None, scale="normal")
        if sigma_e == 0.0:
            # Perfectly flat epoch — trivially passes, skip to avoid division by zero.
            continue
        r_parts.append(np.abs(frame) / sigma_e)

    if not r_parts:
        # All epochs flat — trivially passes.
        return

    r = np.concatenate([rp.ravel() for rp in r_parts])

    # Diagnostic: print empirical quantiles to help calibrate future tightening.
    print(
        f"\nnull DIA r-stat (per-epoch MAD): "
        f"median={np.median(r):.3f}  "
        f"P95={np.percentile(r, 95):.2f}  "
        f"P99={np.percentile(r, 99):.2f}  "
        f"P99.5={np.percentile(r, 99.5):.2f}"
    )

    assert np.median(r) < 0.8, (
        f"Null test: median(|r|) = {np.median(r):.3f} >= 0.8 — "
        "systematic bias in DIA residuals."
    )
    # P99 gate: lenient for AL-correlation tails (non-i.i.d. by construction);
    # catches gross pipeline failures without rejecting correct noise realizations.
    assert np.percentile(r, 99) < 10.0, (
        f"Null test: P99(|r|) = {np.percentile(r, 99):.3f} >= 10.0 — "
        "residual tails far too heavy for a null PSPL event."
    )
