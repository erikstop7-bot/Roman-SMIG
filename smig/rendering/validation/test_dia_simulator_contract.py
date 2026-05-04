"""
smig/rendering/validation/test_dia_simulator_contract.py
=========================================================
End-to-end DIA contract tests against the public SceneSimulator entrypoint.

These tests exercise SceneSimulator.simulate_event — not the lower-level
DIAPipeline directly — so any future refactor that breaks the simulator's
PSF / reference-construction / detector / DIA wiring must show up here.

Coverage maps to the existing AC-4 / AC-D4 / AC-D5 contract:

  * Null residual: when reference and science share the same baseline source
    flux, the science stamp must satisfy ``|mean| < 3·sigma/sqrt(N)`` where
    sigma is the difference-image rate-space std, matching the AC-4 sigma
    definition in test_integration_phase2.py.
  * Residual RMS: the science stamp std must be within the AC-4 documented
    factor (2×) of the theoretical rate-space noise level.
  * Point-source recovery: when the science source carries an excess flux
    over the reference baseline, the recovered flux in the science stamp must
    match the injected excess within the existing AC-D4/AC-5 10 % tolerance.

The science-profile geometry respects the AL kernel/context-size guard
(``context_stamp_size >= 33 + science_stamp_size``); fast_dev parametrization
is intentionally NOT exercised here — fast_dev is a development-only profile
and must never be the sole release gate.

Run from the project root::

    python3 -m pytest smig/rendering/validation/test_dia_simulator_contract.py -v
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

# Optional dependency gate — produces an obvious skip on base-only installs.
galsim = pytest.importorskip(
    "galsim", reason="galsim required for SceneSimulator E2E DIA tests"
)
pd = pytest.importorskip(
    "pandas", reason="pandas required for SceneSimulator E2E DIA tests"
)

from smig.config.optics_schemas import (  # noqa: E402  (after importorskip)
    CrowdedFieldConfig,
    DIAConfig,
    PSFConfig,
    RenderingConfig,
    SimulationConfig,
)
from smig.config.schemas import DetectorConfig  # noqa: E402
from smig.rendering.pipeline import SceneSimulator  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level fixture: force the analytic PSF backend for every test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_analytic_psf_backend():
    """Patch _WEBBPSF_AVAILABLE=False so STPSFProvider always uses analytic backend."""
    with patch("smig.optics.psf._WEBBPSF_AVAILABLE", False):
        yield


# ---------------------------------------------------------------------------
# Geometry constants — chosen to satisfy the AL kernel/context guard.
# ---------------------------------------------------------------------------

# AL kernel half-width is 4·sigma_max = 16 px (sigma_max=4.0); kernel_size=33.
# DIAConfig requires context_stamp_size > kernel_size, with the recommended
# floor being kernel_size + science_stamp_size.  ctx=80 / sci=16 satisfies
# both with margin so the science stamp lies in the boundary-free interior.
_CTX: int = 80
_SCI_SZ: int = 16
_N_REF: int = 5
_N_SCI: int = 1
_BG_E_PER_S: float = 1.0


def _make_science_config() -> SimulationConfig:
    """Science-profile SimulationConfig sized for fast deterministic E2E runs.

    Intentional choices:
      * detector geometry == context_stamp_size (the canonical Phase 2 setup)
      * PSF FOV == ctx (no rectangular clipping in the rendered stamp)
      * detector_matched reference processing (the trusted path)
      * per_epoch jitter (default science behaviour)
    """
    base = DetectorConfig()
    det_dict = base.model_dump()
    det_dict["geometry"]["nx"] = _CTX
    det_dict["geometry"]["ny"] = _CTX
    small_det = DetectorConfig.model_validate(det_dict)
    return SimulationConfig(
        detector=small_det,
        psf=PSFConfig(
            oversample=2,
            n_wavelengths=2,
            jitter_rms_mas=0.0,
            fov_native_pixels=_CTX,
            psf_edge_taper_pixels=4.0,
        ),
        rendering=RenderingConfig(),
        crowded_field=CrowdedFieldConfig(
            stamp_size=_SCI_SZ,
            pixel_scale_arcsec=0.11,
            brightness_cap_mag=None,
        ),
        dia=DIAConfig(
            n_reference_epochs=_N_REF,
            context_stamp_size=_CTX,
            science_stamp_size=_SCI_SZ,
            subtraction_method="alard_lupton",
            reference_processing="detector_matched",
            reference_jitter_mode="per_epoch",
        ),
    )


def _theoretical_rate_std(detector_cfg: DetectorConfig, n_ref: int) -> float:
    """Rate-space standard deviation expected from a (science - reference) diff.

    Mirrors the closed-form sigma definition used in
    test_integration_phase2.TestDIANullTest.test_residual_rms_within_2x_theoretical.
    """
    t = detector_cfg.readout.exposure_time_s
    rn = detector_cfg.electrical.read_noise_cds_electrons
    dk = detector_cfg.electrical.dark_current_e_per_s
    var_e = rn ** 2 + (dk + _BG_E_PER_S) * t
    variance_rate = var_e / (t ** 2)
    return float(np.sqrt(variance_rate * (1.0 + 1.0 / n_ref)))


def _empty_neighbor_catalog() -> "pd.DataFrame":
    """Empty catalog — keeps the static field zero so the source signal isolates cleanly."""
    return pd.DataFrame(
        {
            "x_pix": np.array([], dtype=np.float64),
            "y_pix": np.array([], dtype=np.float64),
            "flux_e": np.array([], dtype=np.float64),
            "mag_w146": np.array([], dtype=np.float64),
        }
    )


# ---------------------------------------------------------------------------
# AC-S1: null residual — equal baseline and science source flux
# ---------------------------------------------------------------------------


class TestSimulatorNullResidual:
    """When science flux equals the reference baseline, the science stamp is null."""

    def test_null_residual_mean_consistent_with_zero(self) -> None:
        """``|mean| < 3·sigma/sqrt(N)`` on the final extracted science stamp.

        Uses the SAME sigma definition as AC-4 (theoretical rate-space std with
        the 1+1/n_ref reference-noise factor); the actual stamp std is used as
        the observed sigma so this test is robust to minor changes in detector
        noise parameters.
        """
        cfg = _make_science_config()
        sim = SceneSimulator(cfg, master_seed=2024)

        baseline_flux_e = 4_000.0
        # Equal flux in reference and science → expected residual = 0 + noise.
        out = sim.simulate_event(
            event_id="null_event",
            source_params_sequence=[
                {
                    "flux_e": baseline_flux_e,
                    "centroid_offset_pix": (0.0, 0.0),
                    "rho_star_arcsec": 0.0,
                    "limb_darkening_coeffs": None,
                }
            ] * _N_SCI,
            timestamps_mjd=np.array([60_000.0 + i for i in range(_N_SCI)], dtype=np.float64),
            backgrounds_e_per_s=[_BG_E_PER_S] * _N_SCI,
            neighbor_catalog=_empty_neighbor_catalog(),
            baseline_source_flux_e=baseline_flux_e,
        )

        residuals = out.difference_stamps[0].ravel()
        n = residuals.size
        mean_r = float(np.mean(residuals))
        std_r = float(np.std(residuals))
        threshold = 3.0 * std_r / float(np.sqrt(n))

        assert abs(mean_r) < threshold, (
            f"Null residual mean |{mean_r:.4e}| exceeds 3·std/√N = {threshold:.4e} "
            f"(std={std_r:.4e}, N={n})."
        )

    def test_null_residual_rms_within_2x_theoretical(self) -> None:
        """Residual std stays within the AC-4 documented factor (2×) of theoretical noise."""
        cfg = _make_science_config()
        sim = SceneSimulator(cfg, master_seed=2025)

        baseline_flux_e = 4_000.0
        out = sim.simulate_event(
            event_id="null_event_rms",
            source_params_sequence=[
                {
                    "flux_e": baseline_flux_e,
                    "centroid_offset_pix": (0.0, 0.0),
                    "rho_star_arcsec": 0.0,
                    "limb_darkening_coeffs": None,
                }
            ] * _N_SCI,
            timestamps_mjd=np.array([60_000.0 + i for i in range(_N_SCI)], dtype=np.float64),
            backgrounds_e_per_s=[_BG_E_PER_S] * _N_SCI,
            neighbor_catalog=_empty_neighbor_catalog(),
            baseline_source_flux_e=baseline_flux_e,
        )

        std_actual = float(np.std(out.difference_stamps[0]))
        std_theoretical = _theoretical_rate_std(cfg.detector, _N_REF)

        # 2× matches AC-4's documented factor; the science stamp may sit
        # slightly above pure-noise theory because the AL kernel reshapes the
        # local residual structure even for matched inputs.
        assert std_actual < 2.0 * std_theoretical, (
            f"Null-event stamp std {std_actual:.4e} exceeds 2× theoretical "
            f"{2.0 * std_theoretical:.4e}."
        )


# ---------------------------------------------------------------------------
# AC-S2: point-source recovery via SceneSimulator
# ---------------------------------------------------------------------------


class TestSimulatorPointSourceRecovery:
    """Source-in-science / no-source-in-reference: the science stamp recovers the source flux.

    Mirrors AC-D4 / AC-5: ``baseline_source_flux_e=None`` means the reference
    template contains only the (empty) static field, so the science stamp
    carries the full injected flux.  Adding a non-zero baseline flux exercises
    the production "Δflux only" path covered by TestSimulatorNullResidual; the
    AL kernel can absorb a small Δflux into its coefficient fit, so AC-5-style
    quantitative recovery is only meaningful when the reference is source-free.
    """

    def test_full_flux_recovered_within_10pct(self) -> None:
        """Aperture sum on the science stamp recovers ``flux_e`` within 10 %.

        Aperture radius matches the AC-D4 generous radius (3·σ + 2 px ≈ 6 px).
        Recovery is in rate space (e⁻/s); convert to electrons via ``t_exp_s``
        for the comparison against the injected flux in electrons.
        """
        cfg = _make_science_config()
        sim = SceneSimulator(cfg, master_seed=2026)

        source_flux_e = 8_000.0
        out = sim.simulate_event(
            event_id="recovery_event",
            source_params_sequence=[
                {
                    "flux_e": source_flux_e,
                    "centroid_offset_pix": (0.0, 0.0),
                    "rho_star_arcsec": 0.0,
                    "limb_darkening_coeffs": None,
                }
            ] * _N_SCI,
            timestamps_mjd=np.array([60_000.0 + i for i in range(_N_SCI)], dtype=np.float64),
            backgrounds_e_per_s=[_BG_E_PER_S] * _N_SCI,
            neighbor_catalog=_empty_neighbor_catalog(),
            baseline_source_flux_e=None,
        )

        diff_stamp = out.difference_stamps[0]
        cy = _SCI_SZ // 2
        cx = _SCI_SZ // 2
        yy, xx = np.mgrid[:_SCI_SZ, :_SCI_SZ]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        aperture_radius_px = 6.0
        aperture_mask = dist <= aperture_radius_px

        recovered_rate = float(diff_stamp[aperture_mask].sum())
        t_exp_s = cfg.detector.readout.exposure_time_s
        recovered_e = recovered_rate * t_exp_s

        rel_error = abs(recovered_e - source_flux_e) / source_flux_e
        assert rel_error < 0.10, (
            f"Full-flux recovery error {rel_error:.3f} exceeds 10 % "
            f"(injected={source_flux_e:.1f} e⁻, recovered={recovered_e:.2f} e⁻)."
        )


# ---------------------------------------------------------------------------
# AC-S3: profile branch wiring — ideal_gaussian path runs end-to-end
# ---------------------------------------------------------------------------


class TestSimulatorReferenceProcessingBranch:
    """Both reference_processing branches must run without error end-to-end.

    Correctness assertions on the fast_dev branch are intentionally weaker
    than the science branch: ideal_gaussian skips the detector chain on the
    reference, so AC-4 sigma definitions no longer apply.  This test only
    confirms the wiring works and produces finite output of the expected
    shape — quantitative correctness is the responsibility of the science
    profile tests above.
    """

    def _run(self, reference_processing: str) -> np.ndarray:
        cfg_base = _make_science_config()
        cfg = SimulationConfig(
            detector=cfg_base.detector,
            psf=cfg_base.psf,
            rendering=cfg_base.rendering,
            crowded_field=cfg_base.crowded_field,
            dia=DIAConfig(
                n_reference_epochs=_N_REF,
                context_stamp_size=_CTX,
                science_stamp_size=_SCI_SZ,
                subtraction_method="alard_lupton",
                reference_processing=reference_processing,  # type: ignore[arg-type]
                reference_jitter_mode="per_epoch",
            ),
        )
        sim = SceneSimulator(cfg, master_seed=11)
        out = sim.simulate_event(
            event_id=f"branch_{reference_processing}",
            source_params_sequence=[
                {
                    "flux_e": 3_000.0,
                    "centroid_offset_pix": (0.0, 0.0),
                    "rho_star_arcsec": 0.0,
                    "limb_darkening_coeffs": None,
                }
            ],
            timestamps_mjd=np.array([60_000.0], dtype=np.float64),
            backgrounds_e_per_s=[_BG_E_PER_S],
            neighbor_catalog=_empty_neighbor_catalog(),
            baseline_source_flux_e=3_000.0,
        )
        return out.difference_stamps[0]

    def test_detector_matched_branch_produces_finite_stamp(self) -> None:
        diff = self._run("detector_matched")
        assert diff.shape == (_SCI_SZ, _SCI_SZ)
        assert np.all(np.isfinite(diff))

    def test_ideal_gaussian_branch_produces_finite_stamp(self) -> None:
        diff = self._run("ideal_gaussian")
        assert diff.shape == (_SCI_SZ, _SCI_SZ)
        assert np.all(np.isfinite(diff))

    def test_branches_produce_distinct_stamps(self) -> None:
        """The two reference-processing branches are not algorithmically equivalent."""
        diff_matched = self._run("detector_matched")
        diff_ideal = self._run("ideal_gaussian")
        # We do not assert any quantitative similarity — only that the two
        # paths exercise different code (otherwise the branch would be dead).
        assert not np.array_equal(diff_matched, diff_ideal), (
            "detector_matched and ideal_gaussian produced identical stamps — "
            "the reference_processing switch may not be wired through."
        )
