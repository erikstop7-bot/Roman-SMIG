"""
smig/rendering/validation/test_release_gate.py
==============================================
End-to-end release-gate test running the science profile through SceneSimulator.

Every other Phase 2 integration test ([test_pipeline.py], [test_integration_phase2.py],
[test_dia_simulator_contract.py]) builds a reduced-PSF SimulationConfig
(oversample=2, n_wavelengths=2, ctx=64) so CI stays fast.  None of them exercise
the production knobs ``oversample=4``, ``n_wavelengths=10``, ``fov_native_pixels=256``
that simulation_science.yaml ships with — meaning a regression specific to those
values has no automated guard.

This test loads ``smig/config/simulation_science.yaml`` directly (model_copying
only ``n_reference_epochs`` down so the run completes in minutes instead of
tens of minutes) and asserts:

* Identity DIA preserves a clean null (|central_aperture_snr| < 3).
* Identity DIA recovers a 5× magnified source (|central_aperture_snr| > 3 on
  the peak epoch).
* Difference stamps are everywhere finite.
* Per-epoch provenance carries the science settings.

Marked ``slow`` so it runs only when explicitly selected.

Run from the project root::

    python -m pytest smig/rendering/validation/test_release_gate.py -m slow -v -s
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

galsim = pytest.importorskip("galsim")
pandas = pytest.importorskip("pandas")

from smig.config.utils import load_simulation_config
from smig.rendering.pipeline import SceneSimulator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCIENCE_YAML = _REPO_ROOT / "smig" / "config" / "simulation_science.yaml"

# Test downsizes purely for runtime: full Phase 2 ships n_reference_epochs=30
# but five reference epochs are sufficient to hit every PSF/detector code path.
_TEST_N_REF = 5
_TEST_N_SCIENCE = 3
_BASELINE_FLUX_E = 800.0          # ~21 AB mag at 46.8 s exposure
_SKY_BG_E_PER_S = 2.0             # ~22.5 AB mag/arcsec² zodiacal
_NULL_SNR_THRESHOLD = 3.0
_LENSED_PEAK_SNR_THRESHOLD = 3.0
_LENSED_PEAK_MAGNIFICATION = 5.0


def _central_aperture_snr(stamp: np.ndarray, radius: int = 3) -> float:
    """Compute aperture-photometry SNR on the central radius-N circle.

    Mirrors scripts/generate_phase2_qa_stamps.py::_central_residual_metrics so
    this gate uses the same metric the pilot QA does.
    """
    h, w = stamp.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    aperture = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    n_ap = int(aperture.sum())
    ap_sum = float(stamp[aperture].sum())

    r_in = radius + 3
    r_out = min(radius + 9, min(h, w) // 2 - 1)
    annulus = (
        ((yy - cy) ** 2 + (xx - cx) ** 2 <= r_out ** 2)
        & ~((yy - cy) ** 2 + (xx - cx) ** 2 <= r_in ** 2)
    )
    bg_per_pix = float(stamp[annulus].mean()) if annulus.any() else 0.0
    bg_std = float(stamp[annulus].std()) if annulus.sum() > 1 else float(stamp.std())
    excess = ap_sum - bg_per_pix * n_ap
    noise = max(bg_std * float(np.sqrt(max(n_ap, 1))), 1e-9)
    return excess / noise


@pytest.fixture(scope="module")
def science_config():
    """Load simulation_science.yaml and shrink n_reference_epochs for runtime."""
    cfg = load_simulation_config(_SCIENCE_YAML)
    return cfg.model_copy(
        update={"dia": cfg.dia.model_copy(update={"n_reference_epochs": _TEST_N_REF})}
    )


@pytest.fixture(scope="module")
def null_event_output(science_config):
    sim = SceneSimulator(science_config, master_seed=20260503)
    timestamps = np.array([60000.0 + i for i in range(_TEST_N_SCIENCE)])
    backgrounds = [_SKY_BG_E_PER_S] * _TEST_N_SCIENCE
    source_params = [{"flux_e": _BASELINE_FLUX_E} for _ in range(_TEST_N_SCIENCE)]
    return sim.simulate_event(
        event_id="release_gate_null",
        source_params_sequence=source_params,
        timestamps_mjd=timestamps,
        backgrounds_e_per_s=backgrounds,
        baseline_source_flux_e=_BASELINE_FLUX_E,
    )


@pytest.fixture(scope="module")
def lensed_event_output(science_config):
    sim = SceneSimulator(science_config, master_seed=20260503)
    timestamps = np.array([60000.0 + i for i in range(_TEST_N_SCIENCE)])
    backgrounds = [_SKY_BG_E_PER_S] * _TEST_N_SCIENCE
    # Gaussian magnification bell centered on the middle epoch.
    center = (_TEST_N_SCIENCE - 1) / 2.0
    sigma = max(center / 2.0, 0.5)
    mags = [
        1.0 + (_LENSED_PEAK_MAGNIFICATION - 1.0)
        * float(np.exp(-0.5 * ((i - center) / sigma) ** 2))
        for i in range(_TEST_N_SCIENCE)
    ]
    source_params = [{"flux_e": float(a) * _BASELINE_FLUX_E} for a in mags]
    return sim.simulate_event(
        event_id="release_gate_lensed",
        source_params_sequence=source_params,
        timestamps_mjd=timestamps,
        backgrounds_e_per_s=backgrounds,
        baseline_source_flux_e=_BASELINE_FLUX_E,
    )


@pytest.mark.slow
class TestScienceProfileEndToEnd:
    """End-to-end gate: simulation_science.yaml must produce trustworthy stamps."""

    def test_difference_stamps_are_finite_null(self, null_event_output) -> None:
        assert np.all(np.isfinite(null_event_output.difference_stamps))

    def test_difference_stamps_are_finite_lensed(self, lensed_event_output) -> None:
        assert np.all(np.isfinite(lensed_event_output.difference_stamps))

    def test_null_central_residual_is_clean(self, null_event_output) -> None:
        """A=1 event must not exceed the pilot QA null-central-SNR threshold."""
        for ep_idx, diff in enumerate(null_event_output.difference_stamps):
            snr = _central_aperture_snr(diff)
            assert abs(snr) < _NULL_SNR_THRESHOLD, (
                f"Null epoch {ep_idx}: |central_aperture_snr|={abs(snr):.3f} "
                f">= {_NULL_SNR_THRESHOLD}.  Identity DIA may be introducing a "
                "bias on the science profile (oversample=4, n_wavelengths=10, "
                "fov_native_pixels=256)."
            )

    def test_lensed_peak_central_signal_is_strong(self, lensed_event_output) -> None:
        """Magnification-peak epoch must produce a high-SNR central residual."""
        snrs = [
            _central_aperture_snr(d)
            for d in lensed_event_output.difference_stamps
        ]
        peak_snr = max(abs(s) for s in snrs)
        assert peak_snr > _LENSED_PEAK_SNR_THRESHOLD, (
            f"Lensed peak |central_aperture_snr|={peak_snr:.3f} <= "
            f"{_LENSED_PEAK_SNR_THRESHOLD}.  Identity DIA may be suppressing "
            "the source-core signal under the science PSF."
        )

    def test_provenance_records_carry_science_settings(self, null_event_output) -> None:
        """B2 fields must be populated under the science profile."""
        for rec in null_event_output.provenance:
            assert rec.dia_method == "identity"
            assert rec.reference_processing == "detector_matched"
            assert rec.reference_jitter_mode == "per_epoch"
            assert rec.psf_config_hash is not None and len(rec.psf_config_hash) == 64
