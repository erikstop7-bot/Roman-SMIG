"""
scripts/test_compare_science_profiles.py

Lightweight tests for compare_science_profiles.py.

Run from the project root:
    python -m pytest scripts/test_compare_science_profiles.py -v

Tests cover:
  - Candidate config loads and has oversample=2 with all other science
    settings identical to the baseline.
  - Production simulation_science.yaml is unmodified (oversample=4).
  - The comparison script imports cleanly (no side effects on import).
  - The argument parser exposes all required CLI flags.
  - Pure-function helpers (_null_metrics, _lensed_metrics, etc.) are
    correct on synthetic data.
  - Pass/fail verdict logic rejects obvious regressions.
  - Comparison CSV rows contain the required schema fields.
  - No production Phase 3 config references the experimental candidate.
  - PSF / EE helpers satisfy mathematical invariants (monotone EE, correct
    block-sum shape).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module loader helpers
# ---------------------------------------------------------------------------

def _load_compare_module():
    """Load compare_science_profiles without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "compare_science_profiles",
        Path(__file__).parent / "compare_science_profiles.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_simulation_config(path: Path):
    """Load a SimulationConfig from YAML; skips test if smig is not importable."""
    try:
        from smig.config.utils import load_simulation_config
        return load_simulation_config(path)
    except ImportError:
        pytest.skip("smig not installed")


# ---------------------------------------------------------------------------
# Config guard: candidate is oversample=2, identical otherwise
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_SCIENCE_CFG = _REPO_ROOT / "smig" / "config" / "simulation_science.yaml"
_CANDIDATE_CFG = _REPO_ROOT / "smig" / "config" / "simulation_science_oversample2.yaml"


def test_candidate_config_exists() -> None:
    """The oversample=2 candidate config must be present in the repo."""
    assert _CANDIDATE_CFG.exists(), (
        f"Missing experimental config: {_CANDIDATE_CFG}"
    )


def test_candidate_oversample_is_2() -> None:
    """Candidate config must declare psf.oversample = 2."""
    cfg = _load_simulation_config(_CANDIDATE_CFG)
    assert cfg.psf.oversample == 2, (
        f"Expected oversample=2, got {cfg.psf.oversample}"
    )


def test_baseline_oversample_is_4_unchanged() -> None:
    """Production simulation_science.yaml must retain oversample=4."""
    cfg = _load_simulation_config(_SCIENCE_CFG)
    assert cfg.psf.oversample == 4, (
        f"Production config oversample changed! Expected 4, got {cfg.psf.oversample}"
    )


def test_candidate_science_settings_match_baseline() -> None:
    """All DIA and detector science settings must be identical in candidate."""
    baseline = _load_simulation_config(_SCIENCE_CFG)
    candidate = _load_simulation_config(_CANDIDATE_CFG)

    # DIA settings must be identical
    assert candidate.dia.subtraction_method == baseline.dia.subtraction_method
    assert candidate.dia.reference_processing == baseline.dia.reference_processing
    assert candidate.dia.reference_jitter_mode == baseline.dia.reference_jitter_mode
    assert candidate.dia.n_reference_epochs == baseline.dia.n_reference_epochs
    assert candidate.dia.context_stamp_size == baseline.dia.context_stamp_size
    assert candidate.dia.science_stamp_size == baseline.dia.science_stamp_size

    # PSF settings that must be identical
    assert candidate.psf.filter_name == baseline.psf.filter_name
    assert candidate.psf.n_wavelengths == baseline.psf.n_wavelengths
    assert candidate.psf.fov_native_pixels == baseline.psf.fov_native_pixels
    assert candidate.psf.jitter_rms_mas == baseline.psf.jitter_rms_mas
    assert candidate.psf.wavelength_range_um == baseline.psf.wavelength_range_um
    assert candidate.psf.psf_edge_taper_pixels == baseline.psf.psf_edge_taper_pixels


def test_candidate_reference_processing_is_detector_matched() -> None:
    """Candidate must keep reference_processing=detector_matched (science-safe path)."""
    cfg = _load_simulation_config(_CANDIDATE_CFG)
    assert cfg.dia.reference_processing == "detector_matched"


def test_candidate_is_not_fast_dev() -> None:
    """Candidate must not reduce n_wavelengths or n_reference_epochs like fast_dev."""
    baseline = _load_simulation_config(_SCIENCE_CFG)
    candidate = _load_simulation_config(_CANDIDATE_CFG)
    assert candidate.psf.n_wavelengths == baseline.psf.n_wavelengths
    assert candidate.dia.n_reference_epochs == baseline.dia.n_reference_epochs


# ---------------------------------------------------------------------------
# Script import smoke test
# ---------------------------------------------------------------------------

def test_script_imports_cleanly() -> None:
    """compare_science_profiles.py must be importable without executing main()."""
    mod = _load_compare_module()
    for attr in ("main", "_build_parser", "_null_metrics", "_lensed_metrics",
                 "_pass_fail_verdict", "_make_comparison_rows",
                 "_compute_psf_comparison", "_compute_stamp_differences"):
        assert hasattr(mod, attr), f"Missing attribute: {attr}"


def test_no_pipeline_internals_imported_on_load() -> None:
    """Loading the script must not pull in sensor physics leaf modules."""
    forbidden = [
        "smig.sensor.ipc",
        "smig.sensor.charge_diffusion",
        "smig.sensor.persistence",
        "smig.sensor.nonlinearity",
        "smig.rendering.dia",
        "smig.rendering.crowding",
    ]
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(
        "_cmp_fresh", Path(__file__).parent / "compare_science_profiles.py"
    )
    fresh = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(fresh)  # type: ignore[union-attr]
    newly = set(sys.modules) - before
    for mod in forbidden:
        assert mod not in newly, f"Script imported forbidden module {mod!r}"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def test_parser_has_required_args() -> None:
    """Parser must expose all required comparison CLI flags."""
    mod = _load_compare_module()
    args = mod._build_parser().parse_args([])
    assert hasattr(args, "baseline_config")
    assert hasattr(args, "candidate_config")
    assert hasattr(args, "output_dir")
    assert hasattr(args, "pilot_batch_size")
    assert hasattr(args, "n_epochs")
    assert hasattr(args, "master_seed")
    assert hasattr(args, "aperture_radius")
    assert hasattr(args, "random_apertures")
    assert hasattr(args, "skip_baseline")
    assert hasattr(args, "skip_candidate")
    assert hasattr(args, "skip_psf_comparison")


def test_parser_defaults() -> None:
    mod = _load_compare_module()
    args = mod._build_parser().parse_args([])
    assert args.pilot_batch_size == mod._DEFAULT_PILOT_BATCH_SIZE
    assert args.n_epochs == mod._DEFAULT_N_EPOCHS
    assert args.master_seed == mod._DEFAULT_MASTER_SEED
    assert args.skip_baseline is False
    assert args.skip_candidate is False
    assert args.skip_psf_comparison is False


def test_parser_explicit_values() -> None:
    mod = _load_compare_module()
    args = mod._build_parser().parse_args([
        "--baseline-config", "a.yaml",
        "--candidate-config", "b.yaml",
        "--pilot-batch-size", "10",
        "--n-epochs", "5",
        "--master-seed", "999",
        "--skip-baseline",
        "--skip-psf-comparison",
    ])
    assert args.baseline_config == "a.yaml"
    assert args.candidate_config == "b.yaml"
    assert args.pilot_batch_size == 10
    assert args.n_epochs == 5
    assert args.master_seed == 999
    assert args.skip_baseline is True
    assert args.skip_psf_comparison is True


# ---------------------------------------------------------------------------
# Metrics aggregation helpers
# ---------------------------------------------------------------------------

def _make_null_rows(n: int = 6, snr: float = 1.0) -> list[dict]:
    """Synthetic null-event rows for testing."""
    return [
        {
            "event_type": "null",
            "event_id": f"pilot_{i * 2:04d}",
            "epoch": ep,
            "central_aperture_snr": snr * (1 if ep % 2 == 0 else -0.5),
            "central_aperture_percentile": 55.0,
            "rim_interior_std_ratio": 1.05,
            "has_nonfinite": False,
            "saturation_pixel_fraction": 0.001,
            "cr_pixel_fraction": 0.002,
            "qa_flags": [],
        }
        for i in range(n) for ep in range(3)
    ]


def _make_lensed_rows(n: int = 6, snr: float = 20.0) -> list[dict]:
    """Synthetic lensed-event rows for testing."""
    return [
        {
            "event_type": "lensed",
            "event_id": f"pilot_{i * 2 + 1:04d}",
            "epoch": ep,
            "central_aperture_snr": snr * (1.0 if ep == 1 else 0.3),
            "central_aperture_percentile": 99.5,
            "rim_interior_std_ratio": 1.1,
            "has_nonfinite": False,
            "saturation_pixel_fraction": 0.001,
            "cr_pixel_fraction": 0.002,
            "central_signal_dominant": True,
            "qa_flags": [],
        }
        for i in range(n) for ep in range(3)
    ]


def test_null_metrics_returns_expected_keys() -> None:
    mod = _load_compare_module()
    rows = _make_null_rows()
    m = mod._null_metrics(rows)
    for key in [
        "n_rows", "central_snr_mean", "central_snr_max_abs",
        "central_pct_mean", "rim_ratio_mean",
        "nonfinite_frac", "saturation_frac_mean", "flagged_frac",
    ]:
        assert key in m, f"Missing key: {key}"


def test_null_metrics_empty_rows() -> None:
    mod = _load_compare_module()
    m = mod._null_metrics([])
    assert m["n_rows"] == 0


def test_null_metrics_filters_by_event_type() -> None:
    mod = _load_compare_module()
    rows = _make_null_rows(3) + _make_lensed_rows(3)
    m = mod._null_metrics(rows)
    assert m["n_rows"] == 9  # 3 events × 3 epochs, null only


def test_null_metrics_max_abs_snr() -> None:
    mod = _load_compare_module()
    rows = _make_null_rows(n=2, snr=3.0)
    m = mod._null_metrics(rows)
    assert m["central_snr_max_abs"] is not None
    assert m["central_snr_max_abs"] >= 3.0


def test_lensed_metrics_returns_expected_keys() -> None:
    mod = _load_compare_module()
    rows = _make_lensed_rows()
    m = mod._lensed_metrics(rows)
    for key in [
        "n_rows", "central_snr_mean", "central_snr_max",
        "rim_ratio_mean", "central_signal_dominant_frac",
        "nonfinite_frac", "flagged_frac",
    ]:
        assert key in m, f"Missing key: {key}"


def test_lensed_metrics_empty_rows() -> None:
    mod = _load_compare_module()
    m = mod._lensed_metrics([])
    assert m["n_rows"] == 0


# ---------------------------------------------------------------------------
# Pass/fail verdict
# ---------------------------------------------------------------------------

def _clean_verdict(mod) -> dict:
    """Verdict from clean synthetic data — should PASS."""
    b_null = mod._null_metrics(_make_null_rows(snr=1.0))
    c_null = mod._null_metrics(_make_null_rows(snr=1.2))
    b_lensed = mod._lensed_metrics(_make_lensed_rows(snr=20.0))
    c_lensed = mod._lensed_metrics(_make_lensed_rows(snr=19.5))
    stamp_diff = {
        "lensed_central_snr_ratio_mean": 0.98,   # within 2% — well inside tolerance
        "null_stamp_rms_diff_mean": 1.5,
        "lensed_stamp_rms_diff_mean": 5.0,
        "max_abs_pixel_diff": 20.0,
    }
    return mod._pass_fail_verdict(b_null, c_null, b_lensed, c_lensed, stamp_diff, None)


def test_pass_fail_verdict_passes_clean_data() -> None:
    mod = _load_compare_module()
    result = _clean_verdict(mod)
    assert result["verdict"] in ("PASS", "WARN"), (
        f"Expected PASS/WARN on clean data, got {result['verdict']}: "
        f"{result['fail_reasons']}"
    )


def test_pass_fail_verdict_fails_on_high_null_snr() -> None:
    """Candidate null SNR above threshold must trigger FAIL."""
    mod = _load_compare_module()
    b_null = mod._null_metrics(_make_null_rows(snr=1.0))
    # Inject a row with very high SNR so max_abs exceeds threshold
    high_snr_rows = _make_null_rows(snr=1.0)
    high_snr_rows[0]["central_aperture_snr"] = mod._NULL_SNR_MAX_THRESHOLD + 1.0
    c_null = mod._null_metrics(high_snr_rows)
    b_lensed = mod._lensed_metrics(_make_lensed_rows())
    c_lensed = mod._lensed_metrics(_make_lensed_rows())
    stamp_diff = {"lensed_central_snr_ratio_mean": 0.99}
    result = mod._pass_fail_verdict(b_null, c_null, b_lensed, c_lensed, stamp_diff, None)
    assert result["verdict"] == "FAIL"
    assert any("SNR" in r for r in result["fail_reasons"])


def test_pass_fail_verdict_fails_on_nonfinite() -> None:
    """Nonfinite pixels in candidate stamps must trigger FAIL."""
    mod = _load_compare_module()
    b_null = mod._null_metrics(_make_null_rows())
    nf_rows = _make_null_rows()
    for r in nf_rows:
        r["has_nonfinite"] = True
    c_null = mod._null_metrics(nf_rows)
    b_lensed = mod._lensed_metrics(_make_lensed_rows())
    c_lensed = mod._lensed_metrics(_make_lensed_rows())
    stamp_diff = {"lensed_central_snr_ratio_mean": 0.99}
    result = mod._pass_fail_verdict(b_null, c_null, b_lensed, c_lensed, stamp_diff, None)
    assert result["verdict"] == "FAIL"
    assert any("nonfinite" in r.lower() for r in result["fail_reasons"])


def test_pass_fail_verdict_fails_on_bad_lensed_ratio() -> None:
    """Lensed flux ratio deviation > 10% must trigger FAIL."""
    mod = _load_compare_module()
    b_null = mod._null_metrics(_make_null_rows())
    c_null = mod._null_metrics(_make_null_rows())
    b_lensed = mod._lensed_metrics(_make_lensed_rows())
    c_lensed = mod._lensed_metrics(_make_lensed_rows())
    stamp_diff = {"lensed_central_snr_ratio_mean": 0.80}   # 20% deviation
    result = mod._pass_fail_verdict(b_null, c_null, b_lensed, c_lensed, stamp_diff, None)
    assert result["verdict"] == "FAIL"
    assert any("flux ratio" in r.lower() for r in result["fail_reasons"])


def test_pass_fail_verdict_fails_on_bad_rim_ratio() -> None:
    """Rim/interior ratio outside [0.5, 2.0] must trigger FAIL."""
    mod = _load_compare_module()
    b_null = mod._null_metrics(_make_null_rows())
    bad_rim_rows = _make_null_rows()
    for r in bad_rim_rows:
        r["rim_interior_std_ratio"] = 3.0   # well above _RIM_RATIO_HIGH
    c_null = mod._null_metrics(bad_rim_rows)
    b_lensed = mod._lensed_metrics(_make_lensed_rows())
    c_lensed = mod._lensed_metrics(_make_lensed_rows())
    stamp_diff = {"lensed_central_snr_ratio_mean": 0.99}
    result = mod._pass_fail_verdict(b_null, c_null, b_lensed, c_lensed, stamp_diff, None)
    assert result["verdict"] == "FAIL"
    assert any("rim" in r.lower() for r in result["fail_reasons"])


def test_verdict_dict_has_required_keys() -> None:
    mod = _load_compare_module()
    result = _clean_verdict(mod)
    for key in ("verdict", "fail_reasons", "warn_reasons", "pass_reasons"):
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["fail_reasons"], list)
    assert isinstance(result["warn_reasons"], list)
    assert isinstance(result["pass_reasons"], list)


# ---------------------------------------------------------------------------
# Comparison CSV row schema
# ---------------------------------------------------------------------------

def test_comparison_rows_have_required_schema_fields() -> None:
    """Every comparison row must contain the required CSV fieldnames."""
    mod = _load_compare_module()
    b_null = mod._null_metrics(_make_null_rows())
    c_null = mod._null_metrics(_make_null_rows())
    b_lensed = mod._lensed_metrics(_make_lensed_rows())
    c_lensed = mod._lensed_metrics(_make_lensed_rows())
    stamp_diff: dict = {
        "null_stamp_rms_diff_mean": 1.0,
        "lensed_stamp_rms_diff_mean": 5.0,
        "max_abs_pixel_diff": 20.0,
        "lensed_central_snr_ratio_mean": 0.98,
        "lensed_central_snr_ratio_std": 0.01,
    }
    rows = mod._make_comparison_rows(
        b_null=b_null, c_null=c_null,
        b_lensed=b_lensed, c_lensed=c_lensed,
        stamp_diff=stamp_diff, psf_cmp=None,
        b_elapsed=120.0, c_elapsed=35.0,
    )
    assert len(rows) > 0
    required = set(mod._COMPARISON_CSV_FIELDNAMES)
    for row in rows:
        missing = required - set(row.keys())
        assert not missing, f"Row '{row.get('metric')}' missing fields: {missing}"


def test_comparison_rows_json_serialisable() -> None:
    """Comparison rows must round-trip through JSON without error."""
    import json as _json
    mod = _load_compare_module()
    b_null = mod._null_metrics(_make_null_rows())
    c_null = mod._null_metrics(_make_null_rows())
    b_lensed = mod._lensed_metrics(_make_lensed_rows())
    c_lensed = mod._lensed_metrics(_make_lensed_rows())
    stamp_diff: dict = {"lensed_central_snr_ratio_mean": 0.98}
    rows = mod._make_comparison_rows(
        b_null=b_null, c_null=c_null,
        b_lensed=b_lensed, c_lensed=c_lensed,
        stamp_diff=stamp_diff, psf_cmp=None,
        b_elapsed=100.0, c_elapsed=30.0,
    )
    serialised = _json.dumps(rows)
    recovered = _json.loads(serialised)
    assert len(recovered) == len(rows)


def test_comparison_rows_speedup_reported() -> None:
    """speedup_factor row must appear with ratio_or_diff set."""
    mod = _load_compare_module()
    rows = mod._make_comparison_rows(
        b_null=mod._null_metrics(_make_null_rows()),
        c_null=mod._null_metrics(_make_null_rows()),
        b_lensed=mod._lensed_metrics(_make_lensed_rows()),
        c_lensed=mod._lensed_metrics(_make_lensed_rows()),
        stamp_diff={},
        psf_cmp=None,
        b_elapsed=120.0,
        c_elapsed=30.0,
    )
    speedup_rows = [r for r in rows if r["metric"] == "speedup_factor"]
    assert len(speedup_rows) == 1
    assert speedup_rows[0]["ratio_or_diff"] == pytest.approx(4.0, rel=1e-3)


# ---------------------------------------------------------------------------
# PSF / EE helpers
# ---------------------------------------------------------------------------

def test_gaussian_psf_on_grid_normalised() -> None:
    """Gaussian PSF must sum to 1."""
    mod = _load_compare_module()
    psf = mod._gaussian_psf_on_grid(64, sigma_pix=2.0)
    assert abs(float(psf.sum()) - 1.0) < 1e-12


def test_gaussian_psf_on_grid_peak_at_centre() -> None:
    mod = _load_compare_module()
    psf = mod._gaussian_psf_on_grid(64, sigma_pix=2.0)
    cy, cx = 32, 32
    assert psf[cy, cx] == psf.max()


def test_bin_to_native_shape() -> None:
    """Block-sum must produce the correct native-pixel shape."""
    mod = _load_compare_module()
    for oversample in (2, 4):
        n = 64 * oversample
        psf_os = np.ones((n, n), dtype=np.float64)
        native = mod._bin_to_native(psf_os, oversample)
        assert native.shape == (64, 64), (
            f"oversample={oversample}: expected (64,64), got {native.shape}"
        )


def test_bin_to_native_conserves_flux() -> None:
    """Block-sum must conserve total flux exactly."""
    mod = _load_compare_module()
    rng = np.random.default_rng(42)
    for oversample in (2, 4):
        n = 32 * oversample
        psf_os = rng.random((n, n))
        native = mod._bin_to_native(psf_os, oversample)
        assert abs(float(psf_os.sum()) - float(native.sum())) < 1e-10


def test_encircled_energy_monotone() -> None:
    """EE curve must be monotonically non-decreasing."""
    mod = _load_compare_module()
    psf = mod._gaussian_psf_on_grid(64, sigma_pix=2.0)
    _, ee = mod._encircled_energy(psf, max_radius_px=20)
    diffs = np.diff(ee)
    assert np.all(diffs >= -1e-12), "EE curve is not monotone non-decreasing"


def test_encircled_energy_bounded() -> None:
    """EE values must lie in [0, 1]."""
    mod = _load_compare_module()
    psf = mod._gaussian_psf_on_grid(64, sigma_pix=2.0)
    _, ee = mod._encircled_energy(psf, max_radius_px=15)
    assert np.all(ee >= 0.0)
    assert np.all(ee <= 1.0 + 1e-12)


def test_second_moment_positive() -> None:
    """Second moment must be a positive finite float for any non-trivial PSF."""
    mod = _load_compare_module()
    psf = mod._gaussian_psf_on_grid(64, sigma_pix=3.0)
    sigma = mod._second_moment(psf)
    assert np.isfinite(sigma)
    assert sigma > 0.0


def test_second_moment_zero_psf() -> None:
    """Second moment of an all-zero PSF must return NaN (not crash)."""
    mod = _load_compare_module()
    result = mod._second_moment(np.zeros((32, 32)))
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# PSF comparison dict schema
# ---------------------------------------------------------------------------

def test_psf_comparison_dict_schema() -> None:
    """_compute_psf_comparison must return all required keys."""
    mod = _load_compare_module()
    # Create minimal fake config objects with the required PSF fields
    class _FakePSF:
        oversample = 4
        fov_native_pixels = 64
        wavelength_range_um = (0.93, 2.00)
        jitter_rms_mas = 5.0
        filter_name = "W146"
        n_wavelengths = 10
        psf_edge_taper_pixels = None

    class _FakePSFCand:
        oversample = 2
        fov_native_pixels = 64
        wavelength_range_um = (0.93, 2.00)
        jitter_rms_mas = 5.0
        filter_name = "W146"
        n_wavelengths = 10
        psf_edge_taper_pixels = None

    class _FakeCfg:
        def __init__(self, psf):
            self.psf = psf

    b_cfg = _FakeCfg(_FakePSF())
    c_cfg = _FakeCfg(_FakePSFCand())

    result = mod._compute_psf_comparison(b_cfg, c_cfg, fov_native=64)

    required_keys = {
        "method", "baseline_oversample", "candidate_oversample",
        "peak_ratio_c_over_b", "width_ratio_c_over_b",
        "ee_diff_max", "rim_energy_ratio_c_over_b",
        "baseline_ee_at_radii", "candidate_ee_at_radii",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"PSF comparison dict missing keys: {missing}"


def test_psf_comparison_uses_analytic_fallback_without_galsim() -> None:
    """PSF comparison must succeed and report analytic_gaussian when GalSim is absent."""
    mod = _load_compare_module()

    class _FakePSF:
        oversample = 4
        fov_native_pixels = 32
        wavelength_range_um = (0.93, 2.00)
        jitter_rms_mas = 5.0
        filter_name = "W146"
        n_wavelengths = 10
        psf_edge_taper_pixels = None

    class _FakePSFCand:
        oversample = 2
        fov_native_pixels = 32
        wavelength_range_um = (0.93, 2.00)
        jitter_rms_mas = 5.0
        filter_name = "W146"
        n_wavelengths = 10
        psf_edge_taper_pixels = None

    class _FakeCfg:
        def __init__(self, psf):
            self.psf = psf

    import unittest.mock as mock
    # Force STPSFProvider to raise ImportError so the analytic fallback is used.
    with mock.patch.dict("sys.modules", {"smig.optics.psf": None}):
        result = mod._compute_psf_comparison(_FakeCfg(_FakePSF()), _FakeCfg(_FakePSFCand()), fov_native=32)
    assert result["method"] == "analytic_gaussian"


# ---------------------------------------------------------------------------
# Stamp difference helpers
# ---------------------------------------------------------------------------

def test_stamp_differences_missing_files_is_graceful(tmp_path: Path) -> None:
    """_compute_stamp_differences must handle missing .npy files without error."""
    mod = _load_compare_module()
    result = mod._compute_stamp_differences(
        baseline_dir=tmp_path / "baseline",
        candidate_dir=tmp_path / "candidate",
        batch_size=4,
        aperture_radius=3,
    )
    assert result["n_events_matched"] == 0


def test_stamp_differences_with_identical_stamps(tmp_path: Path) -> None:
    """Identical stamps must produce zero RMS difference and ratio ≈ 1."""
    mod = _load_compare_module()
    rng = np.random.default_rng(0)
    b_dir = tmp_path / "baseline"
    c_dir = tmp_path / "candidate"

    for i in range(4):
        event_id = f"pilot_{i:04d}"
        for d in (b_dir, c_dir):
            ev = d / event_id
            ev.mkdir(parents=True)
            stamps = rng.standard_normal((3, 64, 64)).astype(np.float32)
            # Lensed events (odd i) get a bright centre
            if i % 2 == 1:
                stamps[:, 32, 32] += 100.0
            np.save(str(ev / "diff_stamps.npy"), stamps)

        # Make candidate identical to baseline
        b_stamps = np.load(str(b_dir / event_id / "diff_stamps.npy"))
        np.save(str(c_dir / event_id / "diff_stamps.npy"), b_stamps)

    result = mod._compute_stamp_differences(
        baseline_dir=b_dir,
        candidate_dir=c_dir,
        batch_size=4,
        aperture_radius=3,
    )
    assert result["n_events_matched"] == 4
    assert result["null_stamp_rms_diff_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["lensed_stamp_rms_diff_mean"] == pytest.approx(0.0, abs=1e-12)
    if result["lensed_central_snr_ratio_mean"] is not None:
        assert result["lensed_central_snr_ratio_mean"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Production Phase 3 config isolation guard
# ---------------------------------------------------------------------------

def test_no_production_config_references_oversample2() -> None:
    """No Phase 3 / production config must reference the experimental YAML."""
    configs_dir = _REPO_ROOT / "configs"
    if not configs_dir.exists():
        pytest.skip("configs/ directory not found")

    candidate_name = "simulation_science_oversample2.yaml"
    for config_file in configs_dir.rglob("*.yaml"):
        text = config_file.read_text(encoding="utf-8")
        assert candidate_name not in text, (
            f"Production config {config_file} references experimental candidate "
            f"{candidate_name!r}. Remove the reference before merging."
        )


def test_candidate_config_not_in_qa_script_defaults() -> None:
    """The candidate YAML must not appear in generate_phase2_qa_stamps default configs."""
    qa_path = Path(__file__).parent / "generate_phase2_qa_stamps.py"
    if not qa_path.exists():
        pytest.skip("generate_phase2_qa_stamps.py not found")
    text = qa_path.read_text(encoding="utf-8")
    assert "simulation_science_oversample2" not in text, (
        "generate_phase2_qa_stamps.py has a reference to the experimental candidate config. "
        "The candidate is non-default; it must be passed via --config only."
    )


def test_simulation_science_yaml_unchanged_oversample() -> None:
    """Canary: raw YAML content of simulation_science.yaml must declare oversample: 4."""
    text = _SCIENCE_CFG.read_text(encoding="utf-8")
    assert "oversample: 4" in text, (
        "simulation_science.yaml no longer declares oversample: 4 — "
        "it may have been accidentally modified."
    )


def test_candidate_yaml_declares_oversample_2() -> None:
    """Canary: raw YAML content of candidate config must declare oversample: 2."""
    text = _CANDIDATE_CFG.read_text(encoding="utf-8")
    assert "oversample: 2" in text, (
        "simulation_science_oversample2.yaml does not declare oversample: 2."
    )
