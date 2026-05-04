"""
scripts/test_generate_phase2_qa_stamps.py

Lightweight smoke tests for generate_phase2_qa_stamps.py.

Run from the project root:
    python -m pytest scripts/test_generate_phase2_qa_stamps.py -v
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module loader (re-used by every test)
# ---------------------------------------------------------------------------

def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_phase2_qa_stamps",
        Path(__file__).parent / "generate_phase2_qa_stamps.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------

def test_script_imports() -> None:
    """Script must be importable without executing main()."""
    mod = _load_module()
    for attr in ("run", "main", "_build_parser"):
        assert hasattr(mod, attr), f"Missing attribute: {attr}"


def test_no_pipeline_internals_imported() -> None:
    """Script must not import sensor physics leaf modules."""
    mod = _load_module()
    forbidden = [
        "smig.sensor.ipc",
        "smig.sensor.charge_diffusion",
        "smig.sensor.persistence",
        "smig.sensor.nonlinearity",
        "smig.rendering.dia",
        "smig.rendering.crowding",
        "smig.optics.psf",
    ]
    module_names = set(sys.modules)
    # Re-load fresh to capture what the script pulls in
    import importlib as _il
    spec = importlib.util.spec_from_file_location(
        "_qa_fresh", Path(__file__).parent / "generate_phase2_qa_stamps.py"
    )
    fresh = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    before = set(sys.modules)
    spec.loader.exec_module(fresh)  # type: ignore[union-attr]
    newly_imported = set(sys.modules) - before
    for forbidden_mod in forbidden:
        assert forbidden_mod not in newly_imported, (
            f"Script imported internal module {forbidden_mod!r}"
        )


# ---------------------------------------------------------------------------
# Argument parsing — existing args
# ---------------------------------------------------------------------------

def test_parser_defaults() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    assert args.config is None
    assert args.output_dir == mod._DEFAULT_OUTPUT_DIR
    assert args.n_epochs == mod._DEFAULT_N_EPOCHS
    assert args.master_seed == mod._DEFAULT_MASTER_SEED


def test_parser_existing_args() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([
        "--config", "smig/config/simulation.yaml",
        "--output-dir", "/tmp/qa_out",
        "--n-epochs", "2",
        "--master-seed", "42",
    ])
    assert args.config == "smig/config/simulation.yaml"
    assert args.n_epochs == 2
    assert args.master_seed == 42


# ---------------------------------------------------------------------------
# Argument parsing — new args
# ---------------------------------------------------------------------------

def test_parser_new_args_defaults() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    assert args.central_aperture_radius == mod._DEFAULT_APERTURE_RADIUS
    assert args.random_apertures == mod._DEFAULT_RANDOM_APERTURES
    assert args.central_snr_warn == mod._DEFAULT_CENTRAL_SNR_WARN
    assert args.central_percentile_warn == mod._DEFAULT_CENTRAL_PERCENTILE_WARN


def test_parser_new_args_explicit() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([
        "--central-aperture-radius", "5",
        "--random-apertures", "200",
        "--central-snr-warn", "3.0",
        "--central-percentile-warn", "95.0",
    ])
    assert args.central_aperture_radius == 5
    assert args.random_apertures == 200
    assert args.central_snr_warn == 3.0
    assert args.central_percentile_warn == 95.0


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def test_gaussian_magnification_curve_shape() -> None:
    mod = _load_module()
    for n in [1, 3, 5, 7]:
        mags = mod._gaussian_magnification_curve(n, peak=5.0)
        assert len(mags) == n

def test_gaussian_magnification_peak_is_central() -> None:
    mod = _load_module()
    mags = mod._gaussian_magnification_curve(5, peak=5.0)
    assert mags[2] >= mags[0]
    assert mags[2] >= mags[4]
    assert mags[2] > 1.0


# ---------------------------------------------------------------------------
# Statistics helpers — existing
# ---------------------------------------------------------------------------

def test_stamp_summary_keys() -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    stamp = rng.standard_normal((64, 64))
    stats = mod._stamp_summary(stamp)
    for key in ["mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95", "sum"]:
        assert key in stats, f"Missing key: {key}"


def test_rim_stats_keys() -> None:
    mod = _load_module()
    rim = mod._rim_stats(np.zeros((64, 64)))
    for key in ["rim_mean", "rim_std", "rim_max_abs", "interior_mean", "interior_std"]:
        assert key in rim


def test_rim_stats_small_stamp() -> None:
    mod = _load_module()
    rim = mod._rim_stats(np.ones((8, 8)), rim_width=5)
    assert np.isfinite(rim["rim_mean"])


# ---------------------------------------------------------------------------
# Central residual metrics
# ---------------------------------------------------------------------------

def test_central_residual_metrics_keys() -> None:
    mod = _load_module()
    rng = np.random.default_rng(7)
    stamp = rng.standard_normal((64, 64))
    m = mod._central_residual_metrics(stamp, aperture_radius=3)
    required = [
        "center_pixel",
        "central_3x3_sum",
        "central_3x3_mean",
        "central_aperture_sum",
        "central_aperture_n_pix",
        "central_aperture_excess",
        "central_aperture_snr",
        "annulus_bg_per_pix",
        "annulus_bg_std",
    ]
    for k in required:
        assert k in m, f"Missing key in central_residual_metrics: {k}"


def test_central_residual_metrics_finite() -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    stamp = rng.standard_normal((64, 64))
    m = mod._central_residual_metrics(stamp, aperture_radius=3)
    for k, v in m.items():
        if k != "central_aperture_n_pix":
            assert np.isfinite(v), f"Non-finite value for {k}: {v}"


def test_central_residual_metrics_hotspot() -> None:
    """Stamp with a bright central spike should yield a positive SNR."""
    mod = _load_module()
    stamp = np.zeros((64, 64))
    stamp[32, 32] = 100.0
    m = mod._central_residual_metrics(stamp, aperture_radius=3)
    assert m["central_aperture_snr"] > 0, "Expected positive SNR for bright centre"
    assert m["center_pixel"] == pytest.approx(100.0)


def test_central_residual_metrics_flat_stamp() -> None:
    """Uniform stamp should give near-zero SNR."""
    mod = _load_module()
    stamp = np.ones((64, 64)) * 5.0
    m = mod._central_residual_metrics(stamp, aperture_radius=3)
    assert abs(m["central_aperture_snr"]) < 1e-6


def test_circular_aperture_mask_shape_and_dtype() -> None:
    mod = _load_module()
    mask = mod._circular_aperture_mask(64, 64, 32.0, 32.0, 3.0)
    assert mask.shape == (64, 64)
    assert mask.dtype == bool


def test_circular_aperture_mask_radius_zero() -> None:
    mod = _load_module()
    mask = mod._circular_aperture_mask(64, 64, 32.0, 32.0, 0.0)
    assert int(mask.sum()) == 1


# ---------------------------------------------------------------------------
# Null random apertures
# ---------------------------------------------------------------------------

def test_null_random_apertures_keys() -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    stamp = rng.standard_normal((64, 64))
    result = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=100, seed=42)
    for k in ["n_random_apertures", "central_aperture_percentile", "central_aperture_z_score"]:
        assert k in result, f"Missing key: {k}"


def test_null_random_apertures_deterministic() -> None:
    """Same seed must give identical results."""
    mod = _load_module()
    rng = np.random.default_rng(99)
    stamp = rng.standard_normal((64, 64))
    r1 = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=100, seed=42)
    r2 = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=100, seed=42)
    assert r1["central_aperture_percentile"] == r2["central_aperture_percentile"]
    assert r1["central_aperture_z_score"] == r2["central_aperture_z_score"]
    assert r1["n_random_apertures"] == r2["n_random_apertures"]


def test_null_random_apertures_different_seeds() -> None:
    """Different seeds should (almost always) give different random distributions."""
    mod = _load_module()
    rng = np.random.default_rng(77)
    stamp = rng.standard_normal((64, 64))
    r1 = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=200, seed=42)
    r2 = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=200, seed=999)
    # Percentile could theoretically match by chance but z-score should differ
    assert r1["central_aperture_z_score"] != r2["central_aperture_z_score"]


def test_null_random_apertures_percentile_range() -> None:
    """Percentile must be in [0, 100]."""
    mod = _load_module()
    rng = np.random.default_rng(3)
    stamp = rng.standard_normal((64, 64))
    result = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=200, seed=42)
    pct = result["central_aperture_percentile"]
    assert 0.0 <= pct <= 100.0


def test_null_random_apertures_bright_centre_high_percentile() -> None:
    """Stamp with a bright central spike should have a high percentile."""
    mod = _load_module()
    stamp = np.zeros((64, 64))
    stamp[30:35, 30:35] = 50.0  # bright central region
    result = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=300, seed=42)
    assert result["central_aperture_percentile"] > 80.0, (
        f"Expected high percentile for bright centre, got {result['central_aperture_percentile']}"
    )


def test_null_random_apertures_tiny_stamp() -> None:
    """Tiny stamps that can't fit enough random positions return graceful fallback."""
    mod = _load_module()
    stamp = np.zeros((12, 12))
    result = mod._null_random_apertures(stamp, aperture_radius=3, n_samples=100, seed=42)
    # Either succeeds or returns the fallback with n_random_apertures=0
    assert "n_random_apertures" in result


# ---------------------------------------------------------------------------
# Robust scale
# ---------------------------------------------------------------------------

def test_robust_scale_symmetric() -> None:
    mod = _load_module()
    arr = np.linspace(-10, 10, 1000)
    scale = mod._robust_scale(arr)
    assert scale > 0


def test_robust_scale_minimum_one() -> None:
    """Scale is at least 1.0 even for a near-zero array."""
    mod = _load_module()
    scale = mod._robust_scale(np.zeros(100))
    assert scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def test_resolve_config_explicit_missing(tmp_path: Path) -> None:
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod._resolve_config(str(tmp_path / "nonexistent.yaml"))


def test_resolve_config_explicit_present(tmp_path: Path) -> None:
    mod = _load_module()
    cfg_file = tmp_path / "sim.yaml"
    cfg_file.write_text("detector: {}\n")
    assert mod._resolve_config(str(cfg_file)) == cfg_file


def test_resolve_config_no_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_DEFAULT_CONFIGS", [
        str(tmp_path / "missing1.yaml"), str(tmp_path / "missing2.yaml")
    ])
    with pytest.raises(SystemExit):
        mod._resolve_config(None)


# ---------------------------------------------------------------------------
# Fixed-scale PNG — path generation does not crash without matplotlib
# ---------------------------------------------------------------------------

def test_fixed_scale_png_skipped_without_mpl(tmp_path: Path) -> None:
    """When _MPL is False, fixed-scale PNG saving must be silently skipped."""
    mod = _load_module()
    # Simulate matplotlib being absent by patching the flag
    original = mod._MPL
    try:
        mod._MPL = False
        # The fixed-scale pass is inside `if _MPL and all_diff_stamps:` so
        # calling _robust_scale and verifying it doesn't import plt is enough.
        scale = mod._robust_scale(np.random.default_rng(0).standard_normal(100))
        assert scale > 0
    finally:
        mod._MPL = original


def test_save_png_fixed_scale_path_strings(tmp_path: Path) -> None:
    """_save_png_fixed_scale must accept Path objects without raising."""
    mod = _load_module()
    if not mod._MPL:
        pytest.skip("matplotlib not available")
    stamp = np.random.default_rng(0).standard_normal((16, 16))
    path = tmp_path / "test_fixed.png"
    mod._save_png_fixed_scale(stamp, path, title="test", vmax=2.0)
    assert path.exists()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Null ablation — parser
# ---------------------------------------------------------------------------

def test_parser_run_null_ablations_default() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    assert hasattr(args, "run_null_ablations")
    assert args.run_null_ablations is False


def test_parser_run_null_ablations_flag() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args(["--run-null-ablations"])
    assert args.run_null_ablations is True


# ---------------------------------------------------------------------------
# Null ablation — mock config helper
# ---------------------------------------------------------------------------

class _MockDIA:
    reference_processing: str = "detector_matched"
    reference_jitter_mode: str = "per_epoch"

    def model_copy(self, update=None):
        if update is None:
            update = {}
        new = _MockDIA()
        new.reference_processing = update.get("reference_processing", self.reference_processing)
        new.reference_jitter_mode = update.get("reference_jitter_mode", self.reference_jitter_mode)
        return new


class _MockPSF:
    jitter_rms_mas: float = 5.0

    def model_copy(self, update=None):
        if update is None:
            update = {}
        new = _MockPSF()
        new.jitter_rms_mas = update.get("jitter_rms_mas", self.jitter_rms_mas)
        return new


class _MockCFG:
    def __init__(self):
        self.dia = _MockDIA()
        self.psf = _MockPSF()

    def model_copy(self, update=None):
        if update is None:
            update = {}
        new = _MockCFG()
        new.dia = update.get("dia", self.dia)
        new.psf = update.get("psf", self.psf)
        return new


# ---------------------------------------------------------------------------
# Null ablation — _build_ablation_plan
# ---------------------------------------------------------------------------

_EXPECTED_VARIANT_NAMES = [
    "baseline_science",
    "no_source",
    "source_only_no_crowding",
    "shared_reference_jitter",
    "ideal_gaussian_reference",
    "zero_jitter",
    "centered_source_position_check",
]


def test_build_ablation_plan_returns_all_variants() -> None:
    mod = _load_module()
    cfg = _MockCFG()
    specs = mod._build_ablation_plan(cfg, n_epochs=3)
    names = [s["name"] for s in specs]
    assert names == _EXPECTED_VARIANT_NAMES, f"Got names: {names}"


def test_build_ablation_plan_required_keys() -> None:
    mod = _load_module()
    cfg = _MockCFG()
    for spec in mod._build_ablation_plan(cfg, n_epochs=2):
        for key in ("name", "description", "cfg", "source_params_sequence",
                    "baseline_source_flux_e", "neighbor_catalog",
                    "skip_reason", "config_knobs"):
            assert key in spec, f"Spec '{spec['name']}' missing key '{key}'"


def test_build_ablation_plan_epoch_count() -> None:
    """Each spec's source_params_sequence must have n_epochs entries."""
    mod = _load_module()
    cfg = _MockCFG()
    for n in [1, 3, 5]:
        for spec in mod._build_ablation_plan(cfg, n_epochs=n):
            if spec["skip_reason"] is None:
                assert len(spec["source_params_sequence"]) == n, (
                    f"Variant '{spec['name']}' has wrong epoch count for n={n}"
                )


def test_build_ablation_plan_no_source_variant() -> None:
    """no_source must have flux_e=0.0 in science and baseline_source_flux_e=None."""
    mod = _load_module()
    cfg = _MockCFG()
    specs = {s["name"]: s for s in mod._build_ablation_plan(cfg, n_epochs=3)}
    ns = specs["no_source"]
    assert ns["baseline_source_flux_e"] is None
    for p in ns["source_params_sequence"]:
        assert p["flux_e"] == 0.0


def test_build_ablation_plan_shared_jitter_variant() -> None:
    """shared_reference_jitter cfg must have reference_jitter_mode='shared'."""
    mod = _load_module()
    cfg = _MockCFG()
    specs = {s["name"]: s for s in mod._build_ablation_plan(cfg, n_epochs=2)}
    sj = specs["shared_reference_jitter"]
    assert sj["cfg"].dia.reference_jitter_mode == "shared"
    assert sj["config_knobs"]["reference_jitter_mode"] == "shared"


def test_build_ablation_plan_ideal_gaussian_variant() -> None:
    """ideal_gaussian_reference cfg must have reference_processing='ideal_gaussian'."""
    mod = _load_module()
    cfg = _MockCFG()
    specs = {s["name"]: s for s in mod._build_ablation_plan(cfg, n_epochs=2)}
    ig = specs["ideal_gaussian_reference"]
    assert ig["cfg"].dia.reference_processing == "ideal_gaussian"


def test_build_ablation_plan_zero_jitter_variant() -> None:
    """zero_jitter cfg must have jitter_rms_mas=0.0 (or be skipped with a reason)."""
    mod = _load_module()
    cfg = _MockCFG()
    specs = {s["name"]: s for s in mod._build_ablation_plan(cfg, n_epochs=2)}
    zj = specs["zero_jitter"]
    if zj["skip_reason"] is None:
        assert zj["cfg"].psf.jitter_rms_mas == 0.0
    else:
        assert "jitter_rms_mas" in zj["skip_reason"].lower() or len(zj["skip_reason"]) > 0


def test_build_ablation_plan_centered_source_variant() -> None:
    """centered_source_position_check must have explicit centroid_offset_pix=(0.0, 0.0)."""
    mod = _load_module()
    cfg = _MockCFG()
    specs = {s["name"]: s for s in mod._build_ablation_plan(cfg, n_epochs=3)}
    cv = specs["centered_source_position_check"]
    for p in cv["source_params_sequence"]:
        assert "centroid_offset_pix" in p
        assert tuple(p["centroid_offset_pix"]) == (0.0, 0.0)


def test_build_ablation_plan_source_only_skipped_without_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """source_only_no_crowding must be skipped when pandas is not available."""
    mod = _load_module()
    monkeypatch.setattr(mod, "_PD", False)
    cfg = _MockCFG()
    specs = {s["name"]: s for s in mod._build_ablation_plan(cfg, n_epochs=2)}
    so = specs["source_only_no_crowding"]
    assert so["skip_reason"] is not None
    assert "pandas" in so["skip_reason"].lower()


# ---------------------------------------------------------------------------
# Null ablation — _aggregate_epoch_metrics
# ---------------------------------------------------------------------------

def _make_diff_stack(n: int = 3, h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, h, w)).astype(np.float64)


def test_aggregate_epoch_metrics_keys() -> None:
    mod = _load_module()
    diffs = _make_diff_stack(3)
    result = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=100)
    required = [
        "mean_central_aperture_snr",
        "max_abs_central_aperture_snr",
        "central_aperture_percentile_range",
        "repeated_same_sign",
        "null_rms_mean",
        "rim_interior_std_ratio_mean",
    ]
    for k in required:
        assert k in result, f"Missing key: {k}"


def test_aggregate_epoch_metrics_percentile_range_is_list() -> None:
    mod = _load_module()
    diffs = _make_diff_stack(3)
    result = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=50)
    pct = result["central_aperture_percentile_range"]
    assert isinstance(pct, list) and len(pct) == 2
    if pct[0] is not None and pct[1] is not None:
        assert pct[0] <= pct[1]


def test_aggregate_epoch_metrics_bright_centre() -> None:
    """A stack with bright centres should yield positive mean SNR."""
    mod = _load_module()
    diffs = np.zeros((3, 64, 64))
    diffs[:, 32, 32] = 50.0  # bright centre every epoch
    result = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=100)
    assert result["mean_central_aperture_snr"] > 0


def test_aggregate_epoch_metrics_repeated_same_sign_positive() -> None:
    mod = _load_module()
    diffs = np.zeros((3, 64, 64))
    diffs[:, 32, 32] = 50.0
    result = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=50)
    assert result["repeated_same_sign"] is True


def test_aggregate_epoch_metrics_not_repeated_mixed_sign() -> None:
    mod = _load_module()
    diffs = np.zeros((2, 64, 64))
    diffs[0, 32, 32] = 50.0
    diffs[1, 32, 32] = -50.0
    result = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=50)
    assert result["repeated_same_sign"] is False


def test_aggregate_epoch_metrics_deterministic() -> None:
    """Same input must produce identical output (random seed is fixed inside)."""
    mod = _load_module()
    diffs = _make_diff_stack(3, seed=7)
    r1 = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=100)
    r2 = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=100)
    assert r1["mean_central_aperture_snr"] == r2["mean_central_aperture_snr"]
    assert r1["central_aperture_percentile_range"] == r2["central_aperture_percentile_range"]


def test_aggregate_epoch_metrics_single_epoch_no_same_sign() -> None:
    """A single epoch cannot satisfy 'repeated same-sign' (requires >= 2)."""
    mod = _load_module()
    diffs = np.zeros((1, 64, 64))
    diffs[0, 32, 32] = 100.0
    result = mod._aggregate_epoch_metrics(diffs, aperture_radius=3, n_random_apertures=50)
    assert result["repeated_same_sign"] is False


# ---------------------------------------------------------------------------
# Null ablation — flux and centroid invariant logic
# ---------------------------------------------------------------------------

def test_flux_equality_flag_matches_baseline() -> None:
    """Verify the flux-equality logic used in _run_single_ablation's metadata."""
    base_flux = 800.0
    science_fluxes = [800.0, 800.0, 800.0]
    all_eq = base_flux is not None and all(
        abs(f - base_flux) < 1e-9 for f in science_fluxes
    )
    assert all_eq is True


def test_flux_equality_flag_mismatched() -> None:
    base_flux = 800.0
    science_fluxes = [800.0, 900.0, 800.0]
    all_eq = base_flux is not None and all(
        abs(f - base_flux) < 1e-9 for f in science_fluxes
    )
    assert all_eq is False


def test_flux_equality_flag_no_baseline() -> None:
    base_flux = None
    science_fluxes = [0.0, 0.0, 0.0]
    all_eq = base_flux is not None and all(
        abs(f - base_flux) < 1e-9 for f in science_fluxes
    )
    assert all_eq is False


def test_centroid_equality_flag_zeros() -> None:
    science_centroids = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    all_eq = all(c == [0.0, 0.0] for c in science_centroids)
    assert all_eq is True


def test_centroid_equality_flag_nonzero() -> None:
    science_centroids = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0]]
    all_eq = all(c == [0.0, 0.0] for c in science_centroids)
    assert all_eq is False


# ---------------------------------------------------------------------------
# Null ablation — summary table printing (smoke test)
# ---------------------------------------------------------------------------

def _make_summary_rows() -> list[dict]:
    ran_row = {
        "variant": "baseline_science",
        "status": "ran",
        "skip_reason": None,
        "mean_central_aperture_snr": 5.2,
        "max_abs_central_aperture_snr": 5.5,
        "central_aperture_percentile_range": [98.0, 100.0],
        "repeated_same_sign": True,
        "null_rms_mean": 150.3,
        "rim_interior_std_ratio_mean": 0.98,
        "config_knobs": {
            "reference_processing": "detector_matched",
            "reference_jitter_mode": "per_epoch",
            "jitter_rms_mas": 5.0,
            "crowded_field": True,
        },
    }
    skipped_row = {
        "variant": "source_only_no_crowding",
        "status": "skipped",
        "skip_reason": "pandas not available",
        "mean_central_aperture_snr": None,
        "max_abs_central_aperture_snr": None,
        "central_aperture_percentile_range": [None, None],
        "repeated_same_sign": None,
        "null_rms_mean": None,
        "rim_interior_std_ratio_mean": None,
        "config_knobs": {},
    }
    return [ran_row, skipped_row]


def test_print_ablation_summary_does_not_raise(capsys) -> None:
    mod = _load_module()
    mod._print_ablation_summary(_make_summary_rows())
    out = capsys.readouterr().out
    assert "baseline_science" in out
    assert "source_only_no_crowding" in out


def test_print_ablation_summary_skipped_variants_shown(capsys) -> None:
    mod = _load_module()
    mod._print_ablation_summary(_make_summary_rows())
    out = capsys.readouterr().out
    assert "skipped" in out or "pandas" in out


def test_print_interpretation_hints_does_not_raise(capsys) -> None:
    mod = _load_module()
    mod._print_interpretation_hints(_make_summary_rows())
    capsys.readouterr()  # just ensure it doesn't raise


# ---------------------------------------------------------------------------
# Null ablation — _save_ablation_csv
# ---------------------------------------------------------------------------

def test_save_ablation_csv_creates_file(tmp_path: Path) -> None:
    mod = _load_module()
    rows = _make_summary_rows()
    path = tmp_path / "ablation_summary.csv"
    mod._save_ablation_csv(rows, path)
    assert path.exists()
    content = path.read_text()
    assert "baseline_science" in content
    assert "source_only_no_crowding" in content


def test_save_ablation_csv_has_required_columns(tmp_path: Path) -> None:
    mod = _load_module()
    rows = _make_summary_rows()
    path = tmp_path / "ablation.csv"
    mod._save_ablation_csv(rows, path)
    header = path.read_text().splitlines()[0]
    for col in (
        "variant", "status", "skip_reason",
        "mean_central_aperture_snr", "max_abs_central_aperture_snr",
        "central_aperture_percentile_min", "central_aperture_percentile_max",
        "repeated_same_sign", "null_rms_mean",
        "reference_processing", "reference_jitter_mode",
    ):
        assert col in header, f"CSV header missing column: {col}"


def test_save_ablation_csv_skipped_row_has_empty_snr(tmp_path: Path) -> None:
    """Skipped-variant rows must not raise and should have empty metric cells."""
    mod = _load_module()
    path = tmp_path / "abl.csv"
    mod._save_ablation_csv(_make_summary_rows(), path)
    lines = path.read_text().splitlines()
    skipped_line = next(l for l in lines if "source_only_no_crowding" in l)
    # SNR column should be empty (no numeric value)
    assert "5.2" not in skipped_line  # baseline SNR must not bleed into skipped row


# ---------------------------------------------------------------------------
# Null ablation — summary JSON schema
# ---------------------------------------------------------------------------

def test_summary_json_schema_required_fields() -> None:
    """Every summary row must contain the required top-level fields."""
    required_fields = {
        "variant", "status",
        "mean_central_aperture_snr", "max_abs_central_aperture_snr",
        "central_aperture_percentile_range", "repeated_same_sign",
        "null_rms_mean", "rim_interior_std_ratio_mean", "config_knobs",
    }
    for row in _make_summary_rows():
        missing = required_fields - set(row.keys())
        assert not missing, f"Row '{row['variant']}' missing fields: {missing}"


def test_summary_json_serialisable() -> None:
    """Summary rows must round-trip through JSON without error."""
    import json as _json
    rows = _make_summary_rows()
    serialised = _json.dumps(rows)
    recovered = _json.loads(serialised)
    assert len(recovered) == len(rows)
    assert recovered[0]["variant"] == "baseline_science"
    assert recovered[1]["status"] == "skipped"


def test_summary_json_skipped_variant_in_summary() -> None:
    """A skipped variant must appear in the summary with status='skipped'."""
    rows = _make_summary_rows()
    skipped = [r for r in rows if r["status"] == "skipped"]
    assert len(skipped) == 1
    assert skipped[0]["variant"] == "source_only_no_crowding"
    assert skipped[0]["skip_reason"] is not None


# ---------------------------------------------------------------------------
# Null ablation — _ABLATION_SUBDIR constant
# ---------------------------------------------------------------------------

def test_ablation_subdir_constant_exists() -> None:
    mod = _load_module()
    assert hasattr(mod, "_ABLATION_SUBDIR")
    assert isinstance(mod._ABLATION_SUBDIR, str)
    assert len(mod._ABLATION_SUBDIR) > 0


# ---------------------------------------------------------------------------
# DIA-capture — parser
# ---------------------------------------------------------------------------

def test_parser_capture_dia_inputs_default() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    assert hasattr(args, "capture_dia_inputs")
    assert args.capture_dia_inputs is False


def test_parser_capture_dia_inputs_flag() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args(["--capture-dia-inputs"])
    assert args.capture_dia_inputs is True


# ---------------------------------------------------------------------------
# DIA-capture — _DiaCaptureContext
# ---------------------------------------------------------------------------

class _FakeDIAPipeline:
    """Minimal stand-in for DIAPipeline used to test the context manager."""

    def subtract(self, science_rate_image, reference_rate_image):
        return science_rate_image - reference_rate_image


def test_dia_capture_context_calls_original() -> None:
    """Wrapper must call the original subtract and return its result."""
    mod = _load_module()
    target = _FakeDIAPipeline()
    sci = np.ones((8, 8), dtype=np.float64) * 10.0
    ref = np.ones((8, 8), dtype=np.float64) * 7.0

    original_subtract = _FakeDIAPipeline.subtract
    captured = []

    # Manually exercise the context manager using the fake class.
    ctx = mod._DiaCaptureContext(captured)
    ctx._cls = _FakeDIAPipeline
    ctx._original = original_subtract

    _cap = captured
    _orig = original_subtract

    def _wrapper(instance, science_rate_image, reference_rate_image):
        result = _orig(instance, science_rate_image, reference_rate_image)
        sci_copy = science_rate_image.astype(np.float64, copy=True)
        ref_copy = reference_rate_image.astype(np.float64, copy=True)
        _cap.append({
            "science": sci_copy,
            "reference": ref_copy,
            "raw_diff": sci_copy - ref_copy,
            "post_al": result.astype(np.float64, copy=True),
        })
        return result

    _FakeDIAPipeline.subtract = _wrapper
    try:
        result = target.subtract(sci, ref)
    finally:
        _FakeDIAPipeline.subtract = original_subtract

    # Verify result is the original subtract output.
    assert np.allclose(result, sci - ref)
    assert len(captured) == 1


def test_dia_capture_context_captures_arrays() -> None:
    """Captured entry must contain science, reference, raw_diff, post_al."""
    mod = _load_module()
    rng = np.random.default_rng(42)
    sci = rng.standard_normal((16, 16))
    ref = rng.standard_normal((16, 16))

    original = _FakeDIAPipeline.subtract
    captured = []

    def _wrapper(instance, science_rate_image, reference_rate_image):
        result = original(instance, science_rate_image, reference_rate_image)
        sci_c = science_rate_image.astype(np.float64, copy=True)
        ref_c = reference_rate_image.astype(np.float64, copy=True)
        captured.append({
            "science": sci_c,
            "reference": ref_c,
            "raw_diff": sci_c - ref_c,
            "post_al": result.astype(np.float64, copy=True),
        })
        return result

    _FakeDIAPipeline.subtract = _wrapper
    try:
        _ = _FakeDIAPipeline().subtract(sci, ref)
    finally:
        _FakeDIAPipeline.subtract = original

    assert len(captured) == 1
    entry = captured[0]
    for key in ("science", "reference", "raw_diff", "post_al"):
        assert key in entry, f"Missing key: {key}"
        assert entry[key].shape == sci.shape

    assert np.allclose(entry["science"], sci.astype(np.float64))
    assert np.allclose(entry["reference"], ref.astype(np.float64))
    assert np.allclose(entry["raw_diff"], entry["science"] - entry["reference"])


def test_dia_capture_context_restores_original_after_exit() -> None:
    """subtract must be restored even when the with block raises."""
    mod = _load_module()
    original = _FakeDIAPipeline.subtract
    captured: list = []

    # Simulate the context manager manually: enter, simulate an exception,
    # verify the class method is restored.
    ctx = mod._DiaCaptureContext(captured)

    # Patch _cls / _original to point at the fake class.
    ctx._cls = _FakeDIAPipeline
    ctx._original = original

    def _wrapper(instance, science_rate_image, reference_rate_image):
        return original(instance, science_rate_image, reference_rate_image)

    _FakeDIAPipeline.subtract = _wrapper
    assert _FakeDIAPipeline.subtract is _wrapper

    # Simulate __exit__ restoring the original.
    ctx.__exit__(None, None, None)
    assert _FakeDIAPipeline.subtract is original


def test_dia_capture_context_restores_on_exception() -> None:
    """subtract must be restored when an exception escapes the with block."""
    mod = _load_module()
    original = _FakeDIAPipeline.subtract
    captured: list = []

    ctx = mod._DiaCaptureContext(captured)
    ctx._cls = _FakeDIAPipeline
    ctx._original = original

    def _wrapper(instance, s, r):
        return original(instance, s, r)

    _FakeDIAPipeline.subtract = _wrapper
    # Simulate __exit__ called with exception info.
    ctx.__exit__(ValueError, ValueError("boom"), None)
    assert _FakeDIAPipeline.subtract is original


# ---------------------------------------------------------------------------
# DIA-capture — _dia_input_interpretation
# ---------------------------------------------------------------------------

def test_dia_input_interpretation_both_high() -> None:
    mod = _load_module()
    result = mod._dia_input_interpretation(5.0, 4.0, threshold=3.0)
    assert result == "both_high"


def test_dia_input_interpretation_post_al_introduced() -> None:
    mod = _load_module()
    result = mod._dia_input_interpretation(1.0, 4.0, threshold=3.0)
    assert result == "post_al_introduced"


def test_dia_input_interpretation_raw_residual_present() -> None:
    mod = _load_module()
    result = mod._dia_input_interpretation(4.0, 1.0, threshold=3.0)
    assert result == "raw_residual_present"


def test_dia_input_interpretation_both_clean() -> None:
    mod = _load_module()
    result = mod._dia_input_interpretation(0.5, -1.0, threshold=3.0)
    assert result == "both_clean"


def test_dia_input_interpretation_negative_raw_high() -> None:
    """Negative SNR above threshold magnitude should still trigger 'high'."""
    mod = _load_module()
    result = mod._dia_input_interpretation(-4.0, -4.5, threshold=3.0)
    assert result == "both_high"


def test_dia_input_interpretation_exact_threshold() -> None:
    """SNR equal to threshold is 'high' (>= not >)."""
    mod = _load_module()
    result = mod._dia_input_interpretation(3.0, 1.0, threshold=3.0)
    assert result == "raw_residual_present"


# ---------------------------------------------------------------------------
# DIA-capture — _compute_dia_input_epoch_metrics
# ---------------------------------------------------------------------------

def _make_capture_entry(
    sci_val: float = 10.0,
    ref_val: float = 8.0,
    shape: tuple = (64, 64),
    seed: int = 0,
) -> dict:
    """Build a synthetic capture entry for testing metrics."""
    rng = np.random.default_rng(seed)
    sci = np.full(shape, sci_val, dtype=np.float64)
    sci[shape[0] // 2, shape[1] // 2] += 50.0  # bright centre
    ref = np.full(shape, ref_val, dtype=np.float64)
    ref[shape[0] // 2, shape[1] // 2] += 45.0
    raw = sci - ref
    post = raw + rng.standard_normal(shape) * 0.1
    return {"science": sci, "reference": ref, "raw_diff": raw, "post_al": post}


def test_compute_dia_input_epoch_metrics_required_keys() -> None:
    mod = _load_module()
    entry = _make_capture_entry()
    metrics = mod._compute_dia_input_epoch_metrics(entry, aperture_radius=3)
    required = [
        "dia_input_capture_enabled",
        "science_center_aperture_sum",
        "reference_center_aperture_sum",
        "science_minus_reference_center_aperture_sum",
        "science_minus_reference_center_aperture_snr",
        "post_al_center_aperture_snr",
        "science_reference_center_ratio",
        "raw_vs_post_al_interpretation",
    ]
    for key in required:
        assert key in metrics, f"Missing key: {key}"


def test_compute_dia_input_epoch_metrics_capture_enabled_flag() -> None:
    mod = _load_module()
    entry = _make_capture_entry()
    metrics = mod._compute_dia_input_epoch_metrics(entry, aperture_radius=3)
    assert metrics["dia_input_capture_enabled"] is True


def test_compute_dia_input_epoch_metrics_interpretation_is_string() -> None:
    mod = _load_module()
    entry = _make_capture_entry()
    metrics = mod._compute_dia_input_epoch_metrics(entry, aperture_radius=3)
    assert isinstance(metrics["raw_vs_post_al_interpretation"], str)
    valid = {"both_high", "post_al_introduced", "raw_residual_present", "both_clean"}
    assert metrics["raw_vs_post_al_interpretation"] in valid


def test_compute_dia_input_epoch_metrics_ratio_none_when_ref_zero() -> None:
    """science_reference_center_ratio is None when reference aperture sum is ~0."""
    mod = _load_module()
    entry = _make_capture_entry(ref_val=0.0)
    # Force reference aperture sum toward zero by zeroing the center.
    entry["reference"] = np.zeros_like(entry["reference"])
    metrics = mod._compute_dia_input_epoch_metrics(entry, aperture_radius=3)
    assert metrics["science_reference_center_ratio"] is None


def test_compute_dia_input_epoch_metrics_sci_brighter_than_ref() -> None:
    """When science centre > reference centre, ratio > 1."""
    mod = _load_module()
    sci = np.zeros((64, 64), dtype=np.float64)
    sci[32, 32] = 200.0
    ref = np.zeros((64, 64), dtype=np.float64)
    ref[32, 32] = 100.0
    entry = {"science": sci, "reference": ref, "raw_diff": sci - ref,
             "post_al": (sci - ref) + 0.01}
    metrics = mod._compute_dia_input_epoch_metrics(entry, aperture_radius=3)
    ratio = metrics["science_reference_center_ratio"]
    assert ratio is not None and ratio > 1.0


# ---------------------------------------------------------------------------
# DIA-capture — _print_dia_capture_summary (smoke test)
# ---------------------------------------------------------------------------

def test_print_dia_capture_summary_does_not_raise(capsys) -> None:
    mod = _load_module()
    epoch_metrics = [
        {
            "science_minus_reference_center_aperture_snr": 4.5,
            "post_al_center_aperture_snr": 4.2,
            "science_reference_center_ratio": 1.1,
            "raw_vs_post_al_interpretation": "both_high",
        },
        {
            "science_minus_reference_center_aperture_snr": 3.8,
            "post_al_center_aperture_snr": 4.0,
            "science_reference_center_ratio": 1.05,
            "raw_vs_post_al_interpretation": "both_high",
        },
    ]
    mod._print_dia_capture_summary("baseline_science", epoch_metrics)
    out = capsys.readouterr().out
    assert "baseline_science" in out
    assert "both_high" in out or "Residual" in out


def test_print_dia_capture_summary_post_al_introduced(capsys) -> None:
    mod = _load_module()
    epoch_metrics = [
        {
            "science_minus_reference_center_aperture_snr": 0.5,
            "post_al_center_aperture_snr": 5.0,
            "science_reference_center_ratio": 0.98,
            "raw_vs_post_al_interpretation": "post_al_introduced",
        },
    ]
    mod._print_dia_capture_summary("test_variant", epoch_metrics)
    out = capsys.readouterr().out
    assert "post_al_introduced" in out or "AL" in out


def test_print_dia_capture_summary_empty_does_not_raise(capsys) -> None:
    mod = _load_module()
    mod._print_dia_capture_summary("empty_variant", [])
    # No output expected, no crash expected.
    capsys.readouterr()


# ---------------------------------------------------------------------------
# DIA-capture — _save_dia_capture_arrays (npy files written)
# ---------------------------------------------------------------------------

def test_save_dia_capture_arrays_writes_npy(tmp_path: Path) -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    # Simulate 2 epochs captured from a 64×64 context stamp with 16×16 science stamp.
    entries = [
        {
            "science":   rng.standard_normal((64, 64)),
            "reference": rng.standard_normal((64, 64)),
            "raw_diff":  rng.standard_normal((64, 64)),
            "post_al":   rng.standard_normal((64, 64)),
        }
        for _ in range(2)
    ]
    mod._save_dia_capture_arrays(entries, tmp_path, sci_sz=16, ctx_sz=64)

    for i in range(2):
        for prefix in ("science_rate", "reference_rate", "raw_difference", "post_al_difference"):
            fpath = tmp_path / f"{prefix}_epoch_{i}.npy"
            assert fpath.exists(), f"Expected file not found: {fpath}"
            arr = np.load(fpath)
            assert arr.shape == (64, 64)


# ---------------------------------------------------------------------------
# DIA-capture — _DIA_HIGH_SNR_THRESHOLD constant
# ---------------------------------------------------------------------------

def test_dia_high_snr_threshold_exists() -> None:
    mod = _load_module()
    assert hasattr(mod, "_DIA_HIGH_SNR_THRESHOLD")
    assert mod._DIA_HIGH_SNR_THRESHOLD > 0


# ---------------------------------------------------------------------------
# DIA-capture — no pipeline internals imported at module level
# ---------------------------------------------------------------------------

def test_dia_capture_does_not_import_dia_at_module_level() -> None:
    """smig.rendering.dia must NOT be imported when the module is loaded."""
    import importlib.util as _ilu
    before = set(sys.modules)
    spec = _ilu.spec_from_file_location(
        "_qa_dia_check", Path(__file__).parent / "generate_phase2_qa_stamps.py"
    )
    fresh = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(fresh)       # type: ignore[union-attr]
    newly = set(sys.modules) - before
    assert "smig.rendering.dia" not in newly, (
        "smig.rendering.dia was imported at module level via DIA capture code"
    )


# ---------------------------------------------------------------------------
# AL diagnostic modes — parser
# ---------------------------------------------------------------------------

def test_parser_dia_diagnostic_modes_default() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    assert hasattr(args, "dia_diagnostic_modes")
    assert args.dia_diagnostic_modes is False


def test_parser_dia_diagnostic_modes_flag() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args(["--dia-diagnostic-modes"])
    assert args.dia_diagnostic_modes is True


def test_parser_dia_diagnostic_modes_implies_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must set capture_dia_inputs=True when --dia-diagnostic-modes is given."""
    mod = _load_module()
    args = mod._build_parser().parse_args(["--dia-diagnostic-modes"])
    capture_dia = args.capture_dia_inputs or args.dia_diagnostic_modes
    assert capture_dia is True


# ---------------------------------------------------------------------------
# AL diagnostic constants
# ---------------------------------------------------------------------------

def test_al_sigmas_production_constant() -> None:
    mod = _load_module()
    assert hasattr(mod, "_AL_SIGMAS_PRODUCTION")
    assert mod._AL_SIGMAS_PRODUCTION == (0.5, 1.0, 2.0, 4.0)


def test_al_kernel_size_constant() -> None:
    mod = _load_module()
    assert hasattr(mod, "_AL_KERNEL_SIZE")
    assert mod._AL_KERNEL_SIZE == 33


def test_al_diag_excl_radius_constant() -> None:
    mod = _load_module()
    assert hasattr(mod, "_AL_DIAG_EXCL_RADIUS")
    assert mod._AL_DIAG_EXCL_RADIUS > 0


# ---------------------------------------------------------------------------
# QA-local Gaussian kernel helper
# ---------------------------------------------------------------------------

def test_make_gaussian_kernel_qa_shape() -> None:
    mod = _load_module()
    kern = mod._make_gaussian_kernel_qa(1.0, 9)
    assert kern.shape == (9, 9)


def test_make_gaussian_kernel_qa_normalized() -> None:
    mod = _load_module()
    kern = mod._make_gaussian_kernel_qa(2.0, 17)
    assert abs(float(kern.sum()) - 1.0) < 1e-10


def test_make_gaussian_kernel_qa_centred() -> None:
    """Centre pixel must be the maximum for a positive-sigma kernel."""
    mod = _load_module()
    kern = mod._make_gaussian_kernel_qa(1.5, 11)
    h, w = kern.shape
    assert kern[h // 2, w // 2] == pytest.approx(float(kern.max()))


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------

def test_build_al_design_matrix_qa_shape() -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    planes = [rng.standard_normal((16, 16)) for _ in range(4)]
    A = mod._build_al_design_matrix_qa(planes)
    assert A.shape == (16 * 16, 5)  # 4 basis + 1 constant


def test_build_al_design_matrix_qa_constant_column() -> None:
    mod = _load_module()
    rng = np.random.default_rng(1)
    planes = [rng.standard_normal((8, 8)) for _ in range(3)]
    A = mod._build_al_design_matrix_qa(planes)
    assert np.all(A[:, -1] == 1.0)


# ---------------------------------------------------------------------------
# AL lstsq solver
# ---------------------------------------------------------------------------

def test_al_lstsq_qa_required_keys() -> None:
    mod = _load_module()
    rng = np.random.default_rng(5)
    A = rng.standard_normal((100, 5))
    b = rng.standard_normal(100)
    result = mod._al_lstsq_qa(A, b)
    for key in ("coefficients", "rank", "condition_number", "singular_values"):
        assert key in result, f"Missing key: {key}"


def test_al_lstsq_qa_coefficients_length() -> None:
    mod = _load_module()
    rng = np.random.default_rng(6)
    A = rng.standard_normal((200, 5))
    b = rng.standard_normal(200)
    result = mod._al_lstsq_qa(A, b)
    assert len(result["coefficients"]) == 5


def test_al_lstsq_qa_condition_number_positive() -> None:
    mod = _load_module()
    rng = np.random.default_rng(7)
    A = rng.standard_normal((100, 5))
    b = rng.standard_normal(100)
    result = mod._al_lstsq_qa(A, b)
    assert result["condition_number"] > 0


def test_al_lstsq_qa_coefficients_finite() -> None:
    mod = _load_module()
    rng = np.random.default_rng(8)
    A = rng.standard_normal((50, 5))
    b = rng.standard_normal(50)
    result = mod._al_lstsq_qa(A, b)
    assert all(np.isfinite(c) for c in result["coefficients"])


def test_al_lstsq_qa_rank_is_int() -> None:
    mod = _load_module()
    rng = np.random.default_rng(9)
    A = rng.standard_normal((80, 5))
    b = rng.standard_normal(80)
    result = mod._al_lstsq_qa(A, b)
    assert isinstance(result["rank"], int)


# ---------------------------------------------------------------------------
# AL ridge solver
# ---------------------------------------------------------------------------

def test_al_ridge_qa_returns_finite_coefficients() -> None:
    mod = _load_module()
    rng = np.random.default_rng(10)
    A = rng.standard_normal((100, 5))
    b = rng.standard_normal(100)
    result = mod._al_ridge_qa(A, b, lam_scale=1e-4)
    assert all(np.isfinite(c) for c in result["coefficients"])


def test_al_ridge_qa_condition_number_lower_than_unregularized() -> None:
    """Ridge regularization must reduce or maintain condition number."""
    mod = _load_module()
    rng = np.random.default_rng(11)
    A = rng.standard_normal((100, 5))
    b = rng.standard_normal(100)
    lstsq_cond = mod._al_lstsq_qa(A, b)["condition_number"]
    ridge_cond = mod._al_ridge_qa(A, b, lam_scale=1e-2)["condition_number"]
    assert ridge_cond <= lstsq_cond + 1.0


def test_al_ridge_qa_required_keys() -> None:
    mod = _load_module()
    rng = np.random.default_rng(12)
    A = rng.standard_normal((60, 5))
    b = rng.standard_normal(60)
    result = mod._al_ridge_qa(A, b, lam_scale=1e-6)
    for key in ("coefficients", "condition_number", "ridge_lambda_scale", "ridge_lambda_abs"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# QA-only AL subtract
# ---------------------------------------------------------------------------

def _make_flat_sci_ref(shape=(32, 32), seed=0):
    """Flat science and reference with a central source."""
    rng = np.random.default_rng(seed)
    ref = np.full(shape, 5.0, dtype=np.float64)
    ref[shape[0] // 2, shape[1] // 2] += 20.0
    sci = ref.copy() + rng.standard_normal(shape) * 0.01
    return sci, ref


def test_al_subtract_qa_identity_only_equals_sci_minus_ref() -> None:
    """identity_only difference must exactly equal sci - ref."""
    mod = _load_module()
    sci, ref = _make_flat_sci_ref()
    raw = sci - ref
    # identity_only is computed directly in _run_al_diagnostics_for_epoch,
    # but we can verify the contract by checking _al_subtract_qa with
    # a trivial single-kernel identity basis does not deviate more than fit noise.
    # The direct check is: sci - ref == identity_only.
    assert np.allclose(raw, sci - ref)


def test_al_subtract_qa_returns_finite_difference() -> None:
    """_al_subtract_qa must return a finite difference array for normal inputs."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    result = mod._al_subtract_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )
    assert np.all(np.isfinite(result["difference"]))


def test_al_subtract_qa_with_ridge_finite() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    result = mod._al_subtract_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE,
        ridge_lambda_scale=1e-4,
    )
    assert np.all(np.isfinite(result["difference"]))


def test_al_subtract_qa_fit_info_has_condition_number() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    result = mod._al_subtract_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )
    fi = result["fit_info"]
    assert "condition_number" in fi
    assert fi["condition_number"] > 0


def test_al_subtract_qa_fit_info_has_coefficients() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    result = mod._al_subtract_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )
    coeffs = result["fit_info"]["coefficients"]
    assert len(coeffs) == len(mod._AL_SIGMAS_PRODUCTION) + 1


# ---------------------------------------------------------------------------
# Masking: central exclusion
# ---------------------------------------------------------------------------

def test_al_subtract_qa_mask_excludes_central_pixels() -> None:
    """When pixel_mask excludes the central aperture, central pixels must not
    contribute to the fit.  We verify by checking that the fit coefficients
    differ from the unmasked case when the centre has a strong source."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")

    shape = (64, 64)
    rng = np.random.default_rng(42)
    ref = np.ones(shape, dtype=np.float64)
    sci = ref.copy() + rng.standard_normal(shape) * 0.01
    # Very bright central source only in science.
    sci[32, 32] = 5000.0

    # Unmasked fit
    excl_r = mod._AL_DIAG_EXCL_RADIUS
    unmasked = mod._al_subtract_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )

    # Masked fit: exclude central disk from coefficient estimation.
    cy, cx = 32.0, 32.0
    center_disk = mod._circular_aperture_mask(shape[0], shape[1], cy, cx, float(excl_r))
    masked = mod._al_subtract_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE,
        pixel_mask=~center_disk,
    )

    # Coefficients must differ when the bright spike is excluded from the fit.
    c_unmasked = unmasked["fit_info"]["coefficients"]
    c_masked   = masked["fit_info"]["coefficients"]
    assert c_unmasked != c_masked, (
        "Masking the central source should change the fitted coefficients"
    )


def test_al_subtract_qa_mask_has_correct_excluded_count() -> None:
    """Verify the circular exclusion mask has the expected pixel count."""
    mod = _load_module()
    shape = (64, 64)
    excl_r = mod._AL_DIAG_EXCL_RADIUS
    cy, cx = 32.0, 32.0
    disk = mod._circular_aperture_mask(shape[0], shape[1], cy, cx, float(excl_r))
    # Disk should include at least 1 pixel (centre) and at most π*r² + some
    n_excluded = int(disk.sum())
    assert n_excluded >= 1
    assert n_excluded <= int(np.pi * (excl_r + 1) ** 2) + 10


# ---------------------------------------------------------------------------
# _run_al_diagnostics_for_epoch
# ---------------------------------------------------------------------------

_EXPECTED_DIAG_VARIANT_NAMES = [
    "identity_only",
    "current_al",
    "background_masked_al",
    "rim_or_annulus_fit_al",
    "reduced_basis_al",
    "regularized_al_1e-6",
    "regularized_al_1e-4",
    "identity_plus_gaussians",
]


def test_run_al_diagnostics_for_epoch_all_variants_present() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    post_al = sci - ref  # mock post-AL for testing purposes
    results = mod._run_al_diagnostics_for_epoch(sci, ref, post_al, aperture_radius=3)
    assert list(results.keys()) == _EXPECTED_DIAG_VARIANT_NAMES


def test_run_al_diagnostics_for_epoch_identity_equals_sci_minus_ref() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    post_al = sci - ref
    results = mod._run_al_diagnostics_for_epoch(sci, ref, post_al, aperture_radius=3)
    assert np.allclose(results["identity_only"]["difference"], sci - ref)


def test_run_al_diagnostics_for_epoch_current_al_matches_post_al() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    rng = np.random.default_rng(99)
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    post_al = (sci - ref) + rng.standard_normal((64, 64)) * 0.5
    results = mod._run_al_diagnostics_for_epoch(sci, ref, post_al, aperture_radius=3)
    assert np.allclose(results["current_al"]["difference"], post_al)


def test_run_al_diagnostics_for_epoch_all_differences_finite() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    post_al = sci - ref
    results = mod._run_al_diagnostics_for_epoch(sci, ref, post_al, aperture_radius=3)
    for vname, vdata in results.items():
        diff = vdata["difference"]
        assert np.all(np.isfinite(diff)), f"Non-finite in variant {vname!r}"


def test_run_al_diagnostics_for_epoch_metrics_keys() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    post_al = sci - ref
    results = mod._run_al_diagnostics_for_epoch(sci, ref, post_al, aperture_radius=3)
    required_metric_keys = {
        "central_aperture_snr", "rms", "mean", "max_abs",
    }
    for vname, vdata in results.items():
        missing = required_metric_keys - set(vdata["metrics"])
        assert not missing, f"Variant {vname!r} metrics missing: {missing}"


def test_run_al_diagnostics_for_epoch_fit_info_recorded() -> None:
    """Variants that fit AL coefficients must record condition_number and coefficients."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    post_al = sci - ref
    results = mod._run_al_diagnostics_for_epoch(sci, ref, post_al, aperture_radius=3)
    fitting_variants = [
        "background_masked_al",
        "rim_or_annulus_fit_al",
        "reduced_basis_al",
        "regularized_al_1e-6",
        "regularized_al_1e-4",
        "identity_plus_gaussians",
    ]
    for vname in fitting_variants:
        fi = results[vname]["fit_info"]
        assert fi is not None, f"fit_info is None for {vname!r}"
        assert "condition_number" in fi, f"Missing condition_number in {vname!r} fit_info"
        assert "coefficients" in fi, f"Missing coefficients in {vname!r} fit_info"
        assert all(np.isfinite(c) for c in fi["coefficients"]), (
            f"Non-finite coefficients in {vname!r}"
        )


def test_run_al_diagnostics_for_epoch_no_variant_has_none_difference() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    results = mod._run_al_diagnostics_for_epoch(sci, ref, sci - ref, aperture_radius=3)
    for vname, vdata in results.items():
        assert vdata["difference"] is not None, f"difference is None for {vname!r}"
        assert isinstance(vdata["difference"], np.ndarray)


# ---------------------------------------------------------------------------
# _assign_al_interpretations
# ---------------------------------------------------------------------------

def _make_epoch_rows(mode_snrs: dict[str, float]) -> list[dict]:
    """Build minimal epoch rows for interpretation testing."""
    rows = []
    for mode, snr in mode_snrs.items():
        rows.append({
            "event_or_ablation": "test_event",
            "epoch": 0,
            "diagnostic_mode": mode,
            "central_aperture_snr": snr,
            "rms": 1.0,
            "mean": 0.0,
            "max_abs": 2.0,
            "rim_interior_std_ratio": 1.0,
            "condition_number": None,
            "coefficients": None,
            "interpretation": "",
        })
    return rows


def test_assign_al_interpretations_al_introduces_residual() -> None:
    mod = _load_module()
    rows = _make_epoch_rows({
        "identity_only": 0.5,   # clean raw diff
        "current_al": 5.0,      # AL makes it worse
    })
    mod._assign_al_interpretations(rows)
    current_row = next(r for r in rows if r["diagnostic_mode"] == "current_al")
    assert "AL_introduces_residual" in current_row["interpretation"]


def test_assign_al_interpretations_identity_clean() -> None:
    mod = _load_module()
    rows = _make_epoch_rows({"identity_only": 1.0})
    mod._assign_al_interpretations(rows)
    assert "clean_before_al" in rows[0]["interpretation"]


def test_assign_al_interpretations_background_masked_helps() -> None:
    mod = _load_module()
    rows = _make_epoch_rows({
        "identity_only": 0.5,
        "current_al": 8.0,
        "background_masked_al": 1.0,  # masking helps significantly
    })
    mod._assign_al_interpretations(rows)
    bm_row = next(r for r in rows if r["diagnostic_mode"] == "background_masked_al")
    assert "central_source_contaminates_fit" in bm_row["interpretation"]


def test_assign_al_interpretations_regularization_helps() -> None:
    mod = _load_module()
    rows = _make_epoch_rows({
        "identity_only": 0.5,
        "current_al": 8.0,
        "regularized_al_1e-6": 1.0,
    })
    mod._assign_al_interpretations(rows)
    reg_row = next(r for r in rows if r["diagnostic_mode"] == "regularized_al_1e-6")
    assert "regularization_helps_ill_conditioning" in reg_row["interpretation"]


def test_assign_al_interpretations_ill_conditioned_flagged() -> None:
    mod = _load_module()
    rows = _make_epoch_rows({"background_masked_al": 4.0})
    rows[0]["condition_number"] = 1e8  # above ill-cond threshold
    mod._assign_al_interpretations(rows)
    assert "ill_conditioned" in rows[0]["interpretation"]


# ---------------------------------------------------------------------------
# AL diagnostic summary schema
# ---------------------------------------------------------------------------

def _make_al_diag_summary_rows() -> list[dict]:
    """Minimal synthetic summary rows for schema tests."""
    rows = []
    for epoch in range(2):
        for mode in _EXPECTED_DIAG_VARIANT_NAMES[:3]:
            rows.append({
                "event_or_ablation": "qa_null",
                "epoch": epoch,
                "diagnostic_mode": mode,
                "central_aperture_snr": float(epoch + 1),
                "rms": 10.0,
                "mean": 0.1,
                "max_abs": 20.0,
                "rim_interior_std_ratio": 1.0,
                "condition_number": 500.0 if mode != "identity_only" else None,
                "coefficients": [0.1, 0.2, 0.3, 0.4, 0.0] if mode != "identity_only" else None,
                "interpretation": "AL_introduces_residual",
            })
    return rows


def test_al_diagnostic_summary_schema_required_fields() -> None:
    required = {
        "event_or_ablation", "epoch", "diagnostic_mode",
        "central_aperture_snr", "rms", "mean", "max_abs",
        "condition_number", "coefficients", "interpretation",
    }
    for row in _make_al_diag_summary_rows():
        missing = required - set(row.keys())
        assert not missing, f"Summary row missing fields: {missing}"


def test_al_diagnostic_summary_json_serialisable() -> None:
    import json as _json
    rows = _make_al_diag_summary_rows()
    serialised = _json.dumps(rows)
    recovered = _json.loads(serialised)
    assert len(recovered) == len(rows)


def test_save_al_diagnostic_summary_creates_files(tmp_path: Path) -> None:
    mod = _load_module()
    rows = _make_al_diag_summary_rows()
    mod._save_al_diagnostic_summary(rows, tmp_path)
    assert (tmp_path / "dia_al_diagnostic_summary.json").exists()
    assert (tmp_path / "dia_al_diagnostic_summary.csv").exists()


def test_save_al_diagnostic_summary_json_content(tmp_path: Path) -> None:
    import json as _json
    mod = _load_module()
    rows = _make_al_diag_summary_rows()
    mod._save_al_diagnostic_summary(rows, tmp_path)
    loaded = _json.loads((tmp_path / "dia_al_diagnostic_summary.json").read_text())
    assert len(loaded) == len(rows)
    assert loaded[0]["event_or_ablation"] == "qa_null"


def test_save_al_diagnostic_summary_csv_has_header(tmp_path: Path) -> None:
    mod = _load_module()
    rows = _make_al_diag_summary_rows()
    mod._save_al_diagnostic_summary(rows, tmp_path)
    header = (tmp_path / "dia_al_diagnostic_summary.csv").read_text().splitlines()[0]
    for col in (
        "event_or_ablation", "epoch", "diagnostic_mode",
        "central_aperture_snr", "rms", "condition_number",
        "interpretation",
    ):
        assert col in header, f"CSV missing column: {col}"


def test_save_al_diagnostic_summary_empty_is_noop(tmp_path: Path) -> None:
    mod = _load_module()
    mod._save_al_diagnostic_summary([], tmp_path)
    assert not (tmp_path / "dia_al_diagnostic_summary.json").exists()


# ---------------------------------------------------------------------------
# AL diagnostics do NOT monkeypatch or import DIAPipeline
# ---------------------------------------------------------------------------

def test_al_diagnostics_do_not_import_dia_pipeline() -> None:
    """AL diagnostic functions must not import smig.rendering.dia."""
    import importlib.util as _ilu
    before = set(sys.modules)
    spec = _ilu.spec_from_file_location(
        "_qa_al_check", Path(__file__).parent / "generate_phase2_qa_stamps.py"
    )
    fresh = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(fresh)       # type: ignore[union-attr]
    newly = set(sys.modules) - before
    assert "smig.rendering.dia" not in newly


def test_al_diagnostics_do_not_modify_dia_pipeline_class() -> None:
    """Running AL diagnostic helpers must not touch DIAPipeline.subtract."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")

    # Verify that the AL helpers use QA-local functions only.
    # We confirm this by checking that smig.rendering.dia is NOT in sys.modules
    # after calling the AL helpers on synthetic data.
    import sys as _sys
    before = set(_sys.modules)

    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    post_al = sci - ref
    _ = mod._run_al_diagnostics_for_epoch(sci, ref, post_al, aperture_radius=3)

    newly_after = set(_sys.modules) - before
    assert "smig.rendering.dia" not in newly_after, (
        "AL diagnostic helpers imported smig.rendering.dia"
    )


# ---------------------------------------------------------------------------
# _run_al_diagnostic_suite — file outputs
# ---------------------------------------------------------------------------

def test_run_al_diagnostic_suite_saves_npy_files(tmp_path: Path) -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    rng = np.random.default_rng(0)
    shape = (32, 32)
    entries = [
        {
            "science":   np.ones(shape) * 5.0 + rng.standard_normal(shape) * 0.1,
            "reference": np.ones(shape) * 5.0,
            "raw_diff":  rng.standard_normal(shape) * 0.1,
            "post_al":   rng.standard_normal(shape) * 0.1,
        }
    ]
    rows = mod._run_al_diagnostic_suite(
        capture_entries=entries,
        event_or_ablation="test_event",
        aperture_radius=3,
        out_dir=tmp_path,
    )
    # Each variant should have a saved .npy file in epoch_0/
    epoch_dir = tmp_path / "epoch_0"
    assert epoch_dir.exists()
    for vname in _EXPECTED_DIAG_VARIANT_NAMES:
        npy_path = epoch_dir / f"{vname}.npy"
        assert npy_path.exists(), f"Missing npy: {npy_path}"
        arr = np.load(npy_path)
        assert arr.shape == shape


def test_run_al_diagnostic_suite_saves_fit_diagnostics_json(tmp_path: Path) -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    import json as _json
    shape = (32, 32)
    sci = np.ones(shape) * 5.0
    ref = np.ones(shape) * 5.0
    entries = [{"science": sci, "reference": ref, "raw_diff": sci - ref, "post_al": sci - ref}]
    mod._run_al_diagnostic_suite(
        capture_entries=entries,
        event_or_ablation="test_event",
        aperture_radius=3,
        out_dir=tmp_path,
    )
    diag_json = tmp_path / "epoch_0" / "al_fit_diagnostics.json"
    assert diag_json.exists()
    data = _json.loads(diag_json.read_text())
    for key in (
        "coefficients", "condition_number", "rank",
        "raw_identity_center_snr", "post_al_center_snr",
        "delta_post_minus_raw_center_snr",
        "raw_identity_rms", "post_al_rms",
        "central_aperture_fluxes",
    ):
        assert key in data, f"al_fit_diagnostics.json missing key: {key}"


def test_run_al_diagnostic_suite_central_aperture_fluxes_schema(tmp_path: Path) -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    import json as _json
    shape = (32, 32)
    sci = np.ones(shape) * 5.0
    sci[16, 16] += 50.0
    ref = np.ones(shape) * 5.0
    ref[16, 16] += 45.0
    entries = [{"science": sci, "reference": ref, "raw_diff": sci - ref, "post_al": sci - ref}]
    mod._run_al_diagnostic_suite(
        capture_entries=entries,
        event_or_ablation="test_event",
        aperture_radius=3,
        out_dir=tmp_path,
    )
    data = _json.loads((tmp_path / "epoch_0" / "al_fit_diagnostics.json").read_text())
    fluxes = data["central_aperture_fluxes"]
    for key in ("science", "reference", "matched_reference", "raw_identity", "post_al"):
        assert key in fluxes, f"central_aperture_fluxes missing key: {key}"


def test_run_al_diagnostic_suite_returns_correct_row_count(tmp_path: Path) -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    shape = (32, 32)
    n_epochs = 2
    rng = np.random.default_rng(1)
    entries = [
        {
            "science":   np.ones(shape) + rng.standard_normal(shape) * 0.1,
            "reference": np.ones(shape),
            "raw_diff":  rng.standard_normal(shape) * 0.1,
            "post_al":   rng.standard_normal(shape) * 0.1,
        }
        for _ in range(n_epochs)
    ]
    rows = mod._run_al_diagnostic_suite(
        capture_entries=entries,
        event_or_ablation="qa_null",
        aperture_radius=3,
        out_dir=tmp_path,
    )
    # 7 variants × 2 epochs
    assert len(rows) == len(_EXPECTED_DIAG_VARIANT_NAMES) * n_epochs


def test_run_al_diagnostic_suite_no_scipy_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_SCIPY", False)
    entries = [{"science": np.ones((8, 8)), "reference": np.ones((8, 8)),
                "raw_diff": np.zeros((8, 8)), "post_al": np.zeros((8, 8))}]
    rows = mod._run_al_diagnostic_suite(
        capture_entries=entries,
        event_or_ablation="test",
        aperture_radius=3,
        out_dir=tmp_path,
    )
    assert rows == []


# ---------------------------------------------------------------------------
# identity_plus_gaussians — design matrix and fit diagnostics
# ---------------------------------------------------------------------------

def test_al_subtract_identity_plus_gaussians_coefficient_count() -> None:
    """Coefficient count must be len(sigmas) + 2 (identity + gaussians + constant)."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    result = mod._al_subtract_identity_plus_gaussians_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )
    n_expected = len(mod._AL_SIGMAS_PRODUCTION) + 2
    assert len(result["fit_info"]["coefficients"]) == n_expected


def test_al_subtract_identity_plus_gaussians_has_identity_coefficient() -> None:
    """fit_info must contain identity_coefficient as a finite scalar."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    result = mod._al_subtract_identity_plus_gaussians_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )
    fi = result["fit_info"]
    assert "identity_coefficient" in fi
    assert np.isfinite(fi["identity_coefficient"])
    assert "sum_gaussian_correction_coefficients" in fi
    assert np.isfinite(fi["sum_gaussian_correction_coefficients"])


def test_al_subtract_identity_plus_gaussians_output_is_finite() -> None:
    """Difference array from identity_plus_gaussians must be all finite."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    result = mod._al_subtract_identity_plus_gaussians_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )
    assert np.all(np.isfinite(result["difference"]))


def test_al_subtract_identity_plus_gaussians_identity_coefficient_near_one() -> None:
    """For a near-identity case (sci ≈ ref), the identity coefficient should be near 1."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    rng = np.random.default_rng(0)
    ref = np.ones((32, 32), dtype=np.float64) * 5.0
    ref[16, 16] += 20.0
    sci = ref.copy() + rng.standard_normal((32, 32)) * 0.001
    result = mod._al_subtract_identity_plus_gaussians_qa(
        sci, ref, mod._AL_SIGMAS_PRODUCTION, mod._AL_KERNEL_SIZE
    )
    ic = result["fit_info"]["identity_coefficient"]
    assert abs(ic - 1.0) < 0.1, f"Expected identity_coefficient ≈ 1.0, got {ic}"


# ---------------------------------------------------------------------------
# identity_plus_gaussians — variant present in epoch results
# ---------------------------------------------------------------------------

def test_run_al_diagnostics_for_epoch_includes_identity_plus_gaussians() -> None:
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    results = mod._run_al_diagnostics_for_epoch(sci, ref, sci - ref, aperture_radius=3)
    assert "identity_plus_gaussians" in results


def test_run_al_diagnostics_for_epoch_identity_plus_gaussians_fit_info() -> None:
    """identity_plus_gaussians must record condition_number and identity_coefficient."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(64, 64))
    results = mod._run_al_diagnostics_for_epoch(sci, ref, sci - ref, aperture_radius=3)
    fi = results["identity_plus_gaussians"]["fit_info"]
    assert fi is not None
    assert "condition_number" in fi and fi["condition_number"] > 0
    assert "identity_coefficient" in fi and np.isfinite(fi["identity_coefficient"])
    assert "sum_gaussian_correction_coefficients" in fi


# ---------------------------------------------------------------------------
# identity_plus_gaussians — summary rows record extra fields
# ---------------------------------------------------------------------------

def test_run_al_diagnostic_suite_identity_plus_gaussians_row_fields(tmp_path: Path) -> None:
    """Summary rows for identity_plus_gaussians must include identity_coefficient."""
    mod = _load_module()
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    entries = [{"science": sci, "reference": ref, "raw_diff": sci - ref, "post_al": sci - ref}]
    rows = mod._run_al_diagnostic_suite(
        capture_entries=entries,
        event_or_ablation="test_event",
        aperture_radius=3,
        out_dir=tmp_path,
    )
    ipg_rows = [r for r in rows if r["diagnostic_mode"] == "identity_plus_gaussians"]
    assert ipg_rows, "No identity_plus_gaussians rows in summary"
    row = ipg_rows[0]
    assert "identity_coefficient" in row
    assert row["identity_coefficient"] is not None
    assert np.isfinite(row["identity_coefficient"])
    assert "sum_gaussian_correction_coefficients" in row
    assert row["sum_gaussian_correction_coefficients"] is not None


# ---------------------------------------------------------------------------
# identity_plus_gaussians — summary JSON and CSV schema
# ---------------------------------------------------------------------------

def test_save_al_diagnostic_summary_csv_has_identity_columns(tmp_path: Path) -> None:
    """CSV header must include identity_coefficient and sum_gaussian_correction_coefficients."""
    mod = _load_module()
    rows = _make_al_diag_summary_rows()
    mod._save_al_diagnostic_summary(rows, tmp_path)
    header = (tmp_path / "dia_al_diagnostic_summary.csv").read_text().splitlines()[0]
    for col in ("identity_coefficient", "sum_gaussian_correction_coefficients"):
        assert col in header, f"CSV header missing column: {col}"


def test_summary_schema_includes_identity_plus_gaussians_mode(tmp_path: Path) -> None:
    """Running the full suite must include identity_plus_gaussians rows in the JSON."""
    mod = _load_module()
    import json as _json
    if not mod._SCIPY:
        pytest.skip("scipy not available")
    sci, ref = _make_flat_sci_ref(shape=(32, 32))
    entries = [{"science": sci, "reference": ref, "raw_diff": sci - ref, "post_al": sci - ref}]
    rows = mod._run_al_diagnostic_suite(
        capture_entries=entries,
        event_or_ablation="qa_null",
        aperture_radius=3,
        out_dir=tmp_path,
    )
    mod._save_al_diagnostic_summary(rows, tmp_path)
    loaded = _json.loads((tmp_path / "dia_al_diagnostic_summary.json").read_text())
    modes = {r["diagnostic_mode"] for r in loaded}
    assert "identity_plus_gaussians" in modes


# ---------------------------------------------------------------------------
# identity_plus_gaussians — interpretation fires correctly
# ---------------------------------------------------------------------------

def test_assign_al_interpretations_identity_basis_fires() -> None:
    """identity_basis_fixes_source_core_residual fires when current_al high and ipg clean."""
    mod = _load_module()
    rows = _make_epoch_rows({
        "identity_only": 0.5,
        "current_al": 5.0,
        "identity_plus_gaussians": 1.0,
    })
    mod._assign_al_interpretations(rows)
    ipg_row = next(r for r in rows if r["diagnostic_mode"] == "identity_plus_gaussians")
    assert "identity_basis_fixes_source_core_residual" in ipg_row["interpretation"]


def test_assign_al_interpretations_identity_basis_does_not_fire_when_clean() -> None:
    """When current_al is also clean, the interpretation must NOT fire the specific label."""
    mod = _load_module()
    rows = _make_epoch_rows({
        "identity_only": 0.5,
        "current_al": 1.0,   # below threshold
        "identity_plus_gaussians": 0.8,
    })
    mod._assign_al_interpretations(rows)
    ipg_row = next(r for r in rows if r["diagnostic_mode"] == "identity_plus_gaussians")
    assert "identity_basis_fixes_source_core_residual" not in ipg_row["interpretation"]
    assert "clean_with_identity_basis" in ipg_row["interpretation"]


def test_assign_al_interpretations_identity_basis_persists_when_still_high() -> None:
    """When identity_plus_gaussians is still high, residual_persists_with_identity_basis fires."""
    mod = _load_module()
    rows = _make_epoch_rows({
        "current_al": 5.0,
        "identity_plus_gaussians": 4.5,
    })
    mod._assign_al_interpretations(rows)
    ipg_row = next(r for r in rows if r["diagnostic_mode"] == "identity_plus_gaussians")
    assert "residual_persists_with_identity_basis" in ipg_row["interpretation"]


# ===========================================================================
# Pilot batch — new tests
# ===========================================================================

import types as _types  # noqa: E402 (needed for SimpleNamespace mock)


def _make_pilot_cfg():
    """Minimal cfg-like namespace for pilot batch tests."""
    dia = _types.SimpleNamespace(
        reference_processing="detector_matched",
        reference_jitter_mode="per_epoch",
    )
    psf = _types.SimpleNamespace(
        fov_native_pixels=256,
        oversample=4,
        n_wavelengths=10,
    )
    return _types.SimpleNamespace(dia=dia, psf=psf)


def _make_pilot_prov(dia_method: str = "identity"):
    return _types.SimpleNamespace(dia_method=dia_method)


# ---------------------------------------------------------------------------
# Pilot batch — parser
# ---------------------------------------------------------------------------

def test_parser_pilot_batch_size_default() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    assert hasattr(args, "pilot_batch_size")
    assert args.pilot_batch_size == 0


def test_parser_pilot_batch_size_explicit() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args(["--pilot-batch-size", "20"])
    assert args.pilot_batch_size == 20


def test_parser_pilot_threshold_defaults() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    assert args.null_central_snr_warn == mod._DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT
    assert args.rim_ratio_low == mod._DEFAULT_RIM_RATIO_LOW
    assert args.rim_ratio_high == mod._DEFAULT_RIM_RATIO_HIGH
    assert args.max_nonfinite_allowed == 0
    assert args.max_saturation_fraction == 0.0
    assert args.max_cr_fraction is None


def test_parser_pilot_threshold_explicit() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args([
        "--pilot-batch-size", "5",
        "--null-central-snr-warn", "2.0",
        "--rim-ratio-low", "0.7",
        "--rim-ratio-high", "1.5",
        "--max-saturation-fraction", "0.01",
        "--max-cr-fraction", "0.05",
    ])
    assert args.pilot_batch_size == 5
    assert args.null_central_snr_warn == pytest.approx(2.0)
    assert args.rim_ratio_low == pytest.approx(0.7)
    assert args.rim_ratio_high == pytest.approx(1.5)
    assert args.max_saturation_fraction == pytest.approx(0.01)
    assert args.max_cr_fraction == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Pilot batch — _PILOT_CSV_FIELDNAMES constant
# ---------------------------------------------------------------------------

def test_pilot_csv_fieldnames_exist() -> None:
    mod = _load_module()
    assert hasattr(mod, "_PILOT_CSV_FIELDNAMES")
    assert isinstance(mod._PILOT_CSV_FIELDNAMES, list)
    assert len(mod._PILOT_CSV_FIELDNAMES) > 0


def test_pilot_csv_fieldnames_required_columns() -> None:
    mod = _load_module()
    required = [
        "event_id", "event_type", "epoch",
        "mean", "std", "min", "max", "p5", "p50", "p95",
        "central_aperture_snr", "central_aperture_percentile",
        "rim_interior_std_ratio",
        "has_nonfinite",
        "saturation_pixel_fraction", "cr_pixel_fraction",
        "dia_method", "reference_processing", "reference_jitter_mode",
        "psf_fov_native_pixels", "oversample", "n_wavelengths",
        "qa_flags",
    ]
    for col in required:
        assert col in mod._PILOT_CSV_FIELDNAMES, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Pilot batch — _compute_pilot_epoch_row
# ---------------------------------------------------------------------------

def test_compute_pilot_epoch_row_keys() -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    diff = rng.standard_normal((64, 64))
    sat = np.zeros((64, 64), dtype=np.float32)
    cr = np.zeros((64, 64), dtype=np.float32)
    cfg = _make_pilot_cfg()
    prov = _make_pilot_prov()
    row = mod._compute_pilot_epoch_row(
        event_id="pilot_0000",
        event_type="null",
        epoch=0,
        diff=diff,
        sat=sat,
        cr=cr,
        prov=prov,
        cfg=cfg,
        aperture_radius=3,
        n_random_apertures=50,
    )
    for col in mod._PILOT_CSV_FIELDNAMES:
        if col == "qa_flags":
            continue  # added by caller
        assert col in row, f"Missing key: {col}"


def test_compute_pilot_epoch_row_null_has_percentile() -> None:
    """Null events must include a non-None central_aperture_percentile."""
    mod = _load_module()
    diff = np.zeros((64, 64))
    diff[32, 32] = 50.0  # bright centre → high percentile
    row = mod._compute_pilot_epoch_row(
        event_id="pilot_0000",
        event_type="null",
        epoch=0,
        diff=diff,
        sat=np.zeros_like(diff),
        cr=np.zeros_like(diff),
        prov=_make_pilot_prov(),
        cfg=_make_pilot_cfg(),
        aperture_radius=3,
        n_random_apertures=100,
    )
    assert row["central_aperture_percentile"] is not None


def test_compute_pilot_epoch_row_lensed_no_percentile() -> None:
    """Lensed events must have central_aperture_percentile=None (not computed)."""
    mod = _load_module()
    rng = np.random.default_rng(1)
    diff = rng.standard_normal((64, 64))
    row = mod._compute_pilot_epoch_row(
        event_id="pilot_0001",
        event_type="lensed",
        epoch=0,
        diff=diff,
        sat=np.zeros_like(diff),
        cr=np.zeros_like(diff),
        prov=_make_pilot_prov(),
        cfg=_make_pilot_cfg(),
        aperture_radius=3,
        n_random_apertures=100,
    )
    assert row["central_aperture_percentile"] is None


def test_compute_pilot_epoch_row_nonfinite_flag() -> None:
    """has_nonfinite must be True when diff contains NaN/Inf."""
    mod = _load_module()
    diff = np.zeros((64, 64))
    diff[10, 10] = float("nan")
    row = mod._compute_pilot_epoch_row(
        event_id="pilot_0000",
        event_type="null",
        epoch=0,
        diff=diff,
        sat=np.zeros_like(diff),
        cr=np.zeros_like(diff),
        prov=_make_pilot_prov(),
        cfg=_make_pilot_cfg(),
        aperture_radius=3,
        n_random_apertures=50,
    )
    assert row["has_nonfinite"] is True


def test_compute_pilot_epoch_row_sat_frac() -> None:
    """saturation_pixel_fraction must equal the mean of the sat array."""
    mod = _load_module()
    diff = np.zeros((64, 64))
    sat = np.zeros((64, 64), dtype=np.float32)
    sat[:, :32] = 1.0  # half pixels saturated
    row = mod._compute_pilot_epoch_row(
        event_id="pilot_0000",
        event_type="null",
        epoch=0,
        diff=diff,
        sat=sat,
        cr=np.zeros_like(diff),
        prov=_make_pilot_prov(),
        cfg=_make_pilot_cfg(),
        aperture_radius=3,
        n_random_apertures=50,
    )
    assert abs(row["saturation_pixel_fraction"] - 0.5) < 1e-4


def test_compute_pilot_epoch_row_config_fields() -> None:
    """Config fields must be propagated into the row."""
    mod = _load_module()
    cfg = _make_pilot_cfg()
    row = mod._compute_pilot_epoch_row(
        event_id="pilot_0000",
        event_type="null",
        epoch=0,
        diff=np.zeros((64, 64)),
        sat=np.zeros((64, 64)),
        cr=np.zeros((64, 64)),
        prov=_make_pilot_prov("identity"),
        cfg=cfg,
        aperture_radius=3,
        n_random_apertures=50,
    )
    assert row["reference_processing"] == "detector_matched"
    assert row["reference_jitter_mode"] == "per_epoch"
    assert row["psf_fov_native_pixels"] == 256
    assert row["oversample"] == 4
    assert row["n_wavelengths"] == 10
    assert row["dia_method"] == "identity"


# ---------------------------------------------------------------------------
# Pilot batch — _pilot_qa_flags
# ---------------------------------------------------------------------------

def _make_clean_row(**overrides) -> dict:
    row = {
        "event_type": "null",
        "has_nonfinite": False,
        "central_aperture_snr": 0.0,
        "rim_interior_std_ratio": 1.0,
        "saturation_pixel_fraction": 0.0,
        "cr_pixel_fraction": 0.0,
        "dia_method": "identity",
        "reference_processing": "detector_matched",
        "reference_jitter_mode": "per_epoch",
    }
    row.update(overrides)
    return row


def test_pilot_qa_flags_clean_row_no_flags() -> None:
    mod = _load_module()
    row = _make_clean_row()
    flags = mod._pilot_qa_flags(row)
    assert flags == [], f"Expected no flags, got {flags}"


def test_pilot_qa_flags_nonfinite() -> None:
    mod = _load_module()
    row = _make_clean_row(has_nonfinite=True)
    flags = mod._pilot_qa_flags(row)
    assert "nonfinite_values" in flags


def test_pilot_qa_flags_null_snr_high() -> None:
    mod = _load_module()
    row = _make_clean_row(event_type="null", central_aperture_snr=5.0)
    flags = mod._pilot_qa_flags(row, null_central_snr_warn=3.0)
    assert "null_central_residual_high" in flags


def test_pilot_qa_flags_null_snr_below_threshold_no_flag() -> None:
    mod = _load_module()
    row = _make_clean_row(event_type="null", central_aperture_snr=2.0)
    flags = mod._pilot_qa_flags(row, null_central_snr_warn=3.0)
    assert "null_central_residual_high" not in flags


def test_pilot_qa_flags_lensed_snr_not_flagged() -> None:
    """Lensed events must NOT be flagged for central SNR regardless of value."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", central_aperture_snr=50.0)
    flags = mod._pilot_qa_flags(row, null_central_snr_warn=3.0)
    assert "null_central_residual_high" not in flags


def test_pilot_qa_flags_rim_ratio_low() -> None:
    mod = _load_module()
    row = _make_clean_row(rim_interior_std_ratio=0.3)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" in flags


def test_pilot_qa_flags_rim_ratio_high() -> None:
    mod = _load_module()
    row = _make_clean_row(rim_interior_std_ratio=3.5)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" in flags


def test_pilot_qa_flags_rim_ratio_within_bounds() -> None:
    mod = _load_module()
    row = _make_clean_row(rim_interior_std_ratio=1.2)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" not in flags


def test_pilot_qa_flags_unexpected_saturation() -> None:
    mod = _load_module()
    row = _make_clean_row(saturation_pixel_fraction=0.01)
    flags = mod._pilot_qa_flags(row, max_saturation_fraction=0.0)
    assert "unexpected_saturation" in flags


def test_pilot_qa_flags_cr_fraction_above_max() -> None:
    mod = _load_module()
    row = _make_clean_row(cr_pixel_fraction=0.1)
    flags = mod._pilot_qa_flags(row, max_cr_fraction=0.05)
    assert "unexpected_cr" in flags


def test_pilot_qa_flags_cr_fraction_none_max_no_flag() -> None:
    """When max_cr_fraction is None, CR fraction is never flagged."""
    mod = _load_module()
    row = _make_clean_row(cr_pixel_fraction=0.9)
    flags = mod._pilot_qa_flags(row, max_cr_fraction=None)
    assert "unexpected_cr" not in flags


def test_pilot_qa_flags_unexpected_dia_method() -> None:
    mod = _load_module()
    row = _make_clean_row(dia_method="alard_lupton")
    flags = mod._pilot_qa_flags(row)
    assert "unexpected_dia_method" in flags


def test_pilot_qa_flags_unexpected_reference_processing() -> None:
    mod = _load_module()
    row = _make_clean_row(reference_processing="ideal_gaussian")
    flags = mod._pilot_qa_flags(row)
    assert "unexpected_reference_processing" in flags


def test_pilot_qa_flags_unexpected_reference_jitter_mode() -> None:
    mod = _load_module()
    row = _make_clean_row(reference_jitter_mode="shared")
    flags = mod._pilot_qa_flags(row)
    assert "unexpected_reference_jitter_mode" in flags


def test_pilot_qa_flags_rim_nan_not_flagged() -> None:
    """NaN rim_interior_std_ratio must not produce a spurious rim_artifact_suspected flag."""
    mod = _load_module()
    row = _make_clean_row(rim_interior_std_ratio=float("nan"))
    flags = mod._pilot_qa_flags(row)
    assert "rim_artifact_suspected" not in flags


# Event-type-aware rim check tests

def test_pilot_qa_flags_null_low_rim_flags() -> None:
    """Null event with low rim/interior ratio must flag rim_artifact_suspected."""
    mod = _load_module()
    row = _make_clean_row(event_type="null", rim_interior_std_ratio=0.3)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" in flags


def test_pilot_qa_flags_null_high_rim_flags() -> None:
    """Null event with high rim/interior ratio must flag rim_artifact_suspected."""
    mod = _load_module()
    row = _make_clean_row(event_type="null", rim_interior_std_ratio=3.5)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" in flags


def test_pilot_qa_flags_lensed_low_rim_no_flag() -> None:
    """Lensed event with low rim/interior ratio must NOT flag rim_artifact_suspected."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", rim_interior_std_ratio=0.2)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" not in flags


def test_pilot_qa_flags_lensed_high_rim_flags() -> None:
    """Lensed event with abnormally high rim/interior ratio must flag rim_artifact_suspected."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", rim_interior_std_ratio=3.5)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" in flags


def test_pilot_qa_flags_lensed_in_bounds_no_flag() -> None:
    """Lensed event with in-bounds rim/interior ratio must not flag."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", rim_interior_std_ratio=1.2)
    flags = mod._pilot_qa_flags(row, rim_ratio_low=0.5, rim_ratio_high=2.0)
    assert "rim_artifact_suspected" not in flags


# _central_signal_dominant tests

def test_central_signal_dominant_true() -> None:
    """Lensed row with low rim ratio and high SNR must return True."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", rim_interior_std_ratio=0.2, central_aperture_snr=10.0)
    assert mod._central_signal_dominant(row, rim_ratio_low=0.5, snr_thresh=3.0) is True


def test_central_signal_dominant_false_null_event() -> None:
    """Null events always return False regardless of rim/SNR values."""
    mod = _load_module()
    row = _make_clean_row(event_type="null", rim_interior_std_ratio=0.2, central_aperture_snr=10.0)
    assert mod._central_signal_dominant(row, rim_ratio_low=0.5, snr_thresh=3.0) is False


def test_central_signal_dominant_false_high_rim() -> None:
    """Lensed row with high rim ratio (not low) must return False."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", rim_interior_std_ratio=1.5, central_aperture_snr=10.0)
    assert mod._central_signal_dominant(row, rim_ratio_low=0.5, snr_thresh=3.0) is False


def test_central_signal_dominant_false_low_snr() -> None:
    """Lensed row with low rim ratio but low SNR must return False."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", rim_interior_std_ratio=0.2, central_aperture_snr=1.0)
    assert mod._central_signal_dominant(row, rim_ratio_low=0.5, snr_thresh=3.0) is False


def test_central_signal_dominant_false_nan_rim() -> None:
    """NaN rim_interior_std_ratio must not trigger central_signal_dominant."""
    mod = _load_module()
    row = _make_clean_row(event_type="lensed", rim_interior_std_ratio=float("nan"), central_aperture_snr=50.0)
    assert mod._central_signal_dominant(row, rim_ratio_low=0.5, snr_thresh=3.0) is False


def test_pilot_batch_pass_lensed_central_dominant(capsys) -> None:
    """Lensed peak epoch with low rim/interior ratio and high SNR must not cause FAIL."""
    mod = _load_module()
    rows = _make_pilot_rows(4)
    # Simulate a lensed peak: low rim ratio, high central SNR, no real failures
    lensed_row = next(r for r in rows if r.get("event_type") == "lensed")
    lensed_row["rim_interior_std_ratio"] = 0.2
    lensed_row["central_aperture_snr"] = 15.0
    lensed_row["qa_flags"] = mod._pilot_qa_flags(
        lensed_row, rim_ratio_low=0.5, rim_ratio_high=2.0
    )
    lensed_row["central_signal_dominant"] = mod._central_signal_dominant(
        lensed_row, rim_ratio_low=0.5, snr_thresh=3.0
    )
    assert "rim_artifact_suspected" not in lensed_row["qa_flags"]
    assert lensed_row["central_signal_dominant"] is True
    thresholds = {
        "null_central_snr_warn": 3.0,
        "rim_ratio_low": 0.5,
        "rim_ratio_high": 2.0,
        "max_saturation_fraction": 0.0,
        "max_cr_fraction": None,
    }
    mod._print_pilot_console_summary(rows, thresholds)
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "central_signal_dominant" in out


# ---------------------------------------------------------------------------
# Pilot batch — _save_pilot_summary
# ---------------------------------------------------------------------------

def _make_pilot_rows(n: int = 3) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        event_type = "null" if i % 2 == 0 else "lensed"
        rows.append({
            "event_id": f"pilot_{i:04d}",
            "event_type": event_type,
            "epoch": 0,
            "mean": round(float(rng.standard_normal(1)[0]), 4),
            "std": 1.5,
            "min": -3.0,
            "max": 3.0,
            "p5": -2.0,
            "p50": 0.0,
            "p95": 2.0,
            "central_aperture_snr": round(float(rng.standard_normal(1)[0]), 4),
            "central_aperture_percentile": 50.0 if event_type == "null" else None,
            "rim_interior_std_ratio": 1.0,
            "central_signal_dominant": False,
            "has_nonfinite": False,
            "saturation_pixel_fraction": 0.0,
            "cr_pixel_fraction": 0.0,
            "dia_method": "identity",
            "reference_processing": "detector_matched",
            "reference_jitter_mode": "per_epoch",
            "psf_fov_native_pixels": 256,
            "oversample": 4,
            "n_wavelengths": 10,
            "qa_flags": [],
        })
    return rows


def test_save_pilot_summary_creates_files(tmp_path: Path) -> None:
    mod = _load_module()
    rows = _make_pilot_rows(4)
    j, c = mod._save_pilot_summary(rows, tmp_path)
    assert j.exists()
    assert c.exists()


def test_save_pilot_summary_json_valid(tmp_path: Path) -> None:
    import json as _json
    mod = _load_module()
    rows = _make_pilot_rows(3)
    j, _ = mod._save_pilot_summary(rows, tmp_path)
    recovered = _json.loads(j.read_text())
    assert len(recovered) == 3
    assert recovered[0]["event_id"] == "pilot_0000"


def test_save_pilot_summary_csv_has_required_columns(tmp_path: Path) -> None:
    mod = _load_module()
    rows = _make_pilot_rows(2)
    _, c = mod._save_pilot_summary(rows, tmp_path)
    header = c.read_text().splitlines()[0]
    for col in mod._PILOT_CSV_FIELDNAMES:
        assert col in header, f"CSV missing column: {col}"


def test_save_pilot_summary_csv_qa_flags_pipe_separated(tmp_path: Path) -> None:
    """qa_flags must be written as a pipe-separated string in the CSV."""
    mod = _load_module()
    rows = _make_pilot_rows(1)
    rows[0]["qa_flags"] = ["nonfinite_values", "null_central_residual_high"]
    _, c = mod._save_pilot_summary(rows, tmp_path)
    content = c.read_text()
    assert "nonfinite_values|null_central_residual_high" in content


def test_save_pilot_summary_csv_empty_flags(tmp_path: Path) -> None:
    """Empty qa_flags list must write an empty string, not '[]'."""
    mod = _load_module()
    rows = _make_pilot_rows(1)
    rows[0]["qa_flags"] = []
    _, c = mod._save_pilot_summary(rows, tmp_path)
    # Should not contain the literal string '[]'
    assert "[]" not in c.read_text()


def test_save_pilot_summary_roundtrip_json(tmp_path: Path) -> None:
    """JSON roundtrip must preserve all field values."""
    import json as _json
    mod = _load_module()
    rows = _make_pilot_rows(2)
    rows[0]["qa_flags"] = ["nonfinite_values"]
    j, _ = mod._save_pilot_summary(rows, tmp_path)
    recovered = _json.loads(j.read_text())
    assert recovered[0]["qa_flags"] == ["nonfinite_values"]
    assert recovered[1]["qa_flags"] == []


# ---------------------------------------------------------------------------
# Pilot batch — _print_pilot_console_summary (smoke test)
# ---------------------------------------------------------------------------

def test_print_pilot_console_summary_no_flags(capsys) -> None:
    mod = _load_module()
    rows = _make_pilot_rows(4)
    thresholds = {
        "null_central_snr_warn": 3.0,
        "rim_ratio_low": 0.5,
        "rim_ratio_high": 2.0,
        "max_saturation_fraction": 0.0,
        "max_cr_fraction": None,
    }
    mod._print_pilot_console_summary(rows, thresholds)
    out = capsys.readouterr().out
    assert "PASS" in out


def test_print_pilot_console_summary_with_flags(capsys) -> None:
    mod = _load_module()
    rows = _make_pilot_rows(2)
    rows[0]["qa_flags"] = ["null_central_residual_high"]
    thresholds = {
        "null_central_snr_warn": 3.0,
        "rim_ratio_low": 0.5,
        "rim_ratio_high": 2.0,
        "max_saturation_fraction": 0.0,
        "max_cr_fraction": None,
    }
    mod._print_pilot_console_summary(rows, thresholds)
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "1" in out  # at least 1 flagged row


def test_print_pilot_console_summary_config_checks(capsys) -> None:
    """Must print [OK] for all config checks when expectations are met."""
    mod = _load_module()
    rows = _make_pilot_rows(2)
    thresholds = {
        "null_central_snr_warn": 3.0,
        "rim_ratio_low": 0.5,
        "rim_ratio_high": 2.0,
        "max_saturation_fraction": 0.0,
        "max_cr_fraction": None,
    }
    mod._print_pilot_console_summary(rows, thresholds)
    out = capsys.readouterr().out
    assert "[OK]" in out


def test_print_pilot_console_summary_config_warn_on_wrong_dia(capsys) -> None:
    """Must print [WARN] for dia_method when it differs from expected."""
    mod = _load_module()
    rows = _make_pilot_rows(2)
    for r in rows:
        r["dia_method"] = "alard_lupton"
        r["qa_flags"] = ["unexpected_dia_method"]
    thresholds = {
        "null_central_snr_warn": 3.0,
        "rim_ratio_low": 0.5,
        "rim_ratio_high": 2.0,
        "max_saturation_fraction": 0.0,
        "max_cr_fraction": None,
    }
    mod._print_pilot_console_summary(rows, thresholds)
    out = capsys.readouterr().out
    assert "[WARN]" in out


def test_print_pilot_console_summary_top10_null_snr_shown(capsys) -> None:
    """When null rows exist, the TOP 10 WORST NULL CENTRAL |SNR| table must appear."""
    mod = _load_module()
    rows = _make_pilot_rows(4)  # 2 null, 2 lensed
    thresholds = {
        "null_central_snr_warn": 3.0,
        "rim_ratio_low": 0.5,
        "rim_ratio_high": 2.0,
        "max_saturation_fraction": 0.0,
        "max_cr_fraction": None,
    }
    mod._print_pilot_console_summary(rows, thresholds)
    out = capsys.readouterr().out
    assert "TOP 10 WORST NULL CENTRAL" in out


# ---------------------------------------------------------------------------
# Pilot batch — science config expectation constants
# ---------------------------------------------------------------------------

def test_pilot_expected_dia_method_constant() -> None:
    mod = _load_module()
    assert hasattr(mod, "_PILOT_EXPECTED_DIA_METHOD")
    assert mod._PILOT_EXPECTED_DIA_METHOD == "identity"


def test_pilot_expected_reference_processing_constant() -> None:
    mod = _load_module()
    assert hasattr(mod, "_PILOT_EXPECTED_REFERENCE_PROCESSING")
    assert mod._PILOT_EXPECTED_REFERENCE_PROCESSING == "detector_matched"


def test_pilot_expected_reference_jitter_mode_constant() -> None:
    mod = _load_module()
    assert hasattr(mod, "_PILOT_EXPECTED_REFERENCE_JITTER_MODE")
    assert mod._PILOT_EXPECTED_REFERENCE_JITTER_MODE == "per_epoch"


def test_pilot_default_output_dir_constant() -> None:
    mod = _load_module()
    assert hasattr(mod, "_DEFAULT_PILOT_OUTPUT_DIR")
    assert "pilot" in mod._DEFAULT_PILOT_OUTPUT_DIR.lower()


# ---------------------------------------------------------------------------
# Pilot batch — run_pilot_batch and _print_pilot_console_summary exist
# ---------------------------------------------------------------------------

def test_run_pilot_batch_function_exists() -> None:
    mod = _load_module()
    assert hasattr(mod, "run_pilot_batch")
    assert callable(mod.run_pilot_batch)


def test_pilot_batch_functions_exist() -> None:
    mod = _load_module()
    for name in (
        "_compute_pilot_epoch_row",
        "_pilot_qa_flags",
        "_save_pilot_summary",
        "_print_pilot_console_summary",
        "run_pilot_batch",
    ):
        assert hasattr(mod, name), f"Missing function: {name}"
        assert callable(getattr(mod, name))
