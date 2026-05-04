#!/usr/bin/env python3
"""
scripts/compare_science_profiles.py

A/B science-profile comparison for SMIG Phase 2.

Runs the pilot-batch QA under two different configs (baseline vs candidate)
using identical seeds and event definitions, then computes cross-config
metrics covering correctness, PSF shape, image-level differences, and
runtime speedup.

Typical use: compare oversample=4 (production) vs oversample=2 (experimental).

Outputs
-------
  <output-dir>/
    baseline/                   pilot batch for baseline config
    candidate/                  pilot batch for candidate config
    comparison_summary.json
    comparison_summary.csv
    comparison_report.txt

Usage
-----
  python scripts/compare_science_profiles.py \\
      --baseline-config smig/config/simulation_science.yaml \\
      --candidate-config smig/config/simulation_science_oversample2.yaml \\
      --output-dir outputs/oversample2_ablation \\
      --pilot-batch-size 20 \\
      --n-epochs 3 \\
      --master-seed 12345

Exit codes
----------
  0 — comparison complete (PASS or WARN; not a hard regression)
  1 — FAIL verdict or fatal error
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from smig.config.utils import get_simulation_config_sha256, load_simulation_config

# ── optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_BASELINE_CONFIG = "smig/config/simulation_science.yaml"
_DEFAULT_CANDIDATE_CONFIG = "smig/config/simulation_science_oversample2.yaml"
_DEFAULT_OUTPUT_DIR = "outputs/oversample2_ablation"
_DEFAULT_PILOT_BATCH_SIZE = 20
_DEFAULT_N_EPOCHS = 3
_DEFAULT_MASTER_SEED = 12345
_DEFAULT_APERTURE_RADIUS = 3
_DEFAULT_RANDOM_APERTURES = 500

# Tolerances for PASS/FAIL verdict
_LENSED_SNR_RATIO_TOLERANCE = 0.10   # 10% — lensed central flux recovery
_NULL_SNR_MAX_THRESHOLD = 5.0        # Max |null central SNR| before FAIL
_NULL_SNR_REGRESSION_FACTOR = 2.0    # WARN if candidate null SNR > 2x baseline
_RIM_RATIO_LOW = 0.5
_RIM_RATIO_HIGH = 2.0

# Comparison summary CSV fieldnames
_COMPARISON_CSV_FIELDNAMES = [
    "metric",
    "baseline_value",
    "candidate_value",
    "ratio_or_diff",
    "unit",
    "note",
]


# ── QA module loader ──────────────────────────────────────────────────────────

def _load_qa_module():
    """Dynamically load generate_phase2_qa_stamps without executing main()."""
    qa_path = Path(__file__).parent / "generate_phase2_qa_stamps.py"
    if not qa_path.exists():
        sys.exit(f"ERROR: cannot find QA module at {qa_path}")
    spec = importlib.util.spec_from_file_location(
        "generate_phase2_qa_stamps", qa_path
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── argument parsing ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A/B comparison of two SMIG simulation profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--baseline-config",
        default=_DEFAULT_BASELINE_CONFIG,
        help="Path to baseline (production) YAML config.",
    )
    p.add_argument(
        "--candidate-config",
        default=_DEFAULT_CANDIDATE_CONFIG,
        help="Path to candidate (experimental) YAML config.",
    )
    p.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=(
            "Root output directory. baseline/ and candidate/ subdirs are "
            "created here for the pilot batch outputs."
        ),
    )
    p.add_argument(
        "--pilot-batch-size",
        type=int,
        default=_DEFAULT_PILOT_BATCH_SIZE,
        metavar="N",
        help="Number of deterministic events per profile run.",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=_DEFAULT_N_EPOCHS,
        help="Number of science epochs per event.",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=_DEFAULT_MASTER_SEED,
        help="Master RNG seed (identical for both runs).",
    )
    p.add_argument(
        "--aperture-radius",
        type=int,
        default=_DEFAULT_APERTURE_RADIUS,
        help="Central aperture radius in pixels.",
    )
    p.add_argument(
        "--random-apertures",
        type=int,
        default=_DEFAULT_RANDOM_APERTURES,
        help="Number of random off-centre apertures for null-event percentile.",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip re-running the baseline profile (use existing outputs in output-dir/baseline/).",
    )
    p.add_argument(
        "--skip-candidate",
        action="store_true",
        default=False,
        help="Skip re-running the candidate profile (use existing outputs in output-dir/candidate/).",
    )
    p.add_argument(
        "--skip-psf-comparison",
        action="store_true",
        default=False,
        help="Skip PSF shape comparison (faster; PSF metrics will be absent from summary).",
    )
    return p


# ── profile runner ────────────────────────────────────────────────────────────

def _run_one_profile(
    qa_mod: Any,
    config_path: Path,
    output_dir: Path,
    batch_size: int,
    n_epochs: int,
    master_seed: int,
    aperture_radius: int,
    n_random_apertures: int,
) -> float:
    """Run run_pilot_batch for one config and return wall-clock seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    qa_mod.run_pilot_batch(
        config_path=config_path,
        output_dir=output_dir,
        batch_size=batch_size,
        n_epochs=n_epochs,
        master_seed=master_seed,
        aperture_radius=aperture_radius,
        n_random_apertures=n_random_apertures,
    )
    return time.perf_counter() - t0


def _load_pilot_rows(output_dir: Path) -> list[dict]:
    """Load pilot_summary.json from output_dir."""
    json_path = output_dir / "pilot_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"pilot_summary.json not found at {output_dir}. "
            "Run without --skip-baseline / --skip-candidate first."
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


# ── metrics aggregation ───────────────────────────────────────────────────────

def _null_metrics(rows: list[dict]) -> dict[str, Any]:
    """Aggregate correctness metrics for null events."""
    null_rows = [r for r in rows if r.get("event_type") == "null"]
    if not null_rows:
        return {"n_rows": 0}

    snrs = [
        float(r["central_aperture_snr"])
        for r in null_rows
        if r.get("central_aperture_snr") is not None
    ]
    pcts = [
        float(r["central_aperture_percentile"])
        for r in null_rows
        if r.get("central_aperture_percentile") is not None
    ]
    ratios = [
        float(r["rim_interior_std_ratio"])
        for r in null_rows
        if r.get("rim_interior_std_ratio") is not None
        and np.isfinite(float(r["rim_interior_std_ratio"]))
    ]
    nonfinite = [
        bool(r["has_nonfinite"])
        for r in null_rows
        if "has_nonfinite" in r
    ]
    sat = [
        float(r["saturation_pixel_fraction"])
        for r in null_rows
        if r.get("saturation_pixel_fraction") is not None
    ]
    cr = [
        float(r["cr_pixel_fraction"])
        for r in null_rows
        if r.get("cr_pixel_fraction") is not None
    ]
    flagged = [bool(r.get("qa_flags")) for r in null_rows]

    def _fm(arr: list[float]) -> float | None:
        return round(float(np.mean(arr)), 6) if arr else None

    def _fp(arr: list[float], q: float) -> float | None:
        return round(float(np.percentile(arr, q)), 6) if arr else None

    return {
        "n_rows": len(null_rows),
        "central_snr_mean": _fm(snrs),
        "central_snr_std": round(float(np.std(snrs)), 6) if snrs else None,
        "central_snr_max_abs": round(float(np.max(np.abs(snrs))), 6) if snrs else None,
        "central_pct_mean": _fm(pcts),
        "central_pct_p50": _fp(pcts, 50),
        "central_pct_p95": _fp(pcts, 95),
        "rim_ratio_mean": _fm(ratios),
        "rim_ratio_max": round(float(np.max(ratios)), 6) if ratios else None,
        "nonfinite_frac": _fm([float(v) for v in nonfinite]),
        "saturation_frac_mean": _fm(sat),
        "cr_frac_mean": _fm(cr),
        "flagged_frac": _fm([float(v) for v in flagged]),
    }


def _lensed_metrics(rows: list[dict]) -> dict[str, Any]:
    """Aggregate correctness metrics for lensed events."""
    lensed_rows = [r for r in rows if r.get("event_type") == "lensed"]
    if not lensed_rows:
        return {"n_rows": 0}

    snrs = [
        float(r["central_aperture_snr"])
        for r in lensed_rows
        if r.get("central_aperture_snr") is not None
    ]
    ratios = [
        float(r["rim_interior_std_ratio"])
        for r in lensed_rows
        if r.get("rim_interior_std_ratio") is not None
        and np.isfinite(float(r["rim_interior_std_ratio"]))
    ]
    dominant = [
        bool(r["central_signal_dominant"])
        for r in lensed_rows
        if "central_signal_dominant" in r
    ]
    nonfinite = [bool(r["has_nonfinite"]) for r in lensed_rows if "has_nonfinite" in r]
    sat = [
        float(r["saturation_pixel_fraction"])
        for r in lensed_rows
        if r.get("saturation_pixel_fraction") is not None
    ]
    flagged = [bool(r.get("qa_flags")) for r in lensed_rows]

    def _fm(arr: list[float]) -> float | None:
        return round(float(np.mean(arr)), 6) if arr else None

    return {
        "n_rows": len(lensed_rows),
        "central_snr_mean": _fm(snrs),
        "central_snr_max": round(float(np.max(snrs)), 6) if snrs else None,
        "central_snr_std": round(float(np.std(snrs)), 6) if snrs else None,
        "rim_ratio_mean": _fm(ratios),
        "central_signal_dominant_frac": _fm([float(v) for v in dominant]),
        "nonfinite_frac": _fm([float(v) for v in nonfinite]),
        "saturation_frac_mean": _fm(sat),
        "flagged_frac": _fm([float(v) for v in flagged]),
    }


# ── stamp-level comparison ────────────────────────────────────────────────────

def _compute_stamp_differences(
    baseline_dir: Path,
    candidate_dir: Path,
    batch_size: int,
    aperture_radius: int,
) -> dict[str, Any]:
    """Compare matched diff_stamps.npy between baseline and candidate."""
    null_rms_diffs: list[float] = []
    lensed_rms_diffs: list[float] = []
    max_abs_diffs: list[float] = []
    lensed_central_ratios: list[float] = []
    n_matched = 0

    for i in range(batch_size):
        event_id = f"pilot_{i:04d}"
        is_null = (i % 2 == 0)

        b_path = baseline_dir / event_id / "diff_stamps.npy"
        c_path = candidate_dir / event_id / "diff_stamps.npy"

        if not b_path.exists() or not c_path.exists():
            continue

        b_stamps = np.load(str(b_path))   # (n_epochs, h, w)
        c_stamps = np.load(str(c_path))

        if b_stamps.shape != c_stamps.shape:
            continue

        n_matched += 1
        diff = c_stamps.astype(np.float64) - b_stamps.astype(np.float64)
        max_abs_diffs.append(float(np.max(np.abs(diff))))

        if is_null:
            null_rms_diffs.append(float(np.sqrt(np.mean(diff ** 2))))
        else:
            lensed_rms_diffs.append(float(np.sqrt(np.mean(diff ** 2))))
            # Compare central aperture mean at lensed peak epoch (middle)
            peak_ep = b_stamps.shape[0] // 2
            b_peak = b_stamps[peak_ep].astype(np.float64)
            c_peak = c_stamps[peak_ep].astype(np.float64)
            h, w = b_peak.shape
            cy, cx = h // 2, w // 2
            yy, xx = np.mgrid[0:h, 0:w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= aperture_radius ** 2
            b_val = float(np.mean(b_peak[mask]))
            c_val = float(np.mean(c_peak[mask]))
            if abs(b_val) > 1.0:   # at least 1 e- to avoid division noise
                lensed_central_ratios.append(c_val / b_val)

    def _fm(arr: list[float]) -> float | None:
        return round(float(np.mean(arr)), 6) if arr else None

    def _fmax(arr: list[float]) -> float | None:
        return round(float(np.max(arr)), 6) if arr else None

    return {
        "n_events_attempted": batch_size,
        "n_events_matched": n_matched,
        "null_stamp_rms_diff_mean": _fm(null_rms_diffs),
        "null_stamp_rms_diff_max": _fmax(null_rms_diffs),
        "lensed_stamp_rms_diff_mean": _fm(lensed_rms_diffs),
        "lensed_stamp_rms_diff_max": _fmax(lensed_rms_diffs),
        "max_abs_pixel_diff": _fmax(max_abs_diffs),
        "lensed_central_snr_ratio_mean": _fm(lensed_central_ratios),
        "lensed_central_snr_ratio_std": (
            round(float(np.std(lensed_central_ratios)), 6) if lensed_central_ratios else None
        ),
    }


# ── PSF shape comparison ──────────────────────────────────────────────────────

def _gaussian_psf_on_grid(n_pix: int, sigma_pix: float) -> np.ndarray:
    """2D isotropic Gaussian, normalized to sum=1, on an n_pix×n_pix grid."""
    center = (n_pix - 1) / 2.0
    y, x = np.mgrid[0:n_pix, 0:n_pix]
    r2 = (x.astype(np.float64) - center) ** 2 + (y.astype(np.float64) - center) ** 2
    psf = np.exp(-0.5 * r2 / sigma_pix ** 2)
    psf /= psf.sum()
    return psf


def _bin_to_native(psf_os: np.ndarray, oversample: int) -> np.ndarray:
    """Block-sum an oversampled PSF array to native pixel scale."""
    h, w = psf_os.shape
    hn, wn = h // oversample, w // oversample
    trimmed = psf_os[: hn * oversample, : wn * oversample]
    return trimmed.reshape(hn, oversample, wn, oversample).sum(axis=(1, 3))


def _encircled_energy(
    psf_native: np.ndarray,
    max_radius_px: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encircled energy curve at native pixel scale.

    Returns (radii_px, ee_fraction) where ee is normalized so ee[max_r] ≤ 1.
    """
    h, w = psf_native.shape
    cy, cx = h // 2, w // 2
    if max_radius_px is None:
        max_radius_px = min(cy, cx)
    radii = np.arange(0, max_radius_px + 1, dtype=float)
    yy, xx = np.mgrid[0:h, 0:w]
    r_map = np.sqrt((xx.astype(float) - cx) ** 2 + (yy.astype(float) - cy) ** 2)
    total = float(psf_native.sum())
    if total <= 0:
        return radii, np.zeros_like(radii)
    ee = np.array([float(psf_native[r_map <= r].sum()) / total for r in radii])
    return radii, ee


def _second_moment(psf_native: np.ndarray) -> float:
    """RMS second moment (effective Gaussian sigma) in native pixels."""
    h, w = psf_native.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    total = float(psf_native.sum())
    if total <= 0:
        return float("nan")
    r2 = (xx.astype(float) - cx) ** 2 + (yy.astype(float) - cy) ** 2
    return float(np.sqrt(float(np.sum(psf_native * r2)) / total))


def _compute_psf_comparison(
    baseline_cfg: Any,
    candidate_cfg: Any,
    fov_native: int = 64,
) -> dict[str, Any]:
    """Compare PSF shapes for baseline and candidate configs at native pixel scale.

    Tries STPSFProvider first (requires GalSim); falls back to an analytic
    Gaussian on the oversampled grid if GalSim is unavailable.  Both paths
    bin the result to native pixel scale before computing comparison metrics.
    """
    b_os = baseline_cfg.psf.oversample
    c_os = candidate_cfg.psf.oversample

    b_psf_os: np.ndarray | None = None
    c_psf_os: np.ndarray | None = None
    method = "analytic_gaussian"

    try:
        from smig.optics.psf import STPSFProvider
        b_provider = STPSFProvider(baseline_cfg.psf)
        c_provider = STPSFProvider(candidate_cfg.psf)
        center_wl = float(np.mean(baseline_cfg.psf.wavelength_range_um))
        b_psf_os = b_provider.get_psf_at_wavelength(1, (0.5, 0.5), center_wl)
        c_psf_os = c_provider.get_psf_at_wavelength(1, (0.5, 0.5), center_wl)
        method = "STPSFProvider"
    except Exception:
        pass

    if b_psf_os is None:
        # Analytic Gaussian at Roman diffraction limit: FWHM ≈ λ/D.
        # At 1.5 μm, λ/D ~ 1.5e-6/2.4 rad = 0.129 arcsec ≈ 1.17 native px.
        # sigma = FWHM / 2.355 ≈ 0.50 native pixels.
        sigma_native = 0.50
        b_n = fov_native * b_os
        c_n = fov_native * c_os
        b_psf_os = _gaussian_psf_on_grid(b_n, sigma_native * b_os)
        c_psf_os = _gaussian_psf_on_grid(c_n, sigma_native * c_os)

    # Bin to native pixel scale and re-normalize.
    b_native = _bin_to_native(b_psf_os, b_os)
    c_native = _bin_to_native(c_psf_os, c_os)
    b_total = float(b_native.sum())
    c_total = float(c_native.sum())
    if b_total > 0:
        b_native = b_native / b_total
    if c_total > 0:
        c_native = c_native / c_total

    # Crop to the same shape (take centre of the larger array).
    min_h = min(b_native.shape[0], c_native.shape[0])
    min_w = min(b_native.shape[1], c_native.shape[1])
    bh, bw = b_native.shape
    ch, cw = c_native.shape
    b_crop = b_native[
        (bh - min_h) // 2 : (bh + min_h) // 2,
        (bw - min_w) // 2 : (bw + min_w) // 2,
    ]
    c_crop = c_native[
        (ch - min_h) // 2 : (ch + min_h) // 2,
        (cw - min_w) // 2 : (cw + min_w) // 2,
    ]
    # Renormalize after crop.
    b_sum = float(b_crop.sum())
    c_sum = float(c_crop.sum())
    if b_sum > 0:
        b_crop = b_crop / b_sum
    if c_sum > 0:
        c_crop = c_crop / c_sum

    # Peak value ratio (candidate / baseline).
    b_peak = float(b_crop.max())
    c_peak = float(c_crop.max())
    peak_ratio = c_peak / b_peak if b_peak > 0 else float("nan")

    # Second moment.
    b_sigma = _second_moment(b_crop)
    c_sigma = _second_moment(c_crop)
    width_ratio = c_sigma / b_sigma if (np.isfinite(b_sigma) and b_sigma > 0) else float("nan")

    # Encircled energy curves.
    max_r = max(0, min(min_h, min_w) // 2 - 1)
    b_radii, b_ee = _encircled_energy(b_crop, max_radius_px=max_r)
    _, c_ee = _encircled_energy(c_crop, max_radius_px=max_r)
    ee_diff_max = float(np.max(np.abs(c_ee - b_ee))) if len(b_ee) > 0 else float("nan")

    # EE at discrete radii of interest (native pixels).
    b_ee_at: dict[str, float] = {}
    c_ee_at: dict[str, float] = {}
    for r in [1, 2, 3, 5, 8]:
        if r <= max_r:
            b_ee_at[f"ee_r{r}px"] = round(float(b_ee[r]), 6)
            c_ee_at[f"ee_r{r}px"] = round(float(c_ee[r]), 6)

    # Rim energy (outer 5-px border at native scale).
    rim_w = 5
    rim_b = np.ones(b_crop.shape, dtype=bool)
    rim_c = np.ones(c_crop.shape, dtype=bool)
    if b_crop.shape[0] > 2 * rim_w:
        rim_b[rim_w:-rim_w, rim_w:-rim_w] = False
    if c_crop.shape[0] > 2 * rim_w:
        rim_c[rim_w:-rim_w, rim_w:-rim_w] = False
    b_rim = float(b_crop[rim_b].sum())
    c_rim = float(c_crop[rim_c].sum())
    rim_ratio = c_rim / b_rim if b_rim > 0 else float("nan")

    return {
        "method": method,
        "baseline_oversample": b_os,
        "candidate_oversample": c_os,
        "peak_ratio_c_over_b": round(peak_ratio, 6) if np.isfinite(peak_ratio) else None,
        "width_ratio_c_over_b": round(width_ratio, 6) if np.isfinite(width_ratio) else None,
        "ee_diff_max": round(ee_diff_max, 6) if np.isfinite(ee_diff_max) else None,
        "rim_energy_ratio_c_over_b": round(rim_ratio, 6) if np.isfinite(rim_ratio) else None,
        "baseline_ee_at_radii": b_ee_at,
        "candidate_ee_at_radii": c_ee_at,
    }


# ── pass/fail verdict ─────────────────────────────────────────────────────────

def _pass_fail_verdict(
    b_null: dict,
    c_null: dict,
    b_lensed: dict,
    c_lensed: dict,
    stamp_diff: dict,
    psf_cmp: dict | None,
) -> dict[str, Any]:
    """Determine PASS / WARN / FAIL from comparison metrics.

    Criteria (from task specification):
      FAIL  — null central residuals reappear, lensed flux shifts > tolerance,
               rim artifacts increase significantly, nonfinite values in candidate.
      WARN  — metrics degrade but stay within secondary bounds.
      PASS  — all metrics comparable to baseline.
    """
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    pass_reasons: list[str] = []

    # 1. Null central SNR absolute threshold
    c_max_snr = c_null.get("central_snr_max_abs")
    b_max_snr = b_null.get("central_snr_max_abs")
    if c_max_snr is not None and c_max_snr > _NULL_SNR_MAX_THRESHOLD:
        fail_reasons.append(
            f"Candidate null max|SNR|={c_max_snr:.2f} exceeds threshold "
            f"{_NULL_SNR_MAX_THRESHOLD:.1f}"
        )
    else:
        _snr_str = f"{c_max_snr:.2f}" if c_max_snr is not None else "N/A"
        pass_reasons.append(
            f"Null central SNR within threshold (candidate max|SNR|={_snr_str})"
        )

    # 2. Null central SNR regression vs baseline
    if c_max_snr is not None and b_max_snr is not None and b_max_snr > 1e-6:
        snr_factor = c_max_snr / b_max_snr
        if snr_factor > _NULL_SNR_REGRESSION_FACTOR:
            warn_reasons.append(
                f"Candidate null max|SNR| is {snr_factor:.1f}x baseline "
                f"({c_max_snr:.2f} vs {b_max_snr:.2f})"
            )

    # 3. Nonfinite pixel check
    c_null_nf = c_null.get("nonfinite_frac") or 0.0
    c_lens_nf = c_lensed.get("nonfinite_frac") or 0.0
    if c_null_nf > 0.0 or c_lens_nf > 0.0:
        fail_reasons.append(
            f"Candidate produced nonfinite pixels (null_frac={c_null_nf:.4f}, "
            f"lensed_frac={c_lens_nf:.4f})"
        )
    else:
        pass_reasons.append("No nonfinite values in candidate stamps")

    # 4. Rim/interior artifact guard
    c_rim = c_null.get("rim_ratio_mean")
    if c_rim is not None:
        if c_rim < _RIM_RATIO_LOW or c_rim > _RIM_RATIO_HIGH:
            fail_reasons.append(
                f"Candidate null rim/interior ratio={c_rim:.3f} outside "
                f"[{_RIM_RATIO_LOW}, {_RIM_RATIO_HIGH}]"
            )
        else:
            pass_reasons.append(
                f"Rim/interior artifact guard within range (mean={c_rim:.3f})"
            )

    # 5. Lensed central flux recovery ratio
    ratio = stamp_diff.get("lensed_central_snr_ratio_mean")
    if ratio is not None:
        dev = abs(ratio - 1.0)
        if dev > _LENSED_SNR_RATIO_TOLERANCE:
            fail_reasons.append(
                f"Lensed central flux ratio={ratio:.4f}, "
                f"deviation {dev:.1%} exceeds tolerance {_LENSED_SNR_RATIO_TOLERANCE:.0%}"
            )
        else:
            pass_reasons.append(
                f"Lensed central flux ratio={ratio:.4f} within "
                f"{_LENSED_SNR_RATIO_TOLERANCE:.0%} tolerance"
            )

    # 6. PSF width check (informational — WARN only, not FAIL)
    if psf_cmp:
        width_ratio = psf_cmp.get("width_ratio_c_over_b")
        if width_ratio is not None and width_ratio > 1.2:
            warn_reasons.append(
                f"PSF width ratio={width_ratio:.3f} > 1.20 — "
                "candidate PSF may be broader/blockier at native pixel scale"
            )
        ee_diff = psf_cmp.get("ee_diff_max")
        if ee_diff is not None and ee_diff > 0.05:
            warn_reasons.append(
                f"PSF EE curve max deviation={ee_diff:.4f} > 0.05 "
                "— encircled energy distribution differs noticeably"
            )

    # Overall verdict
    if fail_reasons:
        verdict = "FAIL"
    elif warn_reasons:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
        "pass_reasons": pass_reasons,
    }


# ── output helpers ────────────────────────────────────────────────────────────

def _fmt(v: Any) -> Any:
    """Round floats for CSV output; pass through None and strings."""
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 6)
    return v


def _make_comparison_rows(
    b_null: dict,
    c_null: dict,
    b_lensed: dict,
    c_lensed: dict,
    stamp_diff: dict,
    psf_cmp: dict | None,
    b_elapsed: float,
    c_elapsed: float,
) -> list[dict]:
    """Build flat list of comparison metric rows for CSV output."""
    rows: list[dict] = []

    def _row(metric: str, bv: Any, cv: Any, unit: str = "", note: str = "") -> dict:
        bv_f = _fmt(bv)
        cv_f = _fmt(cv)
        try:
            ratio = cv / bv if (bv is not None and cv is not None and bv != 0) else None
        except (TypeError, ZeroDivisionError):
            ratio = None
        return {
            "metric": metric,
            "baseline_value": bv_f,
            "candidate_value": cv_f,
            "ratio_or_diff": _fmt(ratio),
            "unit": unit,
            "note": note,
        }

    # Null correctness
    rows.append(_row(
        "null_central_snr_mean",
        b_null.get("central_snr_mean"), c_null.get("central_snr_mean"),
        "sigma",
    ))
    rows.append(_row(
        "null_central_snr_max_abs",
        b_null.get("central_snr_max_abs"), c_null.get("central_snr_max_abs"),
        "sigma", "lower is better; ratio = candidate/baseline",
    ))
    rows.append(_row(
        "null_central_pct_mean",
        b_null.get("central_pct_mean"), c_null.get("central_pct_mean"),
        "%", "vs random apertures; lower is better",
    ))
    rows.append(_row(
        "null_rim_ratio_mean",
        b_null.get("rim_ratio_mean"), c_null.get("rim_ratio_mean"),
        "ratio", "rim/interior std; expect ~1",
    ))
    rows.append(_row(
        "null_nonfinite_frac",
        b_null.get("nonfinite_frac"), c_null.get("nonfinite_frac"),
        "fraction", "must be 0.0",
    ))
    rows.append(_row(
        "null_saturation_frac_mean",
        b_null.get("saturation_frac_mean"), c_null.get("saturation_frac_mean"),
        "fraction",
    ))
    rows.append(_row(
        "null_cr_frac_mean",
        b_null.get("cr_frac_mean"), c_null.get("cr_frac_mean"),
        "fraction",
    ))
    rows.append(_row(
        "null_flagged_frac",
        b_null.get("flagged_frac"), c_null.get("flagged_frac"),
        "fraction", "fraction of null epochs with any QA flag",
    ))

    # Lensed correctness
    rows.append(_row(
        "lensed_central_snr_mean",
        b_lensed.get("central_snr_mean"), c_lensed.get("central_snr_mean"),
        "sigma",
    ))
    rows.append(_row(
        "lensed_central_snr_max",
        b_lensed.get("central_snr_max"), c_lensed.get("central_snr_max"),
        "sigma", "peak epoch signal; ratio = candidate/baseline",
    ))
    rows.append(_row(
        "lensed_signal_dominant_frac",
        b_lensed.get("central_signal_dominant_frac"),
        c_lensed.get("central_signal_dominant_frac"),
        "fraction",
    ))
    rows.append(_row(
        "lensed_nonfinite_frac",
        b_lensed.get("nonfinite_frac"), c_lensed.get("nonfinite_frac"),
        "fraction",
    ))
    rows.append(_row(
        "lensed_flagged_frac",
        b_lensed.get("flagged_frac"), c_lensed.get("flagged_frac"),
        "fraction",
    ))

    # Stamp-level differences (candidate - baseline)
    rows.append({
        "metric": "null_stamp_rms_diff_mean",
        "baseline_value": "", "candidate_value": "",
        "ratio_or_diff": _fmt(stamp_diff.get("null_stamp_rms_diff_mean")),
        "unit": "e-", "note": "RMS(candidate_stamp - baseline_stamp) across null events",
    })
    rows.append({
        "metric": "lensed_stamp_rms_diff_mean",
        "baseline_value": "", "candidate_value": "",
        "ratio_or_diff": _fmt(stamp_diff.get("lensed_stamp_rms_diff_mean")),
        "unit": "e-", "note": "RMS(candidate_stamp - baseline_stamp) across lensed events",
    })
    rows.append({
        "metric": "max_abs_pixel_diff",
        "baseline_value": "", "candidate_value": "",
        "ratio_or_diff": _fmt(stamp_diff.get("max_abs_pixel_diff")),
        "unit": "e-", "note": "max |candidate - baseline| over all events/epochs/pixels",
    })
    rows.append(_row(
        "lensed_central_snr_ratio",
        1.0,
        stamp_diff.get("lensed_central_snr_ratio_mean"),
        "ratio", "candidate/baseline aperture mean at peak epoch; 1.0 = identical",
    ))

    # PSF shape
    if psf_cmp:
        rows.append(_row(
            "psf_peak_ratio",
            1.0, psf_cmp.get("peak_ratio_c_over_b"),
            "ratio", f"native-px peak; method={psf_cmp.get('method')}",
        ))
        rows.append(_row(
            "psf_width_ratio",
            1.0, psf_cmp.get("width_ratio_c_over_b"),
            "ratio", "RMS second moment; >1.0 means candidate is broader",
        ))
        rows.append({
            "metric": "psf_ee_diff_max",
            "baseline_value": 0.0,
            "candidate_value": _fmt(psf_cmp.get("ee_diff_max")),
            "ratio_or_diff": "",
            "unit": "fraction", "note": "max |ΔEE(r)| along encircled energy curve",
        })
        rows.append(_row(
            "psf_rim_energy_ratio",
            1.0, psf_cmp.get("rim_energy_ratio_c_over_b"),
            "ratio", "outer-5px rim energy; >1 means more rim leakage in candidate",
        ))

    # Runtime
    rows.append(_row("elapsed_seconds", b_elapsed, c_elapsed, "s"))
    speedup = b_elapsed / c_elapsed if c_elapsed > 0 else None
    rows.append({
        "metric": "speedup_factor",
        "baseline_value": "",
        "candidate_value": "",
        "ratio_or_diff": _fmt(speedup),
        "unit": "x",
        "note": "baseline / candidate; >1 means candidate is faster",
    })

    return rows


def _save_comparison_outputs(
    output_dir: Path,
    comparison: dict,
    rows: list[dict],
) -> None:
    """Write comparison_summary.json, comparison_summary.csv, comparison_report.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    csv_path = output_dir / "comparison_summary.csv"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_COMPARISON_CSV_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(buf.getvalue(), encoding="utf-8")

    # Human-readable report
    verdict_block = comparison["verdict"]
    speedup = comparison.get("speedup_factor")
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A (--skip-* used)"

    report: list[str] = [
        "=" * 72,
        "SMIG Science Profile A/B Comparison Report",
        "=" * 72,
        "",
        f"  Baseline  : {comparison['baseline_config']}",
        f"  Candidate : {comparison['candidate_config']}",
        f"  oversample: {comparison['baseline_oversample']} (baseline)  "
        f"→  {comparison['candidate_oversample']} (candidate)",
        f"  Batch     : {comparison['batch_size']} events × {comparison['n_epochs']} epochs",
        f"  Seed      : {comparison['master_seed']}",
        "",
        f"  Baseline elapsed : {comparison['baseline_elapsed_s']:.1f} s",
        f"  Candidate elapsed: {comparison['candidate_elapsed_s']:.1f} s",
        f"  Speedup factor   : {speedup_str}",
        "",
        "-" * 72,
        f"VERDICT: {verdict_block['verdict']}",
        "-" * 72,
    ]
    for r in verdict_block["fail_reasons"]:
        report.append(f"  [FAIL]  {r}")
    for w in verdict_block["warn_reasons"]:
        report.append(f"  [WARN]  {w}")
    for p in verdict_block["pass_reasons"]:
        report.append(f"  [PASS]  {p}")

    b_null = comparison["null"]["baseline"]
    c_null = comparison["null"]["candidate"]
    b_lens = comparison["lensed"]["baseline"]
    c_lens = comparison["lensed"]["candidate"]

    def _fv(v: Any, fmt: str = ".3f") -> str:
        if v is None:
            return "N/A"
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    report += [
        "",
        "-" * 72,
        "Key metrics:",
        f"  Null max|SNR|    baseline={_fv(b_null.get('central_snr_max_abs'))}  "
        f"candidate={_fv(c_null.get('central_snr_max_abs'))}",
        f"  Null rim ratio   baseline={_fv(b_null.get('rim_ratio_mean'))}  "
        f"candidate={_fv(c_null.get('rim_ratio_mean'))}",
        f"  Lensed SNR mean  baseline={_fv(b_lens.get('central_snr_mean'))}  "
        f"candidate={_fv(c_lens.get('central_snr_mean'))}",
        f"  Lensed SNR max   baseline={_fv(b_lens.get('central_snr_max'))}  "
        f"candidate={_fv(c_lens.get('central_snr_max'))}",
        f"  Lensed flux ratio (candidate/baseline): "
        f"{_fv(comparison['stamp_diff'].get('lensed_central_snr_ratio_mean'))}",
    ]

    psf = comparison.get("psf_comparison")
    if psf:
        report += [
            f"  PSF method      : {psf.get('method', 'N/A')}",
            f"  PSF peak ratio  : {_fv(psf.get('peak_ratio_c_over_b'))}  (c/b)",
            f"  PSF width ratio : {_fv(psf.get('width_ratio_c_over_b'))}  (c/b)",
            f"  PSF EE diff max : {_fv(psf.get('ee_diff_max'))}",
        ]

    report += ["", "=" * 72]
    (output_dir / "comparison_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"\n  Summary JSON   : {json_path}")
    print(f"  Summary CSV    : {csv_path}")
    print(f"  Report         : {output_dir / 'comparison_report.txt'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    baseline_cfg_path = Path(args.baseline_config)
    candidate_cfg_path = Path(args.candidate_config)

    for p in (baseline_cfg_path, candidate_cfg_path):
        if not p.exists():
            sys.exit(f"ERROR: config not found: {p}")

    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "baseline"
    candidate_dir = output_dir / "candidate"

    print("=" * 72)
    print("SMIG Science Profile A/B Comparison")
    print("=" * 72)
    print(f"  Baseline : {baseline_cfg_path}")
    print(f"  Candidate: {candidate_cfg_path}")
    print(f"  Output   : {output_dir.resolve()}")
    print(f"  Batch    : {args.pilot_batch_size} events × {args.n_epochs} epochs")
    print(f"  Seed     : {args.master_seed}")

    # Load configs for metadata and PSF comparison
    baseline_cfg = load_simulation_config(baseline_cfg_path)
    candidate_cfg = load_simulation_config(candidate_cfg_path)

    print(f"\n  Baseline oversample : {baseline_cfg.psf.oversample}")
    print(f"  Candidate oversample: {candidate_cfg.psf.oversample}")

    # Load QA module (dynamic import — avoids circular dependency)
    qa_mod = _load_qa_module()

    # Run baseline
    b_elapsed = 0.0
    if not args.skip_baseline:
        print(f"\n{'─' * 72}")
        print("[1/2] Running BASELINE profile ...")
        b_elapsed = _run_one_profile(
            qa_mod=qa_mod,
            config_path=baseline_cfg_path,
            output_dir=baseline_dir,
            batch_size=args.pilot_batch_size,
            n_epochs=args.n_epochs,
            master_seed=args.master_seed,
            aperture_radius=args.aperture_radius,
            n_random_apertures=args.random_apertures,
        )
        print(f"\n  Baseline completed in {b_elapsed:.1f} s")
    else:
        print(
            f"\n[1/2] Skipping baseline run "
            f"(using existing outputs in {baseline_dir})"
        )

    # Run candidate
    c_elapsed = 0.0
    if not args.skip_candidate:
        print(f"\n{'─' * 72}")
        print("[2/2] Running CANDIDATE profile ...")
        c_elapsed = _run_one_profile(
            qa_mod=qa_mod,
            config_path=candidate_cfg_path,
            output_dir=candidate_dir,
            batch_size=args.pilot_batch_size,
            n_epochs=args.n_epochs,
            master_seed=args.master_seed,
            aperture_radius=args.aperture_radius,
            n_random_apertures=args.random_apertures,
        )
        print(f"\n  Candidate completed in {c_elapsed:.1f} s")
    else:
        print(
            f"\n[2/2] Skipping candidate run "
            f"(using existing outputs in {candidate_dir})"
        )

    speedup = b_elapsed / c_elapsed if c_elapsed > 0 else None
    if speedup is not None:
        print(f"\n  Speedup factor: {speedup:.2f}x (baseline / candidate)")

    # Load pilot rows
    print(f"\n{'─' * 72}")
    print("[ANALYSIS] Loading pilot batch summaries ...")
    try:
        b_rows = _load_pilot_rows(baseline_dir)
        c_rows = _load_pilot_rows(candidate_dir)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    # Aggregate correctness metrics
    b_null = _null_metrics(b_rows)
    c_null = _null_metrics(c_rows)
    b_lensed = _lensed_metrics(b_rows)
    c_lensed = _lensed_metrics(c_rows)

    # Stamp-level comparison
    print("[ANALYSIS] Computing stamp-level pixel differences ...")
    stamp_diff = _compute_stamp_differences(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        batch_size=args.pilot_batch_size,
        aperture_radius=args.aperture_radius,
    )
    print(f"  Matched {stamp_diff['n_events_matched']}/{stamp_diff['n_events_attempted']} events")

    # PSF shape comparison
    psf_cmp: dict | None = None
    if not args.skip_psf_comparison:
        print("[ANALYSIS] Computing PSF shape comparison ...")
        try:
            psf_cmp = _compute_psf_comparison(
                baseline_cfg=baseline_cfg,
                candidate_cfg=candidate_cfg,
                fov_native=64,
            )
            print(f"  PSF comparison method: {psf_cmp['method']}")
        except Exception as exc:
            print(f"  [WARN] PSF comparison failed: {exc}")

    # Pass/fail verdict
    verdict = _pass_fail_verdict(
        b_null=b_null,
        c_null=c_null,
        b_lensed=b_lensed,
        c_lensed=c_lensed,
        stamp_diff=stamp_diff,
        psf_cmp=psf_cmp,
    )

    # Assemble full comparison dict
    comparison: dict = {
        "baseline_config": str(baseline_cfg_path.resolve()),
        "candidate_config": str(candidate_cfg_path.resolve()),
        "baseline_config_sha256": get_simulation_config_sha256(baseline_cfg),
        "candidate_config_sha256": get_simulation_config_sha256(candidate_cfg),
        "batch_size": args.pilot_batch_size,
        "n_epochs": args.n_epochs,
        "master_seed": args.master_seed,
        "baseline_oversample": baseline_cfg.psf.oversample,
        "candidate_oversample": candidate_cfg.psf.oversample,
        "baseline_elapsed_s": round(b_elapsed, 2),
        "candidate_elapsed_s": round(c_elapsed, 2),
        "speedup_factor": round(speedup, 3) if speedup is not None else None,
        "null": {"baseline": b_null, "candidate": c_null},
        "lensed": {"baseline": b_lensed, "candidate": c_lensed},
        "stamp_diff": stamp_diff,
        "psf_comparison": psf_cmp,
        "verdict": verdict,
    }

    rows = _make_comparison_rows(
        b_null=b_null,
        c_null=c_null,
        b_lensed=b_lensed,
        c_lensed=c_lensed,
        stamp_diff=stamp_diff,
        psf_cmp=psf_cmp,
        b_elapsed=b_elapsed,
        c_elapsed=c_elapsed,
    )

    print(f"\n{'─' * 72}")
    print("[OUTPUT] Writing comparison summaries ...")
    _save_comparison_outputs(
        output_dir=output_dir,
        comparison=comparison,
        rows=rows,
    )

    # Print verdict summary
    print(f"\n{'=' * 72}")
    print(f"VERDICT: {verdict['verdict']}")
    for r in verdict["fail_reasons"]:
        print(f"  [FAIL] {r}")
    for w in verdict["warn_reasons"]:
        print(f"  [WARN] {w}")
    for p in verdict["pass_reasons"]:
        print(f"  [PASS] {p}")
    print("=" * 72)

    if verdict["verdict"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
