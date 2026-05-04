#!/usr/bin/env python3
"""
scripts/generate_phase2_qa_stamps.py

QA generator for SMIG Phase 2 difference-stamp pipeline.

This script is a standalone diagnostic utility, not a pipeline replacement.
It exercises SceneSimulator with two deterministic synthetic events (null and
lensed) and saves difference stamps, saturation/CR masks, a metadata JSON,
and optional PNG previews to an output directory.  All outputs are reproducible
given the same --master-seed.  The script does not modify any pipeline module.

Central residual checks
-----------------------
For every difference stamp the script computes a circular-aperture metric at
the stamp centre, compares it against an annulus background estimate, and
derives a per-epoch SNR.  For null events it additionally samples
``--random-apertures`` off-centre same-radius apertures (fixed RNG seed) and
reports where the central aperture falls in that distribution.  Warnings are
printed when the null-event central SNR or percentile exceeds configurable
thresholds, or when the same-sign residual persists across all epochs.

Fixed-scale PNGs
----------------
In addition to per-stamp autoscaled PNGs, a second set of fixed-scale PNGs
is saved: one using a per-event robust scale and one using a global scale
across all events.  The robust scale is max(|p1|, |p99|) of all pixel values
in scope.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

try:
    from astropy.io import fits as _fits
    _FITS = True
except ImportError:
    _FITS = False

try:
    import pandas as _pd
    _PD = True
except ImportError:
    _PD = False

try:
    from scipy.signal import convolve2d as _convolve2d
    _SCIPY = True
except ImportError:
    _convolve2d = None  # type: ignore[assignment]
    _SCIPY = False

# ── SMIG public API ──────────────────────────────────────────────────────────
from smig.config.utils import get_simulation_config_sha256, load_simulation_config
from smig.rendering.pipeline import SceneSimulator

# ── defaults ─────────────────────────────────────────────────────────────────
_DEFAULT_CONFIGS = [
    "smig/config/simulation_science.yaml",
    "smig/config/simulation.yaml",
]
_DEFAULT_OUTPUT_DIR = "outputs/phase2_qa_stamps"
_DEFAULT_N_EPOCHS = 3
_DEFAULT_MASTER_SEED = 12345
_DEFAULT_APERTURE_RADIUS = 3
_DEFAULT_RANDOM_APERTURES = 500
_DEFAULT_CENTRAL_SNR_WARN = 5.0
_DEFAULT_CENTRAL_PERCENTILE_WARN = 99.0

# Baseline source flux: ~21 AB mag at 46.8 s exposure, Roman W146
_BASELINE_SOURCE_FLUX_E = 800.0

# Sky background in e⁻/s per pixel (~22.5 AB mag/arcsec² zodiacal light)
_SKY_BG_E_PER_S = 2.0

# Subdirectory for null ablation outputs
_ABLATION_SUBDIR = "null_ablations"

# |SNR| threshold for classifying a DIA-layer residual as "high".
_DIA_HIGH_SNR_THRESHOLD = 3.0

# ── AL diagnostic constants ────────────────────────────────────────────────────
# These must match smig/rendering/dia.py DIAPipeline exactly.
_AL_SIGMAS_PRODUCTION: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
_AL_KERNEL_SIZE: int = 33           # 2 * int(ceil(4 * 4.0)) + 1
_AL_DIAG_EXCL_RADIUS: int = 8      # central exclusion radius for mask ablations
_AL_ILL_COND_THRESHOLD: float = 1e6

# ── pilot batch defaults ──────────────────────────────────────────────────────
_DEFAULT_PILOT_BATCH_SIZE = 20
_DEFAULT_PILOT_OUTPUT_DIR = "outputs/phase2_pilot_qa"
_DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT = 3.0
_DEFAULT_RIM_RATIO_LOW = 0.5
_DEFAULT_RIM_RATIO_HIGH = 2.0

_PILOT_EXPECTED_DIA_METHOD = "identity"
_PILOT_EXPECTED_REFERENCE_PROCESSING = "detector_matched"
_PILOT_EXPECTED_REFERENCE_JITTER_MODE = "per_epoch"

_PILOT_CSV_FIELDNAMES = [
    "event_id", "event_type", "epoch",
    "mean", "std", "min", "max", "p5", "p50", "p95",
    "central_aperture_snr", "central_aperture_percentile",
    "rim_interior_std_ratio", "central_signal_dominant",
    "has_nonfinite",
    "saturation_pixel_fraction", "cr_pixel_fraction",
    "dia_method", "reference_processing", "reference_jitter_mode",
    "psf_fov_native_pixels", "oversample", "n_wavelengths",
    "qa_flags",
]

# ── argument parsing ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Phase 2 QA difference stamps using SMIG public APIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to simulation YAML config.")
    p.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR, help="Output directory.")
    p.add_argument("--n-epochs", type=int, default=_DEFAULT_N_EPOCHS, help="Number of science epochs.")
    p.add_argument("--master-seed", type=int, default=_DEFAULT_MASTER_SEED, help="Master RNG seed.")
    p.add_argument(
        "--central-aperture-radius",
        type=int, default=_DEFAULT_APERTURE_RADIUS,
        help="Circular aperture radius in pixels for central residual metrics.",
    )
    p.add_argument(
        "--random-apertures",
        type=int, default=_DEFAULT_RANDOM_APERTURES,
        help="Number of random off-centre apertures sampled for null-event comparison.",
    )
    p.add_argument(
        "--central-snr-warn",
        type=float, default=_DEFAULT_CENTRAL_SNR_WARN,
        help="Warn if null-event central aperture SNR exceeds this threshold.",
    )
    p.add_argument(
        "--central-percentile-warn",
        type=float, default=_DEFAULT_CENTRAL_PERCENTILE_WARN,
        help="Warn if null-event central aperture percentile (vs random) exceeds this.",
    )
    p.add_argument(
        "--run-null-ablations",
        action="store_true",
        default=False,
        help=(
            "Run null-ablation variant suite after main QA to diagnose sources of "
            "central residual (flux mismatch, centroid mismatch, PSF jitter, crowding, "
            "reference processing, DIA kernel). Outputs to <output-dir>/null_ablations/."
        ),
    )
    p.add_argument(
        "--capture-dia-inputs",
        action="store_true",
        default=False,
        help=(
            "Capture the science_rate and reference_rate images passed to "
            "DIAPipeline.subtract for each epoch, via a QA-only monkeypatch. "
            "Saves per-epoch science/reference/raw_difference/post_al .npy arrays "
            "and optional PNGs. Adds DIA-layer SNR metrics and a "
            "raw_vs_post_al_interpretation field to metadata. "
            "Does not modify any pipeline source file."
        ),
    )
    p.add_argument(
        "--dia-diagnostic-modes",
        action="store_true",
        default=False,
        help=(
            "Run QA-only AL fit diagnostics and ablation variants for every captured "
            "DIA input (implies --capture-dia-inputs). Computes seven diagnostic "
            "subtraction variants (identity_only, current_al, background_masked_al, "
            "rim_or_annulus_fit_al, reduced_basis_al, regularized_al_1e-6/1e-4), "
            "records AL fit coefficients and condition numbers, saves fixed-scale PNGs "
            "and dia_al_diagnostic_summary.json/.csv. "
            "Does NOT modify any pipeline source file."
        ),
    )
    # ── pilot batch ───────────────────────────────────────────────────────────
    p.add_argument(
        "--pilot-batch-size",
        type=int, default=0,
        metavar="N",
        help=(
            "If > 0, run a pilot batch of N deterministic events instead of the "
            "default two-event QA run. Events alternate null/lensed. "
            f"Output defaults to {_DEFAULT_PILOT_OUTPUT_DIR!r} unless --output-dir is "
            "set explicitly."
        ),
    )
    p.add_argument(
        "--null-central-snr-warn",
        type=float, default=_DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT,
        help="[pilot] Flag null epochs with |central_aperture_snr| above this value.",
    )
    p.add_argument(
        "--rim-ratio-low",
        type=float, default=_DEFAULT_RIM_RATIO_LOW,
        help="[pilot] Flag epochs with rim/interior std ratio below this.",
    )
    p.add_argument(
        "--rim-ratio-high",
        type=float, default=_DEFAULT_RIM_RATIO_HIGH,
        help="[pilot] Flag epochs with rim/interior std ratio above this.",
    )
    p.add_argument(
        "--max-nonfinite-allowed",
        type=int, default=0,
        help="[pilot] Allowed nonfinite pixels before flagging (0 = any nonfinite flags).",
    )
    p.add_argument(
        "--max-saturation-fraction",
        type=float, default=0.0,
        help="[pilot] Flag epochs with saturation_pixel_fraction above this.",
    )
    p.add_argument(
        "--max-cr-fraction",
        type=float, default=None,
        help="[pilot] If set, flag epochs with cr_pixel_fraction above this.",
    )
    return p


def _resolve_config(path_arg: str | None) -> Path:
    if path_arg is not None:
        p = Path(path_arg)
        if not p.exists():
            sys.exit(f"ERROR: config not found: {p}")
        return p
    for candidate in _DEFAULT_CONFIGS:
        p = Path(candidate)
        if p.exists():
            return p
    sys.exit(
        f"ERROR: no config found at default paths {_DEFAULT_CONFIGS}. "
        "Pass --config explicitly."
    )


# ── signal helpers ────────────────────────────────────────────────────────────

def _gaussian_magnification_curve(n: int, peak: float = 5.0) -> list[float]:
    """Gaussian magnification bell centered on the middle epoch."""
    if n == 1:
        return [peak]
    center = (n - 1) / 2.0
    sigma = max(center / 2.0, 0.5)
    return [
        1.0 + (peak - 1.0) * float(np.exp(-0.5 * ((i - center) / sigma) ** 2))
        for i in range(n)
    ]


# ── statistics helpers ────────────────────────────────────────────────────────

def _rim_stats(stamp: np.ndarray, rim_width: int = 5) -> dict[str, float]:
    h, w = stamp.shape
    mask_rim = np.ones((h, w), dtype=bool)
    if h > 2 * rim_width and w > 2 * rim_width:
        mask_rim[rim_width:-rim_width, rim_width:-rim_width] = False
    rim = stamp[mask_rim]
    interior = stamp[~mask_rim]
    rim_std = float(np.std(rim)) if rim.size else float("nan")
    interior_std = float(np.std(interior)) if interior.size else float("nan")
    if interior.size and interior_std > 0.0:
        rim_interior_std_ratio = float(rim_std / interior_std)
    else:
        rim_interior_std_ratio = float("nan")
    return {
        "rim_mean": float(np.mean(rim)) if rim.size else float("nan"),
        "rim_std": rim_std,
        "rim_max_abs": float(np.max(np.abs(rim))) if rim.size else float("nan"),
        "interior_mean": float(np.mean(interior)) if interior.size else float("nan"),
        "interior_std": interior_std,
        "rim_interior_std_ratio": rim_interior_std_ratio,
    }


def _stamp_summary(arr: np.ndarray) -> dict[str, float]:
    flat = arr.ravel()
    pcts = np.percentile(flat, [5, 25, 50, 75, 95])
    return {
        "mean": round(float(np.mean(flat)), 4),
        "std": round(float(np.std(flat)), 4),
        "min": round(float(np.min(flat)), 4),
        "max": round(float(np.max(flat)), 4),
        "p5": round(float(pcts[0]), 4),
        "p25": round(float(pcts[1]), 4),
        "p50": round(float(pcts[2]), 4),
        "p75": round(float(pcts[3]), 4),
        "p95": round(float(pcts[4]), 4),
        "sum": round(float(np.sum(flat)), 4),
    }


def _circular_aperture_mask(
    h: int, w: int, cy: float, cx: float, radius: float
) -> np.ndarray:
    """Boolean mask: True for pixels whose centre is within `radius` of (cy, cx)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def _central_residual_metrics(
    stamp: np.ndarray,
    aperture_radius: int = 3,
) -> dict[str, float]:
    """
    Aperture photometry centred on the stamp.

    Background estimate: annulus from (aperture_radius + 3) to
    (aperture_radius + 9) pixels.  SNR = aperture_excess / (bg_std * sqrt(n_ap)).
    """
    h, w = stamp.shape
    cy, cx = float(h // 2), float(w // 2)

    center_pixel = float(stamp[h // 2, w // 2])

    # 3×3 box sum/mean
    r0, r1 = max(0, h // 2 - 1), min(h, h // 2 + 2)
    c0, c1 = max(0, w // 2 - 1), min(w, w // 2 + 2)
    box3 = stamp[r0:r1, c0:c1]
    central_3x3_sum = round(float(box3.sum()), 4)
    central_3x3_mean = round(float(box3.mean()), 4)

    # Circular aperture
    ap_mask = _circular_aperture_mask(h, w, cy, cx, aperture_radius)
    n_ap = int(ap_mask.sum())
    ap_sum = float(stamp[ap_mask].sum())

    # Annulus background
    r_inner = float(aperture_radius + 3)
    r_outer = float(min(aperture_radius + 9, min(h, w) // 2 - 1))
    if r_outer > r_inner:
        outer_mask = _circular_aperture_mask(h, w, cy, cx, r_outer)
        inner_mask = _circular_aperture_mask(h, w, cy, cx, r_inner)
        annulus = stamp[outer_mask & ~inner_mask]
        bg_per_pix = float(annulus.mean()) if annulus.size > 0 else 0.0
        bg_std = float(annulus.std()) if annulus.size > 1 else float(stamp.std())
    else:
        bg_per_pix = 0.0
        bg_std = float(stamp.std())

    ap_excess = ap_sum - bg_per_pix * n_ap
    # Gaussian noise propagation: sigma_excess = bg_std * sqrt(n_ap).
    # Floor at 1e-9 so SNR retains the correct sign even when bg_std == 0.
    noise_est = max(bg_std * float(np.sqrt(max(n_ap, 1))), 1e-9)
    snr = ap_excess / noise_est

    return {
        "center_pixel": round(center_pixel, 4),
        "central_3x3_sum": central_3x3_sum,
        "central_3x3_mean": central_3x3_mean,
        "central_aperture_sum": round(ap_sum, 4),
        "central_aperture_n_pix": n_ap,
        "central_aperture_excess": round(ap_excess, 4),
        "central_aperture_snr": round(snr, 4),
        "annulus_bg_per_pix": round(bg_per_pix, 4),
        "annulus_bg_std": round(bg_std, 4),
    }


def _null_random_apertures(
    stamp: np.ndarray,
    aperture_radius: int,
    n_samples: int,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compare the central aperture excess against a distribution of same-radius
    apertures placed at random off-centre positions.

    Positions are drawn uniformly within the stamp (margin = aperture_radius + 1)
    and excluding a circle of radius 3 × aperture_radius around the stamp centre.

    Returns percentile and z-score of the central excess in the random
    distribution.  The seed is fixed for reproducibility independent of the
    master RNG.
    """
    h, w = stamp.shape
    cy, cx = h // 2, w // 2
    margin = aperture_radius + 1
    center_excl_r = aperture_radius * 3

    rng = np.random.default_rng(seed)
    global_bg = float(stamp.mean())

    # Central aperture
    ap_mask_ctr = _circular_aperture_mask(h, w, float(cy), float(cx), aperture_radius)
    n_ap = int(ap_mask_ctr.sum())
    central_sum = float(stamp[ap_mask_ctr].sum())
    central_excess = central_sum - global_bg * n_ap

    # Sample random candidate positions
    candidates_y = rng.integers(margin, h - margin, size=n_samples * 4)
    candidates_x = rng.integers(margin, w - margin, size=n_samples * 4)
    dist2 = (candidates_y - cy) ** 2 + (candidates_x - cx) ** 2
    valid = dist2 > center_excl_r ** 2
    ys = candidates_y[valid][:n_samples]
    xs = candidates_x[valid][:n_samples]

    if len(ys) < 10:
        return {
            "n_random_apertures": 0,
            "central_aperture_percentile": float("nan"),
            "central_aperture_z_score": float("nan"),
        }

    random_sums = np.array([
        float(stamp[_circular_aperture_mask(h, w, float(y), float(x), aperture_radius)].sum())
        for y, x in zip(ys, xs)
    ])
    random_excesses = random_sums - global_bg * n_ap

    percentile = float(np.mean(random_excesses < central_excess) * 100.0)
    rand_mean = float(np.mean(random_excesses))
    rand_std = float(np.std(random_excesses))
    z_score = (central_excess - rand_mean) / rand_std if rand_std > 0.0 else 0.0

    return {
        "n_random_apertures": int(len(ys)),
        "random_aperture_mean_excess": round(rand_mean, 4),
        "random_aperture_std_excess": round(rand_std, 4),
        "central_aperture_percentile": round(percentile, 2),
        "central_aperture_z_score": round(z_score, 4),
    }


def _robust_scale(diffs: np.ndarray) -> float:
    """Symmetric robust scale for a fixed-scale colormap: max(|p1|, |p99|)."""
    flat = diffs.ravel()
    p1, p99 = np.percentile(flat, [1, 99])
    return max(abs(float(p1)), abs(float(p99)), 1.0)


# ── DIA-input capture (QA-only monkeypatch) ──────────────────────────────────


class _DiaCaptureContext:
    """Monkeypatches DIAPipeline.subtract to record its inputs and outputs.

    One entry per ``subtract`` call (one per science epoch) is appended to
    ``capture_list`` as::

        {
            "science":   np.ndarray (float64, context-stamp shape),
            "reference": np.ndarray (float64, context-stamp shape),
            "raw_diff":  np.ndarray = science - reference (float64),
            "post_al":   np.ndarray = subtract result (float64),
        }

    The original ``DIAPipeline.subtract`` is always restored on ``__exit__``,
    including when an exception is raised inside the ``with`` block.
    Import of ``smig.rendering.dia`` is deferred so it does not appear in
    module-level scope (preserving the "no pipeline internals at import"
    contract verified by the test suite).
    """

    def __init__(self, capture_list: list) -> None:
        self._capture_list = capture_list
        self._cls = None
        self._original = None

    def __enter__(self) -> "_DiaCaptureContext":
        from smig.rendering.dia import DIAPipeline  # deferred: must stay here

        self._cls = DIAPipeline
        self._original = DIAPipeline.subtract
        _cap = self._capture_list
        _orig = self._original

        def _wrapper(instance, science_rate_image, reference_rate_image):
            result = _orig(instance, science_rate_image, reference_rate_image)
            sci = science_rate_image.astype(np.float64, copy=True)
            ref = reference_rate_image.astype(np.float64, copy=True)
            _cap.append({
                "science":   sci,
                "reference": ref,
                "raw_diff":  sci - ref,
                "post_al":   result.astype(np.float64, copy=True),
            })
            return result

        DIAPipeline.subtract = _wrapper
        return self

    def __exit__(self, *exc_info) -> bool:
        if self._cls is not None and self._original is not None:
            self._cls.subtract = self._original
        return False  # propagate any exception


def _dia_input_interpretation(
    raw_snr: float,
    post_snr: float,
    threshold: float = _DIA_HIGH_SNR_THRESHOLD,
) -> str:
    """Classify the relationship between raw-DIA and post-AL residuals.

    Returns one of four labels:
        "both_high"         — residual present before and after AL subtraction
        "raw_residual_present" — raw high, post-AL clean (AL fixed it)
        "post_al_introduced"   — raw clean, post-AL introduces residual
        "both_clean"           — neither layer shows a significant residual
    """
    raw_high = abs(raw_snr) >= threshold
    post_high = abs(post_snr) >= threshold
    if raw_high and post_high:
        return "both_high"
    if raw_high and not post_high:
        return "raw_residual_present"
    if not raw_high and post_high:
        return "post_al_introduced"
    return "both_clean"


def _compute_dia_input_epoch_metrics(
    capture_entry: dict,
    aperture_radius: int,
) -> dict:
    """Compute per-epoch DIA-layer metrics from one captured subtract call.

    Parameters
    ----------
    capture_entry :
        One element of the list produced by ``_DiaCaptureContext``.
    aperture_radius :
        Circular aperture radius in pixels (same as the main QA metric).

    Returns
    -------
    dict
        Per-epoch metadata fields as specified in the task description.
    """
    sci_metrics = _central_residual_metrics(capture_entry["science"], aperture_radius)
    ref_metrics = _central_residual_metrics(capture_entry["reference"], aperture_radius)
    raw_metrics = _central_residual_metrics(capture_entry["raw_diff"], aperture_radius)
    post_metrics = _central_residual_metrics(capture_entry["post_al"], aperture_radius)

    sci_ap = sci_metrics["central_aperture_sum"]
    ref_ap = ref_metrics["central_aperture_sum"]
    raw_snr = raw_metrics["central_aperture_snr"]
    post_snr = post_metrics["central_aperture_snr"]

    ratio = (sci_ap / ref_ap) if abs(ref_ap) > 1e-9 else None

    return {
        "dia_input_capture_enabled": True,
        "science_center_aperture_sum": round(sci_ap, 4),
        "reference_center_aperture_sum": round(ref_ap, 4),
        "science_minus_reference_center_aperture_sum": round(
            raw_metrics["central_aperture_sum"], 4
        ),
        "science_minus_reference_center_aperture_snr": round(raw_snr, 4),
        "post_al_center_aperture_snr": round(post_snr, 4),
        "science_reference_center_ratio": round(ratio, 4) if ratio is not None else None,
        "raw_vs_post_al_interpretation": _dia_input_interpretation(raw_snr, post_snr),
    }


def _save_dia_capture_arrays(
    capture_entries: list[dict],
    out_dir: Path,
    sci_sz: int,
    ctx_sz: int,
) -> None:
    """Save captured DIA-layer arrays as .npy and optional fixed-scale PNGs.

    Saves four files per epoch:
        science_rate_epoch_{i}.npy
        reference_rate_epoch_{i}.npy
        raw_difference_epoch_{i}.npy   (= science - reference)
        post_al_difference_epoch_{i}.npy

    If matplotlib is available, also saves PNG central crops (science_stamp_size)
    at a fixed scale shared across all epochs for each layer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Central-crop slice: identical to pipeline.py's row_start/col_start logic.
    half = sci_sz // 2
    center = ctx_sz // 2
    slc = slice(center - half, center + half)

    for i, entry in enumerate(capture_entries):
        np.save(out_dir / f"science_rate_epoch_{i}.npy",      entry["science"])
        np.save(out_dir / f"reference_rate_epoch_{i}.npy",    entry["reference"])
        np.save(out_dir / f"raw_difference_epoch_{i}.npy",    entry["raw_diff"])
        np.save(out_dir / f"post_al_difference_epoch_{i}.npy", entry["post_al"])

    if not _MPL or not capture_entries:
        return

    # Fixed scale per layer across all epochs for fair visual comparison.
    def _layer_scale(key: str) -> float:
        all_vals = np.concatenate([e[key][slc, slc].ravel() for e in capture_entries])
        return _robust_scale(all_vals)

    scales = {k: _layer_scale(k) for k in ("science", "reference", "raw_diff", "post_al")}
    labels = {
        "science":   "science_rate",
        "reference": "reference_rate",
        "raw_diff":  "raw_diff (sci−ref)",
        "post_al":   "post_AL diff",
    }

    for i, entry in enumerate(capture_entries):
        for key, label in labels.items():
            crop = entry[key][slc, slc]
            _save_png_fixed_scale(
                crop,
                out_dir / f"{key}_epoch_{i}.png",
                title=f"{label} epoch {i}",
                vmax=scales[key],
            )


def _print_dia_capture_summary(
    variant_name: str,
    epoch_metrics: list[dict],
) -> None:
    """Print a concise per-variant DIA-layer residual summary to stdout."""
    if not epoch_metrics:
        return

    raw_snrs = [m["science_minus_reference_center_aperture_snr"] for m in epoch_metrics]
    post_snrs = [m["post_al_center_aperture_snr"] for m in epoch_metrics]
    ratios = [m["science_reference_center_ratio"] for m in epoch_metrics if m["science_reference_center_ratio"] is not None]
    interps = [m["raw_vs_post_al_interpretation"] for m in epoch_metrics]

    print(f"  [DIA-capture] {variant_name}:")
    print(
        f"    raw (sci−ref) central SNR  "
        f"mean={float(np.mean(raw_snrs)):+.3f}  "
        f"max|{max(abs(s) for s in raw_snrs):.3f}|"
    )
    print(
        f"    post-AL central SNR        "
        f"mean={float(np.mean(post_snrs)):+.3f}  "
        f"max|{max(abs(s) for s in post_snrs):.3f}|"
    )
    if ratios:
        print(f"    sci/ref center ratio mean  {float(np.mean(ratios)):.4f}")
    # Majority interpretation
    from collections import Counter
    dominant = Counter(interps).most_common(1)[0][0]
    print(f"    interpretation (dominant): {dominant}")
    if dominant == "both_high":
        print(
            "    -> Residual present BEFORE AL subtraction. "
            "Inspect science/reference rendering consistency."
        )
    elif dominant == "post_al_introduced":
        print(
            "    -> Raw difference is clean; AL kernel fit introduces the residual. "
            "Inspect Alard-Lupton kernel / model degrees of freedom."
        )
    elif dominant == "raw_residual_present":
        print(
            "    -> Residual present before AL, removed after. "
            "AL partially compensates for science/reference mismatch."
        )
    elif dominant == "both_clean":
        print("    -> Both layers are clean at this threshold.")


# ── QA-local AL fitting (no production code touched) ─────────────────────────
#
# Reproduces DIAPipeline._make_gaussian_kernel + subtract design-matrix logic
# so QA ablation variants can vary fit parameters without touching any pipeline
# source file.  Source of truth for production constants is documented in
# smig/rendering/dia.py (class variable _AL_SIGMAS, method subtract).


def _make_gaussian_kernel_qa(sigma: float, size: int) -> np.ndarray:
    """Normalized 2D Gaussian kernel; matches DIAPipeline._make_gaussian_kernel."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def _al_convolve_ref_qa(
    ref: np.ndarray,
    sigmas: tuple[float, ...],
    kernel_size: int,
) -> list[np.ndarray]:
    """Convolve reference with each Gaussian basis kernel (matches production)."""
    if not _SCIPY:
        raise RuntimeError("scipy.signal not available; AL diagnostics disabled.")
    planes: list[np.ndarray] = []
    for sigma in sigmas:
        kern = _make_gaussian_kernel_qa(sigma, kernel_size)
        planes.append(_convolve2d(ref, kern, mode="same", boundary="symm"))
    return planes


def _build_al_design_matrix_qa(convolved_planes: list[np.ndarray]) -> np.ndarray:
    """Build [N_pix, n_basis+1] design matrix with a constant column."""
    n_pixels = convolved_planes[0].size
    n_basis = len(convolved_planes)
    A = np.empty((n_pixels, n_basis + 1), dtype=np.float64)
    for k, plane in enumerate(convolved_planes):
        A[:, k] = plane.ravel()
    A[:, n_basis] = 1.0
    return A


def _al_lstsq_qa(A: np.ndarray, b: np.ndarray) -> dict:
    """Solve via standard least squares; return fit diagnostics dict."""
    coeffs, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    cond = float(sv[0] / sv[-1]) if (len(sv) > 0 and float(sv[-1]) > 0) else float("inf")
    return {
        "coefficients": [round(float(c), 8) for c in coeffs],
        "rank": int(rank),
        "condition_number": round(cond, 4),
        "singular_values": [round(float(s), 6) for s in sv],
    }


def _al_ridge_qa(A: np.ndarray, b: np.ndarray, lam_scale: float) -> dict:
    """Ridge-regularized LS: (AᵀA + λ·‖AᵀA‖_F · I)x = Aᵀb."""
    AtA = A.T @ A
    frob = float(np.linalg.norm(AtA, "fro"))
    lam_abs = lam_scale * frob
    AtA_reg = AtA + lam_abs * np.eye(AtA.shape[0])
    coeffs = np.linalg.solve(AtA_reg, A.T @ b)
    sv = np.linalg.svd(AtA_reg, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if (len(sv) > 0 and float(sv[-1]) > 0) else float("inf")
    return {
        "coefficients": [round(float(c), 8) for c in coeffs],
        "rank": None,
        "condition_number": round(cond, 4),
        "singular_values": None,
        "ridge_lambda_scale": lam_scale,
        "ridge_lambda_abs": round(lam_abs, 8),
    }


def _al_subtract_qa(
    sci: np.ndarray,
    ref: np.ndarray,
    sigmas: tuple[float, ...],
    kernel_size: int,
    pixel_mask: np.ndarray | None = None,
    ridge_lambda_scale: float = 0.0,
) -> dict:
    """
    QA-only AL subtraction without touching DIAPipeline.

    Parameters
    ----------
    pixel_mask :
        Boolean mask (True = pixel included in fit).  All pixels are used for
        reconstruction regardless of this mask.
    ridge_lambda_scale :
        If > 0, ridge regression: λ = ridge_lambda_scale × ‖AᵀA‖_F.
    """
    ny, nx = sci.shape
    planes = _al_convolve_ref_qa(ref, sigmas, kernel_size)
    A_full = _build_al_design_matrix_qa(planes)
    b_full = sci.ravel()

    if pixel_mask is not None:
        flat_mask = pixel_mask.ravel()
        A_fit = A_full[flat_mask]
        b_fit = b_full[flat_mask]
    else:
        A_fit = A_full
        b_fit = b_full

    if ridge_lambda_scale > 0.0:
        fit_info = _al_ridge_qa(A_fit, b_fit, ridge_lambda_scale)
    else:
        fit_info = _al_lstsq_qa(A_fit, b_fit)

    coeffs = np.array(fit_info["coefficients"], dtype=np.float64)
    matched_ref = np.zeros((ny, nx), dtype=np.float64)
    for k, plane in enumerate(planes):
        matched_ref += coeffs[k] * plane
    matched_ref += coeffs[len(planes)]

    return {
        "difference": sci - matched_ref,
        "matched_ref": matched_ref,
        "fit_info": fit_info,
    }


def _al_subtract_identity_plus_gaussians_qa(
    sci: np.ndarray,
    ref: np.ndarray,
    sigmas: tuple[float, ...],
    kernel_size: int,
    pixel_mask: np.ndarray | None = None,
) -> dict:
    """
    QA-only AL subtraction with an augmented basis: raw reference + Gaussian basis + constant.

    Design matrix columns:
        [ref.ravel(), conv_ref_0.ravel(), ..., conv_ref_k.ravel(), 1.0]

    Matched reference:
        matched_ref = c_identity * ref
                    + sum(c_k * conv_ref_k)
                    + c_bg

    This tests whether providing an exact identity component allows the solver to
    represent the trivial subtraction limit, eliminating the source-core residual
    caused by the pure-Gaussian basis lacking that component.

    fit_info extras:
        identity_coefficient                 — c_identity (coefficient on raw ref)
        sum_gaussian_correction_coefficients — sum of c_k for all Gaussian planes
    """
    ny, nx = sci.shape
    planes = _al_convolve_ref_qa(ref, sigmas, kernel_size)

    n_pixels = ref.size
    n_gaussians = len(planes)
    n_cols = 1 + n_gaussians + 1  # identity + gaussians + constant
    A_full = np.empty((n_pixels, n_cols), dtype=np.float64)
    A_full[:, 0] = ref.ravel()
    for k, plane in enumerate(planes):
        A_full[:, k + 1] = plane.ravel()
    A_full[:, -1] = 1.0

    b_full = sci.astype(np.float64).ravel()

    if pixel_mask is not None:
        flat_mask = pixel_mask.ravel()
        A_fit = A_full[flat_mask]
        b_fit = b_full[flat_mask]
    else:
        A_fit = A_full
        b_fit = b_full

    fit_info = _al_lstsq_qa(A_fit, b_fit)
    coeffs = np.array(fit_info["coefficients"], dtype=np.float64)

    matched_ref = coeffs[0] * ref.astype(np.float64)
    for k, plane in enumerate(planes):
        matched_ref += coeffs[k + 1] * plane
    matched_ref += coeffs[-1]

    fit_info["identity_coefficient"] = round(float(coeffs[0]), 8)
    fit_info["sum_gaussian_correction_coefficients"] = round(
        float(np.sum(coeffs[1:-1])), 8
    )

    return {
        "difference": sci.astype(np.float64) - matched_ref,
        "matched_ref": matched_ref,
        "fit_info": fit_info,
    }


def _al_diag_variant_metrics(diff: np.ndarray, aperture_radius: int) -> dict:
    """Compute per-variant metrics for one diagnostic difference stamp."""
    central = _central_residual_metrics(diff, aperture_radius)
    rim = _rim_stats(diff)
    interior_std = rim["interior_std"]
    rim_std = rim["rim_std"]
    ratio = (rim_std / interior_std) if interior_std > 1e-12 else float("nan")
    flat = diff.ravel()
    return {
        "central_aperture_snr": central["central_aperture_snr"],
        "rms": round(float(np.std(flat)), 6),
        "mean": round(float(np.mean(flat)), 6),
        "max_abs": round(float(np.max(np.abs(flat))), 6),
        "rim_interior_std_ratio": (
            round(float(ratio), 6) if np.isfinite(ratio) else None
        ),
    }


def _run_al_diagnostics_for_epoch(
    sci: np.ndarray,
    ref: np.ndarray,
    post_al: np.ndarray,
    aperture_radius: int,
    excl_radius: int = _AL_DIAG_EXCL_RADIUS,
) -> dict[str, dict]:
    """
    Run all QA-only AL diagnostic variants for one captured epoch.

    Returns a dict keyed by variant name.  Each value has:
        difference   np.ndarray
        metrics      dict from _al_diag_variant_metrics
        fit_info     dict from _al_lstsq_qa / _al_ridge_qa, or None
    """
    if not _SCIPY:
        return {}

    ny, nx = sci.shape
    cy, cx = float(ny // 2), float(nx // 2)
    prod_sigmas = _AL_SIGMAS_PRODUCTION
    kern_sz = _AL_KERNEL_SIZE
    results: dict[str, dict] = {}

    # 1. identity_only — raw difference, no AL fit.
    ident = (sci - ref).astype(np.float64)
    results["identity_only"] = {
        "difference": ident,
        "metrics": _al_diag_variant_metrics(ident, aperture_radius),
        "fit_info": None,
    }

    # 2. current_al — captured from production DIAPipeline.subtract.
    results["current_al"] = {
        "difference": post_al.astype(np.float64),
        "metrics": _al_diag_variant_metrics(post_al, aperture_radius),
        "fit_info": None,
    }

    # Central exclusion mask: True = pixel is included in the fit.
    center_disk = _circular_aperture_mask(ny, nx, cy, cx, float(excl_radius))

    # 3. background_masked_al — exclude central aperture from coefficient fit.
    r3 = _al_subtract_qa(sci, ref, prod_sigmas, kern_sz, pixel_mask=~center_disk)
    results["background_masked_al"] = {
        "difference": r3["difference"],
        "metrics": _al_diag_variant_metrics(r3["difference"], aperture_radius),
        "fit_info": r3["fit_info"],
    }

    # 4. rim_or_annulus_fit_al — fit only the outermost border pixels.
    rim_w = max(excl_radius, 5)
    rim_mask = np.zeros((ny, nx), dtype=bool)
    rim_mask[:rim_w, :] = True
    rim_mask[-rim_w:, :] = True
    rim_mask[:, :rim_w] = True
    rim_mask[:, -rim_w:] = True
    r4 = _al_subtract_qa(sci, ref, prod_sigmas, kern_sz, pixel_mask=rim_mask)
    results["rim_or_annulus_fit_al"] = {
        "difference": r4["difference"],
        "metrics": _al_diag_variant_metrics(r4["difference"], aperture_radius),
        "fit_info": r4["fit_info"],
    }

    # 5. reduced_basis_al — three narrower Gaussian sigmas instead of four.
    red_sigmas: tuple[float, ...] = (0.5, 1.0, 2.0)
    red_kern_sz = 2 * int(np.ceil(4.0 * max(red_sigmas))) + 1
    r5 = _al_subtract_qa(sci, ref, red_sigmas, red_kern_sz)
    results["reduced_basis_al"] = {
        "difference": r5["difference"],
        "metrics": _al_diag_variant_metrics(r5["difference"], aperture_radius),
        "fit_info": r5["fit_info"],
    }

    # 6–7. regularized_al — ridge regression with two λ scales.
    for lam in (1e-6, 1e-4):
        key = f"regularized_al_1e-{int(round(-np.log10(lam)))}"
        r = _al_subtract_qa(sci, ref, prod_sigmas, kern_sz, ridge_lambda_scale=lam)
        results[key] = {
            "difference": r["difference"],
            "metrics": _al_diag_variant_metrics(r["difference"], aperture_radius),
            "fit_info": r["fit_info"],
        }

    # 8. identity_plus_gaussians — augmented basis: raw ref + Gaussian planes + constant.
    r8 = _al_subtract_identity_plus_gaussians_qa(sci, ref, prod_sigmas, kern_sz)
    results["identity_plus_gaussians"] = {
        "difference": r8["difference"],
        "metrics": _al_diag_variant_metrics(r8["difference"], aperture_radius),
        "fit_info": r8["fit_info"],
    }

    return results


def _assign_al_interpretations(epoch_rows: list[dict]) -> None:
    """Assign interpretation strings to epoch_rows in-place."""
    snr_by_mode: dict[str, float] = {
        r["diagnostic_mode"]: r["central_aperture_snr"] for r in epoch_rows
    }
    ident_snr = snr_by_mode.get("identity_only")
    current_snr = snr_by_mode.get("current_al")

    for row in epoch_rows:
        vname = row["diagnostic_mode"]
        snr = row["central_aperture_snr"]
        cond = row.get("condition_number")
        hints: list[str] = []

        if cond is not None and cond > _AL_ILL_COND_THRESHOLD:
            hints.append(f"ill_conditioned(cond={cond:.1e})")

        if vname == "identity_only":
            hints.append(
                "clean_before_al" if abs(snr) < _DIA_HIGH_SNR_THRESHOLD
                else "residual_in_raw_diff"
            )
        elif vname == "current_al":
            if (
                ident_snr is not None
                and abs(ident_snr) < _DIA_HIGH_SNR_THRESHOLD
                and abs(snr) >= _DIA_HIGH_SNR_THRESHOLD
            ):
                hints.append("AL_introduces_residual")
            elif abs(snr) < _DIA_HIGH_SNR_THRESHOLD:
                hints.append("AL_clean")
            else:
                hints.append("AL_residual_present")
        elif vname in ("background_masked_al", "rim_or_annulus_fit_al"):
            if current_snr is not None and abs(snr) < abs(current_snr) - 1.0:
                hints.append("central_source_contaminates_fit")
            elif abs(snr) < _DIA_HIGH_SNR_THRESHOLD:
                hints.append("clean_with_mask")
            else:
                hints.append("residual_persists_with_mask")
        elif vname == "reduced_basis_al":
            if current_snr is not None and abs(snr) < abs(current_snr) - 1.0:
                hints.append("wide_basis_drives_bias")
            elif abs(snr) < _DIA_HIGH_SNR_THRESHOLD:
                hints.append("clean_with_reduced_basis")
            else:
                hints.append("residual_persists_with_reduced_basis")
        elif vname.startswith("regularized_al"):
            if current_snr is not None and abs(snr) < abs(current_snr) - 1.0:
                hints.append("regularization_helps_ill_conditioning")
            elif abs(snr) < _DIA_HIGH_SNR_THRESHOLD:
                hints.append("clean_with_regularization")
            else:
                hints.append("residual_persists_with_regularization")
        elif vname == "identity_plus_gaussians":
            if (
                current_snr is not None
                and abs(current_snr) >= _DIA_HIGH_SNR_THRESHOLD
                and abs(snr) < _DIA_HIGH_SNR_THRESHOLD
            ):
                hints.append("identity_basis_fixes_source_core_residual")
            elif abs(snr) < _DIA_HIGH_SNR_THRESHOLD:
                hints.append("clean_with_identity_basis")
            else:
                hints.append("residual_persists_with_identity_basis")

        if not hints:
            hints.append("no_strong_signal")
        row["interpretation"] = "; ".join(hints)


def _run_al_diagnostic_suite(
    capture_entries: list[dict],
    event_or_ablation: str,
    aperture_radius: int,
    out_dir: Path,
    excl_radius: int = _AL_DIAG_EXCL_RADIUS,
) -> list[dict]:
    """
    Run and save AL diagnostic ablations for all epochs of one event/ablation.

    Saves per-epoch:
      - .npy difference arrays for each variant
      - al_fit_diagnostics.json (full-fit coefficients and condition number)
      - fixed-scale PNGs when matplotlib is available
      - best_diagnostic_mode.png (variant with lowest |central_aperture_snr|)

    Returns a flat list of summary rows (one per epoch × diagnostic_mode).
    """
    if not _SCIPY:
        print("  [WARN] scipy not available; skipping AL diagnostics.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    all_epoch_results: list[dict[str, dict]] = []

    for epoch_idx, entry in enumerate(capture_entries):
        sci = entry["science"].astype(np.float64)
        ref = entry["reference"].astype(np.float64)
        post_al = entry["post_al"].astype(np.float64)

        epoch_results = _run_al_diagnostics_for_epoch(
            sci, ref, post_al, aperture_radius, excl_radius
        )
        if not epoch_results:
            continue
        all_epoch_results.append(epoch_results)

        # Full-fit diagnostics for metadata (production sigmas, no mask).
        planes = _al_convolve_ref_qa(ref, _AL_SIGMAS_PRODUCTION, _AL_KERNEL_SIZE)
        A = _build_al_design_matrix_qa(planes)
        b = sci.ravel()
        fit_diag = _al_lstsq_qa(A, b)

        # Reconstruct matched reference for central-flux measurements.
        coeffs = np.array(fit_diag["coefficients"], dtype=np.float64)
        ny, nx = sci.shape
        matched_ref = np.zeros((ny, nx), dtype=np.float64)
        for k, plane in enumerate(planes):
            matched_ref += coeffs[k] * plane
        matched_ref += coeffs[len(planes)]

        raw_id = epoch_results["identity_only"]["difference"]
        ident_snr = epoch_results["identity_only"]["metrics"]["central_aperture_snr"]
        post_snr = epoch_results["current_al"]["metrics"]["central_aperture_snr"]

        al_fit_meta: dict = {
            "coefficients": fit_diag["coefficients"],
            "condition_number": fit_diag["condition_number"],
            "rank": fit_diag["rank"],
            "singular_values": fit_diag["singular_values"],
            "raw_identity_center_snr": ident_snr,
            "post_al_center_snr": post_snr,
            "delta_post_minus_raw_center_snr": round(post_snr - ident_snr, 4),
            "raw_identity_rms": round(float(np.std(raw_id)), 6),
            "post_al_rms": round(float(np.std(post_al)), 6),
            "central_aperture_fluxes": {
                "science": round(
                    float(_central_residual_metrics(sci, aperture_radius)["central_aperture_sum"]), 4
                ),
                "reference": round(
                    float(_central_residual_metrics(ref, aperture_radius)["central_aperture_sum"]), 4
                ),
                "matched_reference": round(
                    float(_central_residual_metrics(matched_ref, aperture_radius)["central_aperture_sum"]), 4
                ),
                "raw_identity": round(
                    float(_central_residual_metrics(raw_id, aperture_radius)["central_aperture_sum"]), 4
                ),
                "post_al": round(
                    float(_central_residual_metrics(post_al, aperture_radius)["central_aperture_sum"]), 4
                ),
            },
        }

        # Save epoch directory.
        epoch_dir = out_dir / f"epoch_{epoch_idx}"
        epoch_dir.mkdir(exist_ok=True)

        for vname, vdata in epoch_results.items():
            np.save(epoch_dir / f"{vname}.npy", vdata["difference"])

        (epoch_dir / "al_fit_diagnostics.json").write_text(
            json.dumps(al_fit_meta, indent=2)
        )

        # Build per-variant summary rows for this epoch.
        epoch_rows_local: list[dict] = []
        for vname, vdata in epoch_results.items():
            m = vdata["metrics"]
            fi = vdata.get("fit_info")
            row: dict = {
                "event_or_ablation": event_or_ablation,
                "epoch": epoch_idx,
                "diagnostic_mode": vname,
                "central_aperture_snr": m["central_aperture_snr"],
                "rms": m["rms"],
                "mean": m["mean"],
                "max_abs": m["max_abs"],
                "rim_interior_std_ratio": m.get("rim_interior_std_ratio"),
                "condition_number": fi["condition_number"] if fi else None,
                "coefficients": fi["coefficients"] if fi else None,
                "identity_coefficient": fi.get("identity_coefficient") if fi else None,
                "sum_gaussian_correction_coefficients": (
                    fi.get("sum_gaussian_correction_coefficients") if fi else None
                ),
                "interpretation": "",  # filled below
            }
            epoch_rows_local.append(row)

        _assign_al_interpretations(epoch_rows_local)
        summary_rows.extend(epoch_rows_local)

        # Print per-epoch table.
        print(f"  [AL-diag] {event_or_ablation} epoch {epoch_idx}:")
        print(f"    {'Mode':<32} {'cSNR':>8}  {'RMS':>10}  {'cond':>12}  Interpretation")
        for row in epoch_rows_local:
            cond_str = (
                f"{row['condition_number']:.2e}"
                if row["condition_number"] is not None
                else "       N/A"
            )
            print(
                f"    {row['diagnostic_mode']:<32} "
                f"{row['central_aperture_snr']:>+8.3f}  "
                f"{row['rms']:>10.4f}  "
                f"{cond_str:>12}  "
                f"{row['interpretation']}"
            )

    # Fixed-scale PNGs across all epochs per variant.
    if _MPL and all_epoch_results:
        variant_names = list(all_epoch_results[0].keys())
        for vname in variant_names:
            epoch_diffs = [
                er[vname]["difference"]
                for er in all_epoch_results
                if vname in er
            ]
            if not epoch_diffs:
                continue
            scale = _robust_scale(np.concatenate([d.ravel() for d in epoch_diffs]))
            for ei, er in enumerate(all_epoch_results):
                if vname not in er:
                    continue
                epoch_dir = out_dir / f"epoch_{ei}"
                epoch_dir.mkdir(exist_ok=True)
                _save_png_fixed_scale(
                    er[vname]["difference"],
                    epoch_dir / f"{vname}.png",
                    title=f"{vname} ep{ei} [{event_or_ablation}]",
                    vmax=scale,
                )

        # Best-variant PNG per epoch: lowest |central_aperture_snr|.
        rows_by_epoch: dict[int, list[dict]] = {}
        for row in summary_rows:
            rows_by_epoch.setdefault(row["epoch"], []).append(row)

        for ei, epoch_rows in rows_by_epoch.items():
            best = min(epoch_rows, key=lambda r: abs(r["central_aperture_snr"]))
            best_vname = best["diagnostic_mode"]
            if ei < len(all_epoch_results) and best_vname in all_epoch_results[ei]:
                best_diff = all_epoch_results[ei][best_vname]["difference"]
                epoch_dir = out_dir / f"epoch_{ei}"
                epoch_dir.mkdir(exist_ok=True)
                _save_png_fixed_scale(
                    best_diff,
                    epoch_dir / "best_diagnostic_mode.png",
                    title=f"best={best_vname} cSNR={best['central_aperture_snr']:+.3f}",
                    vmax=_robust_scale(best_diff),
                )

    return summary_rows


def _save_al_diagnostic_summary(
    all_rows: list[dict],
    out_dir: Path,
    filename_stem: str = "dia_al_diagnostic_summary",
) -> None:
    """Save dia_al_diagnostic_summary.json and .csv to out_dir."""
    if not all_rows:
        return

    def _json_safe(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    json_rows = [
        {k: _json_safe(v) for k, v in row.items()}
        for row in all_rows
    ]
    json_path = out_dir / f"{filename_stem}.json"
    json_path.write_text(json.dumps(json_rows, indent=2))

    csv_fieldnames = [
        "event_or_ablation", "epoch", "diagnostic_mode",
        "central_aperture_snr", "rms", "mean", "max_abs",
        "rim_interior_std_ratio", "condition_number",
        "identity_coefficient", "sum_gaussian_correction_coefficients",
        "coefficients", "interpretation",
    ]
    flat: list[dict] = []
    for row in all_rows:
        ic = row.get("identity_coefficient")
        sgcc = row.get("sum_gaussian_correction_coefficients")
        flat.append({
            "event_or_ablation": row["event_or_ablation"],
            "epoch": row["epoch"],
            "diagnostic_mode": row["diagnostic_mode"],
            "central_aperture_snr": row["central_aperture_snr"],
            "rms": row["rms"],
            "mean": row["mean"],
            "max_abs": row["max_abs"],
            "rim_interior_std_ratio": row.get("rim_interior_std_ratio") or "",
            "condition_number": row.get("condition_number") or "",
            "identity_coefficient": ic if ic is not None else "",
            "sum_gaussian_correction_coefficients": sgcc if sgcc is not None else "",
            "coefficients": (
                json.dumps(row["coefficients"]) if row.get("coefficients") else ""
            ),
            "interpretation": row.get("interpretation", ""),
        })

    csv_path = out_dir / f"{filename_stem}.csv"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(flat)
    csv_path.write_text(buf.getvalue())

    print(f"  AL diagnostic summary JSON: {json_path}")
    print(f"  AL diagnostic summary CSV:  {csv_path}")


# ── PNG / FITS output ─────────────────────────────────────────────────────────

def _save_png(stamp: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    vmax = max(float(np.abs(stamp).max()), 1.0)
    ax.imshow(stamp, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_png_fixed_scale(
    stamp: np.ndarray, path: Path, title: str, vmax: float
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(stamp, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_fits(stamp: np.ndarray, path: Path, header_kws: dict | None = None) -> None:
    hdu = _fits.PrimaryHDU(stamp.astype(np.float32))
    if header_kws:
        for k, v in header_kws.items():
            hdu.header[k[:8]] = v
    hdu.writeto(path, overwrite=True)


# ── null ablation helpers ─────────────────────────────────────────────────────

def _aggregate_epoch_metrics(
    diffs: np.ndarray,
    aperture_radius: int,
    n_random_apertures: int,
) -> dict:
    """Aggregate null-event central residual metrics across all epochs in a stack."""
    snrs: list[float] = []
    percentiles: list[float] = []
    null_rms_list: list[float] = []
    rim_interior_ratios: list[float] = []

    for diff in diffs:
        central = _central_residual_metrics(diff, aperture_radius)
        snrs.append(central["central_aperture_snr"])
        null_rms_list.append(float(np.std(diff)))

        rim = _rim_stats(diff)
        if rim["interior_std"] > 0:
            rim_interior_ratios.append(rim["rim_std"] / rim["interior_std"])

        rand_ap = _null_random_apertures(diff, aperture_radius, n_random_apertures, seed=42)
        pct = rand_ap.get("central_aperture_percentile", float("nan"))
        if pct == pct:  # nan-safe: nan != nan
            percentiles.append(pct)

    mean_snr = float(np.mean(snrs)) if snrs else float("nan")
    max_abs_snr = float(max(abs(s) for s in snrs)) if snrs else float("nan")
    pct_min = float(min(percentiles)) if percentiles else float("nan")
    pct_max = float(max(percentiles)) if percentiles else float("nan")

    repeated_same_sign = False
    if len(snrs) >= 2:
        repeated_same_sign = all(s > 0 for s in snrs) or all(s < 0 for s in snrs)

    null_rms_mean = float(np.mean(null_rms_list)) if null_rms_list else float("nan")
    rim_ratio_mean = (
        float(np.mean(rim_interior_ratios)) if rim_interior_ratios else float("nan")
    )

    return {
        "mean_central_aperture_snr": round(mean_snr, 4),
        "max_abs_central_aperture_snr": round(max_abs_snr, 4),
        "central_aperture_percentile_range": [round(pct_min, 2), round(pct_max, 2)],
        "repeated_same_sign": repeated_same_sign,
        "null_rms_mean": round(null_rms_mean, 4),
        "rim_interior_std_ratio_mean": round(rim_ratio_mean, 4),
    }


def _build_ablation_plan(base_cfg, n_epochs: int) -> list[dict]:
    """
    Build the deterministic list of null-ablation variant specs.

    Each spec is a dict with keys:
      name, description, cfg, source_params_sequence,
      baseline_source_flux_e, neighbor_catalog, skip_reason, config_knobs.

    neighbor_catalog=None means the pipeline generates a synthetic crowded field.
    skip_reason not None means the variant is skipped before simulation.
    """
    specs: list[dict] = []
    base_flux = _BASELINE_SOURCE_FLUX_E
    base_params = [{"flux_e": base_flux} for _ in range(n_epochs)]

    # A. baseline_science — mirrors the existing null event exactly.
    specs.append({
        "name": "baseline_science",
        "description": (
            "Current config: detector_matched reference, per-epoch jitter, crowded field, "
            "source present at baseline flux in both reference and science."
        ),
        "cfg": base_cfg,
        "source_params_sequence": base_params,
        "baseline_source_flux_e": base_flux,
        "neighbor_catalog": None,
        "skip_reason": None,
        "config_knobs": {
            "reference_processing": base_cfg.dia.reference_processing,
            "reference_jitter_mode": base_cfg.dia.reference_jitter_mode,
            "jitter_rms_mas": base_cfg.psf.jitter_rms_mas,
            "crowded_field": True,
        },
    })

    # B. no_source — no source in science or reference; tests if residual is source-specific.
    specs.append({
        "name": "no_source",
        "description": (
            "No source in science (flux_e=0.0) and no reference source "
            "(baseline_source_flux_e=None). Only neighbors + noise. "
            "Goal: verify if central residual requires a source to be present."
        ),
        "cfg": base_cfg,
        "source_params_sequence": [{"flux_e": 0.0} for _ in range(n_epochs)],
        "baseline_source_flux_e": None,
        "neighbor_catalog": None,
        "skip_reason": None,
        "config_knobs": {
            "reference_processing": base_cfg.dia.reference_processing,
            "reference_jitter_mode": base_cfg.dia.reference_jitter_mode,
            "jitter_rms_mas": base_cfg.psf.jitter_rms_mas,
            "crowded_field": True,
            "source_present": False,
        },
    })

    # C. source_only_no_crowding — empty neighbor catalog; isolates source subtraction.
    if _PD:
        empty_catalog = _pd.DataFrame({
            "x_pix": _pd.array([], dtype=float),
            "y_pix": _pd.array([], dtype=float),
            "flux_e": _pd.array([], dtype=float),
            "mag_w146": _pd.array([], dtype=float),
        })
        specs.append({
            "name": "source_only_no_crowding",
            "description": (
                "Source present in both reference and science at baseline flux; "
                "empty neighbor catalog (no crowded-field neighbors). "
                "Goal: isolate source DIA residual from crowding."
            ),
            "cfg": base_cfg,
            "source_params_sequence": base_params,
            "baseline_source_flux_e": base_flux,
            "neighbor_catalog": empty_catalog,
            "skip_reason": None,
            "config_knobs": {
                "reference_processing": base_cfg.dia.reference_processing,
                "reference_jitter_mode": base_cfg.dia.reference_jitter_mode,
                "jitter_rms_mas": base_cfg.psf.jitter_rms_mas,
                "crowded_field": False,
            },
        })
    else:
        specs.append({
            "name": "source_only_no_crowding",
            "description": "Source only, empty neighbor catalog.",
            "cfg": base_cfg,
            "source_params_sequence": base_params,
            "baseline_source_flux_e": base_flux,
            "neighbor_catalog": None,
            "skip_reason": "pandas not available; cannot construct empty neighbor_catalog DataFrame",
            "config_knobs": {},
        })

    # D. shared_reference_jitter — reference_jitter_mode=shared.
    cfg_shared = base_cfg.model_copy(
        update={"dia": base_cfg.dia.model_copy(update={"reference_jitter_mode": "shared"})}
    )
    specs.append({
        "name": "shared_reference_jitter",
        "description": (
            "reference_jitter_mode=shared: every reference epoch reuses the same jitter "
            "seed, suppressing jitter averaging in the template. "
            "Goal: test whether PSF jitter ensemble averaging drives central residual."
        ),
        "cfg": cfg_shared,
        "source_params_sequence": base_params,
        "baseline_source_flux_e": base_flux,
        "neighbor_catalog": None,
        "skip_reason": None,
        "config_knobs": {
            "reference_processing": cfg_shared.dia.reference_processing,
            "reference_jitter_mode": "shared",
            "jitter_rms_mas": cfg_shared.psf.jitter_rms_mas,
            "crowded_field": True,
        },
    })

    # E. ideal_gaussian_reference — reference_processing=ideal_gaussian.
    cfg_ideal = base_cfg.model_copy(
        update={"dia": base_cfg.dia.model_copy(update={"reference_processing": "ideal_gaussian"})}
    )
    specs.append({
        "name": "ideal_gaussian_reference",
        "description": (
            "reference_processing=ideal_gaussian: reference epochs skip the detector chain. "
            "Goal: compare full detector-matched reference vs cheap scalar-noise reference."
        ),
        "cfg": cfg_ideal,
        "source_params_sequence": base_params,
        "baseline_source_flux_e": base_flux,
        "neighbor_catalog": None,
        "skip_reason": None,
        "config_knobs": {
            "reference_processing": "ideal_gaussian",
            "reference_jitter_mode": cfg_ideal.dia.reference_jitter_mode,
            "jitter_rms_mas": cfg_ideal.psf.jitter_rms_mas,
            "crowded_field": True,
        },
    })

    # F. zero_jitter — psf.jitter_rms_mas=0.0.
    try:
        cfg_zero_jitter = base_cfg.model_copy(
            update={"psf": base_cfg.psf.model_copy(update={"jitter_rms_mas": 0.0})}
        )
        zero_jitter_skip = None
    except Exception as exc:
        cfg_zero_jitter = base_cfg
        zero_jitter_skip = f"model_copy(jitter_rms_mas=0.0) failed: {type(exc).__name__}: {exc}"

    specs.append({
        "name": "zero_jitter",
        "description": (
            "psf.jitter_rms_mas=0.0: no line-of-sight jitter in either reference or science. "
            "Goal: test whether PSF jitter / DIA kernel mismatch drives central residual."
        ),
        "cfg": cfg_zero_jitter,
        "source_params_sequence": base_params,
        "baseline_source_flux_e": base_flux,
        "neighbor_catalog": None,
        "skip_reason": zero_jitter_skip,
        "config_knobs": {
            "reference_processing": cfg_zero_jitter.dia.reference_processing,
            "reference_jitter_mode": cfg_zero_jitter.dia.reference_jitter_mode,
            "jitter_rms_mas": 0.0,
            "crowded_field": True,
        },
    })

    # G. centered_source_position_check — explicit centroid_offset_pix=(0.0, 0.0).
    centered_params = [
        {"flux_e": base_flux, "centroid_offset_pix": (0.0, 0.0)}
        for _ in range(n_epochs)
    ]
    specs.append({
        "name": "centered_source_position_check",
        "description": (
            "Science centroid_offset_pix=(0.0, 0.0) explicitly set per epoch; "
            "expected reference centroid is also (0.0, 0.0). "
            "Goal: catch any reference/science centroid mismatch."
        ),
        "cfg": base_cfg,
        "source_params_sequence": centered_params,
        "baseline_source_flux_e": base_flux,
        "neighbor_catalog": None,
        "skip_reason": None,
        "config_knobs": {
            "reference_processing": base_cfg.dia.reference_processing,
            "reference_jitter_mode": base_cfg.dia.reference_jitter_mode,
            "jitter_rms_mas": base_cfg.psf.jitter_rms_mas,
            "crowded_field": True,
            "centroid_offset_pix": [0.0, 0.0],
        },
    })

    return specs


def _run_single_ablation(
    variant_name: str,
    cfg_variant,
    source_params_sequence: list[dict],
    baseline_source_flux_e: float | None,
    neighbor_catalog,
    timestamps_mjd: np.ndarray,
    backgrounds: list[float],
    out_dir: Path,
    aperture_radius: int,
    n_random_apertures: int,
    master_seed: int,
    config_knobs: dict,
    capture_dia_inputs: bool = False,
    dia_diagnostic_modes: bool = False,
) -> dict:
    """Run one ablation variant. Returns a summary-row dict."""
    variant_dir = out_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Building SceneSimulator ...")
        sim = SceneSimulator(cfg_variant, master_seed=master_seed)

        dia_capture: list[dict] = []
        ctx_mgr = (
            _DiaCaptureContext(dia_capture)
            if capture_dia_inputs
            else contextlib.nullcontext()
        )

        with ctx_mgr:
            result = sim.simulate_event(
                event_id=f"ablation_{variant_name}",
                source_params_sequence=source_params_sequence,
                timestamps_mjd=timestamps_mjd,
                backgrounds_e_per_s=backgrounds,
                neighbor_catalog=neighbor_catalog,
                baseline_source_flux_e=baseline_source_flux_e,
            )

        diffs = result.difference_stamps
        np.save(variant_dir / "diff_stamps.npy", diffs)

        agg = _aggregate_epoch_metrics(diffs, aperture_radius, n_random_apertures)

        # Flux / centroid invariants
        science_fluxes = [float(p["flux_e"]) for p in source_params_sequence]
        science_centroids = [
            list(p.get("centroid_offset_pix", (0.0, 0.0)))
            for p in source_params_sequence
        ]
        all_fluxes_equal = baseline_source_flux_e is not None and all(
            abs(f - baseline_source_flux_e) < 1e-9 for f in science_fluxes
        )
        all_centroids_equal = all(c == [0.0, 0.0] for c in science_centroids)

        if _MPL:
            for i, diff in enumerate(diffs):
                _save_png(
                    diff,
                    variant_dir / f"epoch_{i:02d}_diff.png",
                    title=f"{variant_name} epoch {i}",
                )

        # DIA-input capture: save arrays + compute per-epoch metrics.
        dia_epoch_metrics: list[dict] = []
        al_diag_rows: list[dict] = []
        if capture_dia_inputs and dia_capture:
            sci_sz = cfg_variant.dia.science_stamp_size
            ctx_sz = cfg_variant.dia.context_stamp_size
            dia_dir = variant_dir / "dia_inputs"
            _save_dia_capture_arrays(dia_capture, dia_dir, sci_sz, ctx_sz)
            for entry in dia_capture:
                dia_epoch_metrics.append(
                    _compute_dia_input_epoch_metrics(entry, aperture_radius)
                )
            _print_dia_capture_summary(variant_name, dia_epoch_metrics)

            if dia_diagnostic_modes:
                al_diag_rows = _run_al_diagnostic_suite(
                    capture_entries=dia_capture,
                    event_or_ablation=variant_name,
                    aperture_radius=aperture_radius,
                    out_dir=variant_dir / "al_diagnostics",
                )

        meta: dict = {
            "variant": variant_name,
            "status": "ran",
            "baseline_source_flux_e": baseline_source_flux_e,
            "science_flux_e_per_epoch": science_fluxes,
            "science_centroid_offset_pix_per_epoch": science_centroids,
            "expected_reference_centroid_offset": [0.0, 0.0],
            "all_science_fluxes_equal_baseline": all_fluxes_equal,
            "all_science_centroids_equal_reference": all_centroids_equal,
            "config_knobs": config_knobs,
            "aggregated_metrics": agg,
        }
        if capture_dia_inputs:
            meta["dia_input_capture"] = {
                "enabled": True,
                "epoch_metrics": dia_epoch_metrics,
            }
        (variant_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        return {
            "variant": variant_name,
            "status": "ran",
            "skip_reason": None,
            **agg,
            "config_knobs": config_knobs,
            "al_diag_rows": al_diag_rows,
        }

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"  [ERROR] {reason}")
        (variant_dir / "metadata.json").write_text(
            json.dumps({"variant": variant_name, "status": "error", "error": reason}, indent=2)
        )
        return {
            "variant": variant_name,
            "status": "error",
            "skip_reason": reason,
            "mean_central_aperture_snr": None,
            "max_abs_central_aperture_snr": None,
            "central_aperture_percentile_range": [None, None],
            "repeated_same_sign": None,
            "null_rms_mean": None,
            "rim_interior_std_ratio_mean": None,
            "config_knobs": config_knobs,
            "al_diag_rows": [],
        }


def _print_ablation_summary(summary_rows: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print("[NULL ABLATION SUMMARY TABLE]")
    hdr = (
        f"  {'Variant':<35} {'Status':<8} {'MeanSNR':>8} "
        f"{'MaxAbsSNR':>10} {'PctRange [min,max]':>20} {'SameSign':>9}"
    )
    print(hdr)
    print("  " + "─" * 92)
    for row in summary_rows:
        name = row["variant"][:34]
        status = row["status"]
        if status in ("skipped", "error"):
            reason = (row.get("skip_reason") or "")[:40]
            print(f"  {name:<35} {status:<8}  -- {reason}")
            continue
        mean_snr = row.get("mean_central_aperture_snr")
        max_abs = row.get("max_abs_central_aperture_snr")
        pct = row.get("central_aperture_percentile_range", [None, None])
        same = row.get("repeated_same_sign")
        snr_str = f"{mean_snr:+.2f}" if mean_snr is not None else "  N/A"
        max_str = f"{max_abs:.2f}" if max_abs is not None else "   N/A"
        pct_str = (
            f"[{pct[0]:.1f},{pct[1]:.1f}]"
            if pct[0] is not None
            else "         N/A"
        )
        print(
            f"  {name:<35} {status:<8} {snr_str:>8} {max_str:>10} "
            f"{pct_str:>20} {str(same):>9}"
        )


def _print_interpretation_hints(summary_rows: list[dict]) -> None:
    print(f"\n{'─' * 60}")
    print("[INTERPRETATION HINTS]")

    _IMPROVE_THRESH = 2.0  # SNR reduction considered "greatly improved"
    _PASS_THRESH = 1.5     # |SNR| below this is "passes"

    def snr(name: str) -> float | None:
        for r in summary_rows:
            if r["variant"] == name and r["status"] == "ran":
                v = r.get("mean_central_aperture_snr")
                return float(v) if v is not None else None
        return None

    def same_sign(name: str) -> bool | None:
        for r in summary_rows:
            if r["variant"] == name and r["status"] == "ran":
                return r.get("repeated_same_sign")
        return None

    base = snr("baseline_science")
    no_src = snr("no_source")
    src_only = snr("source_only_no_crowding")
    shared = snr("shared_reference_jitter")
    zero_j = snr("zero_jitter")
    ideal_r = snr("ideal_gaussian_reference")

    hints_printed = False

    if no_src is not None and src_only is not None:
        if abs(no_src) < _PASS_THRESH and abs(src_only) > _IMPROVE_THRESH:
            print(
                "  -> no_source passes but source_only_no_crowding fails:\n"
                "     Source/reference flux or centroid mismatch is likely."
            )
            hints_printed = True

    if base is not None and shared is not None:
        if abs(base) - abs(shared) > _IMPROVE_THRESH:
            print(
                "  -> shared_reference_jitter greatly reduces central residual:\n"
                "     Reference/science PSF mismatch from per-epoch jitter averaging is likely."
            )
            hints_printed = True

    if base is not None and zero_j is not None:
        if abs(base) - abs(zero_j) > _IMPROVE_THRESH:
            print(
                "  -> zero_jitter greatly reduces central residual:\n"
                "     PSF jitter / DIA kernel matching is likely driving the residual."
            )
            hints_printed = True

    if base is not None and ideal_r is not None:
        delta = abs(base) - abs(ideal_r)
        if abs(delta) > _IMPROVE_THRESH:
            direction = "reduces" if delta > 0 else "worsens"
            print(
                f"  -> ideal_gaussian_reference {direction} central residual:\n"
                "     Detector reference processing contributes to the null residual."
            )
            hints_printed = True

    ran = [r for r in summary_rows if r["status"] == "ran"]
    if ran and all(
        abs(r.get("mean_central_aperture_snr") or 0.0) > _IMPROVE_THRESH
        and r.get("repeated_same_sign")
        for r in ran
    ):
        print(
            "  -> All ran variants show repeated same-sign residual:\n"
            "     DIA kernel / model limitation or source rendering mismatch likely."
        )
        hints_printed = True

    if not hints_printed:
        print(
            "  No strong differentiation between ablation variants.\n"
            "  Inspect raw SNRs in the summary table for subtle differences."
        )


def _save_ablation_csv(summary_rows: list[dict], path: Path) -> None:
    fieldnames = [
        "variant", "status", "skip_reason",
        "mean_central_aperture_snr", "max_abs_central_aperture_snr",
        "central_aperture_percentile_min", "central_aperture_percentile_max",
        "repeated_same_sign", "null_rms_mean", "rim_interior_std_ratio_mean",
        "reference_processing", "reference_jitter_mode", "jitter_rms_mas", "crowded_field",
    ]
    flat_rows = []
    for row in summary_rows:
        pct = row.get("central_aperture_percentile_range", [None, None])
        knobs = row.get("config_knobs", {})
        flat_rows.append({
            "variant": row["variant"],
            "status": row["status"],
            "skip_reason": row.get("skip_reason") or "",
            "mean_central_aperture_snr": row.get("mean_central_aperture_snr", ""),
            "max_abs_central_aperture_snr": row.get("max_abs_central_aperture_snr", ""),
            "central_aperture_percentile_min": pct[0] if pct[0] is not None else "",
            "central_aperture_percentile_max": pct[1] if pct[1] is not None else "",
            "repeated_same_sign": "" if row.get("repeated_same_sign") is None else row["repeated_same_sign"],
            "null_rms_mean": row.get("null_rms_mean", ""),
            "rim_interior_std_ratio_mean": row.get("rim_interior_std_ratio_mean", ""),
            "reference_processing": knobs.get("reference_processing", ""),
            "reference_jitter_mode": knobs.get("reference_jitter_mode", ""),
            "jitter_rms_mas": knobs.get("jitter_rms_mas", ""),
            "crowded_field": knobs.get("crowded_field", ""),
        })
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flat_rows)
    path.write_text(buf.getvalue())


def run_null_ablations(
    cfg,
    output_dir: Path,
    n_epochs: int,
    master_seed: int,
    aperture_radius: int = _DEFAULT_APERTURE_RADIUS,
    n_random_apertures: int = _DEFAULT_RANDOM_APERTURES,
    capture_dia_inputs: bool = False,
    dia_diagnostic_modes: bool = False,
) -> None:
    """
    Run the null-ablation variant suite and save per-variant outputs plus a
    summary JSON/CSV.  Activated by --run-null-ablations.
    """
    abl_dir = output_dir / _ABLATION_SUBDIR
    abl_dir.mkdir(parents=True, exist_ok=True)

    timestamps_mjd = np.array([60000.0 + i for i in range(n_epochs)])
    backgrounds = [_SKY_BG_E_PER_S] * n_epochs

    specs = _build_ablation_plan(cfg, n_epochs)

    print(f"\n{'=' * 60}")
    print(f"[NULL ABLATIONS]  {len(specs)} variants  master_seed={master_seed}")
    print(f"  Output dir: {abl_dir.resolve()}")

    summary_rows: list[dict] = []
    all_al_diag_rows: list[dict] = []

    for spec in specs:
        name = spec["name"]
        print(f"\n{'─' * 60}")
        print(f"Ablation variant: {name}")
        print(f"  {spec['description']}")

        skip = spec["skip_reason"]
        if skip is not None:
            print(f"  [SKIP] {skip}")
            variant_dir = abl_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "metadata.json").write_text(
                json.dumps(
                    {"variant": name, "status": "skipped", "skip_reason": skip},
                    indent=2,
                )
            )
            summary_rows.append({
                "variant": name,
                "status": "skipped",
                "skip_reason": skip,
                "mean_central_aperture_snr": None,
                "max_abs_central_aperture_snr": None,
                "central_aperture_percentile_range": [None, None],
                "repeated_same_sign": None,
                "null_rms_mean": None,
                "rim_interior_std_ratio_mean": None,
                "config_knobs": spec["config_knobs"],
            })
            continue

        # Log flux and centroid invariants before running
        science_fluxes = [p["flux_e"] for p in spec["source_params_sequence"]]
        science_centroids = [
            list(p.get("centroid_offset_pix", (0.0, 0.0)))
            for p in spec["source_params_sequence"]
        ]
        bsf = spec["baseline_source_flux_e"]
        all_flux_eq = bsf is not None and all(abs(f - bsf) < 1e-9 for f in science_fluxes)
        all_ctr_eq = all(c == [0.0, 0.0] for c in science_centroids)
        print(
            f"  baseline_source_flux_e={bsf}  "
            f"science_fluxes_match_baseline={all_flux_eq}  "
            f"centroids_at_zero={all_ctr_eq}"
        )

        row = _run_single_ablation(
            variant_name=name,
            cfg_variant=spec["cfg"],
            source_params_sequence=spec["source_params_sequence"],
            baseline_source_flux_e=spec["baseline_source_flux_e"],
            neighbor_catalog=spec["neighbor_catalog"],
            timestamps_mjd=timestamps_mjd,
            backgrounds=backgrounds,
            out_dir=abl_dir,
            aperture_radius=aperture_radius,
            n_random_apertures=n_random_apertures,
            master_seed=master_seed,
            config_knobs=spec["config_knobs"],
            capture_dia_inputs=capture_dia_inputs,
            dia_diagnostic_modes=dia_diagnostic_modes,
        )
        all_al_diag_rows.extend(row.get("al_diag_rows", []))

        if row["status"] == "ran":
            pct = row["central_aperture_percentile_range"]
            print(
                f"  mean_snr={row['mean_central_aperture_snr']:+.3f}  "
                f"max_abs_snr={row['max_abs_central_aperture_snr']:.3f}  "
                f"pct=[{pct[0]:.1f},{pct[1]:.1f}]  "
                f"same_sign={row['repeated_same_sign']}"
            )

        summary_rows.append(row)

    _print_ablation_summary(summary_rows)
    _print_interpretation_hints(summary_rows)

    # Strip non-serialisable al_diag_rows from summary before JSON dump.
    summary_rows_clean = [
        {k: v for k, v in row.items() if k != "al_diag_rows"}
        for row in summary_rows
    ]
    summary_json = abl_dir / "null_ablation_summary.json"
    summary_json.write_text(json.dumps(summary_rows_clean, indent=2))

    summary_csv = abl_dir / "null_ablation_summary.csv"
    try:
        _save_ablation_csv(summary_rows_clean, summary_csv)
        print(f"\n  Ablation summary JSON: {summary_json}")
        print(f"  Ablation summary CSV:  {summary_csv}")
    except Exception as exc:
        print(f"\n  Ablation summary JSON: {summary_json}")
        print(f"  [WARN] CSV save failed: {exc}")

    if dia_diagnostic_modes and all_al_diag_rows:
        print(f"\n{'─' * 60}")
        print("[AL DIAGNOSTIC SUMMARY — ABLATION VARIANTS]")
        _save_al_diagnostic_summary(all_al_diag_rows, abl_dir)


# ── pilot batch helpers ───────────────────────────────────────────────────────

def _compute_pilot_epoch_row(
    event_id: str,
    event_type: str,
    epoch: int,
    diff: np.ndarray,
    sat: np.ndarray,
    cr: np.ndarray,
    prov,
    cfg,
    aperture_radius: int,
    n_random_apertures: int,
) -> dict:
    """Compute a flat summary row for one event/epoch (no qa_flags; caller adds those)."""
    stats = _stamp_summary(diff)
    rim = _rim_stats(diff)
    central = _central_residual_metrics(diff, aperture_radius)
    has_nonfinite = not bool(np.all(np.isfinite(diff)))

    central_aperture_percentile = None
    if event_type == "null":
        rand_ap = _null_random_apertures(diff, aperture_radius, n_random_apertures, seed=42)
        pct = rand_ap.get("central_aperture_percentile", float("nan"))
        if pct == pct:  # nan-safe
            central_aperture_percentile = round(float(pct), 2)

    sat_frac = round(float(sat.mean()), 6) if sat.size > 0 else 0.0
    cr_frac = round(float(cr.mean()), 6) if cr.size > 0 else 0.0
    dia_method = getattr(prov, "dia_method", None) if prov is not None else None

    return {
        "event_id": event_id,
        "event_type": event_type,
        "epoch": epoch,
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "p5": stats["p5"],
        "p50": stats["p50"],
        "p95": stats["p95"],
        "central_aperture_snr": central["central_aperture_snr"],
        "central_aperture_percentile": central_aperture_percentile,
        "rim_interior_std_ratio": rim["rim_interior_std_ratio"],
        "central_signal_dominant": (
            event_type == "lensed"
            and rim["rim_interior_std_ratio"] == rim["rim_interior_std_ratio"]  # nan-safe
            and rim["rim_interior_std_ratio"] < _DEFAULT_RIM_RATIO_LOW
            and abs(central["central_aperture_snr"]) > _DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT
        ),
        "has_nonfinite": has_nonfinite,
        "saturation_pixel_fraction": sat_frac,
        "cr_pixel_fraction": cr_frac,
        "dia_method": dia_method,
        "reference_processing": cfg.dia.reference_processing,
        "reference_jitter_mode": cfg.dia.reference_jitter_mode,
        "psf_fov_native_pixels": cfg.psf.fov_native_pixels,
        "oversample": cfg.psf.oversample,
        "n_wavelengths": cfg.psf.n_wavelengths,
    }


def _central_signal_dominant(
    row: dict,
    rim_ratio_low: float = _DEFAULT_RIM_RATIO_LOW,
    snr_thresh: float = _DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT,
) -> bool:
    """Return True when a lensed epoch has strong central signal suppressing rim std.

    Low rim/interior ratio in a lensed event is expected physics (central source
    dominates interior std), not a rim artifact.  Null events always return False.
    """
    if row.get("event_type") != "lensed":
        return False
    rim = row.get("rim_interior_std_ratio", float("nan"))
    snr = row.get("central_aperture_snr", 0.0)
    return rim == rim and rim < rim_ratio_low and abs(snr) > snr_thresh


def _pilot_qa_flags(
    row: dict,
    null_central_snr_warn: float = _DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT,
    rim_ratio_low: float = _DEFAULT_RIM_RATIO_LOW,
    rim_ratio_high: float = _DEFAULT_RIM_RATIO_HIGH,
    max_saturation_fraction: float = 0.0,
    max_cr_fraction: float | None = None,
    expected_dia_method: str = _PILOT_EXPECTED_DIA_METHOD,
    expected_reference_processing: str = _PILOT_EXPECTED_REFERENCE_PROCESSING,
    expected_reference_jitter_mode: str = _PILOT_EXPECTED_REFERENCE_JITTER_MODE,
) -> list[str]:
    """Return QA flag strings for one pilot batch row.

    Rim-artifact logic is event-type-aware:
    - null events: flag if rim/interior ratio is outside [rim_ratio_low, rim_ratio_high].
    - lensed events: flag only if rim/interior ratio is above rim_ratio_high.
      A low ratio for a lensed event indicates central signal dominance, not an
      artifact; use central_signal_dominant to track that separately.
    """
    flags: list[str] = []

    if row.get("has_nonfinite"):
        flags.append("nonfinite_values")

    if row.get("event_type") == "null":
        snr = row.get("central_aperture_snr", 0.0)
        if abs(snr) > null_central_snr_warn:
            flags.append("null_central_residual_high")

    rim_ratio = row.get("rim_interior_std_ratio", float("nan"))
    if rim_ratio == rim_ratio:  # nan-safe
        is_lensed = row.get("event_type") == "lensed"
        if is_lensed:
            # Low rim/interior is expected for lensed events; only flag abnormally high.
            if rim_ratio > rim_ratio_high:
                flags.append("rim_artifact_suspected")
        else:
            if rim_ratio < rim_ratio_low or rim_ratio > rim_ratio_high:
                flags.append("rim_artifact_suspected")

    if row.get("saturation_pixel_fraction", 0.0) > max_saturation_fraction:
        flags.append("unexpected_saturation")

    if max_cr_fraction is not None and row.get("cr_pixel_fraction", 0.0) > max_cr_fraction:
        flags.append("unexpected_cr")

    if row.get("dia_method") != expected_dia_method:
        flags.append("unexpected_dia_method")

    if row.get("reference_processing") != expected_reference_processing:
        flags.append("unexpected_reference_processing")

    if row.get("reference_jitter_mode") != expected_reference_jitter_mode:
        flags.append("unexpected_reference_jitter_mode")

    return flags


def _save_pilot_summary(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Write pilot_summary.json and pilot_summary.csv to output_dir."""
    # JSON: rows as-is (qa_flags as list)
    json_path = output_dir / "pilot_summary.json"
    json_path.write_text(json.dumps(rows, indent=2))

    # CSV: qa_flags as pipe-separated string
    flat: list[dict] = []
    for row in rows:
        f = dict(row)
        flags = f.get("qa_flags", [])
        f["qa_flags"] = "|".join(flags) if isinstance(flags, list) else str(flags or "")
        flat.append(f)

    csv_path = output_dir / "pilot_summary.csv"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_PILOT_CSV_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat)
    csv_path.write_text(buf.getvalue())

    return json_path, csv_path


def _print_pilot_console_summary(rows: list[dict], thresholds: dict) -> None:
    """Print top-outlier tables and overall pass/fail result to stdout."""
    n_events = len(set(r["event_id"] for r in rows))
    n_total = len(rows)
    flagged = [r for r in rows if r.get("qa_flags")]
    n_flagged = len(flagged)

    print(f"\n{'=' * 70}")
    print("[PILOT BATCH QA SUMMARY]")
    print(f"  Total events  : {n_events}")
    print(f"  Total rows    : {n_total} (events × epochs)")
    print(f"  Flagged rows  : {n_flagged} / {n_total}")

    # Science config / provenance checks
    dia_methods = set(r.get("dia_method") for r in rows)
    ref_procs = set(r.get("reference_processing") for r in rows)
    ref_modes = set(r.get("reference_jitter_mode") for r in rows)

    def _ok(cond: bool, msg: str) -> str:
        return f"  {'[OK]  ' if cond else '[WARN]'} {msg}"

    print("\n  [CONFIG/PROVENANCE CHECKS]")
    print(_ok(
        dia_methods == {_PILOT_EXPECTED_DIA_METHOD},
        f"dia_method = {dia_methods}  (expected: {{'{_PILOT_EXPECTED_DIA_METHOD}'}})",
    ))
    print(_ok(
        ref_procs == {_PILOT_EXPECTED_REFERENCE_PROCESSING},
        f"reference_processing = {ref_procs}  "
        f"(expected: {{'{_PILOT_EXPECTED_REFERENCE_PROCESSING}'}})",
    ))
    print(_ok(
        ref_modes == {_PILOT_EXPECTED_REFERENCE_JITTER_MODE},
        f"reference_jitter_mode = {ref_modes}  "
        f"(expected: {{'{_PILOT_EXPECTED_REFERENCE_JITTER_MODE}'}})",
    ))

    # Top 10 worst null central |SNR|
    null_rows = [r for r in rows if r.get("event_type") == "null"]
    if null_rows:
        top_snr = sorted(
            null_rows,
            key=lambda r: abs(r.get("central_aperture_snr", 0.0)),
            reverse=True,
        )[:10]
        thresh = thresholds.get("null_central_snr_warn", _DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT)
        print(f"\n  [TOP 10 WORST NULL CENTRAL |SNR|] (threshold={thresh})")
        print(f"  {'event_id':<20} {'ep':>3} {'cSNR':>10} {'pct':>8}  flags")
        for r in top_snr:
            pct_v = r.get("central_aperture_percentile")
            pct_s = f"{pct_v:5.1f}%" if pct_v is not None else "    N/A"
            flags_s = "|".join(r.get("qa_flags") or []) or "--"
            print(
                f"  {r['event_id']:<20} {r['epoch']:>3} "
                f"{r.get('central_aperture_snr', float('nan')):>+10.3f} "
                f"{pct_s:>8}  {flags_s}"
            )

    # Top 10 worst null rim/interior ratio (true artifact check)
    low = thresholds.get("rim_ratio_low", _DEFAULT_RIM_RATIO_LOW)
    high = thresholds.get("rim_ratio_high", _DEFAULT_RIM_RATIO_HIGH)
    null_rim_rows = [
        r for r in rows
        if r.get("event_type") == "null"
        and r.get("rim_interior_std_ratio") == r.get("rim_interior_std_ratio")
    ]
    if null_rim_rows:
        top_null_rim = sorted(
            null_rim_rows,
            key=lambda r: abs(r.get("rim_interior_std_ratio", 1.0) - 1.0),
            reverse=True,
        )[:10]
        print(f"\n  [TOP 10 WORST NULL RIM/INTERIOR RATIO] (expected: [{low}, {high}])")
        print(f"  {'event_id':<20} {'ep':>3} {'rim_ratio':>10}  flags")
        for r in top_null_rim:
            flags_s = "|".join(r.get("qa_flags") or []) or "--"
            print(
                f"  {r['event_id']:<20} {r['epoch']:>3} "
                f"{r.get('rim_interior_std_ratio', float('nan')):>10.4f}  {flags_s}"
            )

    # Lensed epochs with central signal dominance (informational, not a failure)
    lensed_csd_rows = [r for r in rows if r.get("central_signal_dominant")]
    if lensed_csd_rows:
        print(
            f"\n  [LENSED CENTRAL SIGNAL DOMINANT] "
            f"(rim/interior < {low} with high cSNR — expected physics, not artifact)"
        )
        print(f"  {'event_id':<20} {'ep':>3} {'rim_ratio':>10} {'cSNR':>10}  note")
        for r in lensed_csd_rows:
            print(
                f"  {r['event_id']:<20} {r['epoch']:>3} "
                f"{r.get('rim_interior_std_ratio', float('nan')):>10.4f} "
                f"{r.get('central_aperture_snr', float('nan')):>+10.3f}  central_signal_dominant"
            )

    # Rows with nonfinite / saturation / CR flags
    for flag_key, label in [
        ("nonfinite_values", "NONFINITE VALUES"),
        ("unexpected_saturation", "UNEXPECTED SATURATION"),
        ("unexpected_cr", "UNEXPECTED CR"),
    ]:
        affected = [r for r in rows if flag_key in (r.get("qa_flags") or [])]
        if affected:
            print(f"\n  [FLAGGED: {label}]")
            for r in affected:
                print(
                    f"  event_id={r['event_id']}  epoch={r['epoch']}  "
                    f"sat={r.get('saturation_pixel_fraction', 0.0):.4f}  "
                    f"cr={r.get('cr_pixel_fraction', 0.0):.4f}  "
                    f"flags={r.get('qa_flags')}"
                )

    # Rows with unexpected config/provenance flags
    cfg_flags = {
        "unexpected_dia_method",
        "unexpected_reference_processing",
        "unexpected_reference_jitter_mode",
    }
    cfg_bad = [r for r in rows if cfg_flags & set(r.get("qa_flags") or [])]
    if cfg_bad:
        print("\n  [FLAGGED: UNEXPECTED CONFIG/PROVENANCE]")
        for r in cfg_bad:
            bad = [f for f in (r.get("qa_flags") or []) if f in cfg_flags]
            print(
                f"  event_id={r['event_id']}  epoch={r['epoch']}  "
                f"dia_method={r.get('dia_method')!r}  "
                f"ref_proc={r.get('reference_processing')!r}  "
                f"flags={bad}"
            )

    print(f"\n  {'─' * 60}")
    if n_flagged == 0:
        print("  [BATCH QA RESULT] PASS — no flagged rows.")
    else:
        n_ev_flagged = len(set(r["event_id"] for r in flagged))
        print(
            f"  [BATCH QA RESULT] FAIL — {n_flagged} flagged rows "
            f"across {n_ev_flagged} event(s)."
        )
    print(f"{'=' * 70}")


def run_pilot_batch(
    config_path: Path,
    output_dir: Path,
    batch_size: int = _DEFAULT_PILOT_BATCH_SIZE,
    n_epochs: int = _DEFAULT_N_EPOCHS,
    master_seed: int = _DEFAULT_MASTER_SEED,
    aperture_radius: int = _DEFAULT_APERTURE_RADIUS,
    n_random_apertures: int = _DEFAULT_RANDOM_APERTURES,
    null_central_snr_warn: float = _DEFAULT_NULL_CENTRAL_SNR_WARN_PILOT,
    rim_ratio_low: float = _DEFAULT_RIM_RATIO_LOW,
    rim_ratio_high: float = _DEFAULT_RIM_RATIO_HIGH,
    max_nonfinite_allowed: int = 0,
    max_saturation_fraction: float = 0.0,
    max_cr_fraction: float | None = None,
) -> None:
    """
    Run a pilot batch of ``batch_size`` deterministic events and write aggregate
    QA summaries to ``output_dir``.

    Events alternate null/lensed: even indices (0, 2, …) are null (A=1),
    odd indices (1, 3, …) are lensed (Gaussian bell, peak A=5).  All randomness
    is derived from (master_seed, event_id) so results are fully reproducible.

    Outputs per event:
        diff_stamps.npy, sat_stamps.npy, cr_stamps.npy, metadata.json,
        optional PNG previews.

    Aggregate outputs:
        pilot_summary.json, pilot_summary.csv, pilot_run_config.json.
    """
    print(f"Loading config: {config_path}")
    cfg = load_simulation_config(config_path)
    config_sha = get_simulation_config_sha256(cfg)

    print(f"\n[CONFIG] SHA-256: {config_sha[:16]}...")
    print(f"  reference_processing   = {cfg.dia.reference_processing}")
    print(f"  reference_jitter_mode  = {cfg.dia.reference_jitter_mode}")
    print(f"  dia.subtraction_method = {cfg.dia.subtraction_method}")
    print(f"  psf.fov_native_pixels  = {cfg.psf.fov_native_pixels}")
    print(f"  psf.oversample         = {cfg.psf.oversample}")
    print(f"  psf.n_wavelengths      = {cfg.psf.n_wavelengths}")

    if cfg.dia.reference_processing != _PILOT_EXPECTED_REFERENCE_PROCESSING:
        print(
            f"\n[WARN] reference_processing={cfg.dia.reference_processing!r} "
            f"is not {_PILOT_EXPECTED_REFERENCE_PROCESSING!r}. "
            "DIA residuals are NOT science-safe."
        )

    print(f"\nBuilding SceneSimulator (master_seed={master_seed}) ...")
    sim = SceneSimulator(cfg, master_seed=master_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {output_dir.resolve()}")
    print(f"Batch size : {batch_size}  n_epochs: {n_epochs}")

    timestamps_mjd = np.array([60000.0 + i for i in range(n_epochs)])
    backgrounds = [_SKY_BG_E_PER_S] * n_epochs

    thresholds: dict = {
        "null_central_snr_warn": null_central_snr_warn,
        "rim_ratio_low": rim_ratio_low,
        "rim_ratio_high": rim_ratio_high,
        "max_saturation_fraction": max_saturation_fraction,
        "max_cr_fraction": max_cr_fraction,
    }

    all_rows: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"[PILOT BATCH]  {batch_size} events  seed={master_seed}")

    for i in range(batch_size):
        event_id = f"pilot_{i:04d}"
        is_null = (i % 2 == 0)
        event_type = "null" if is_null else "lensed"

        if is_null:
            source_params = [{"flux_e": _BASELINE_SOURCE_FLUX_E} for _ in range(n_epochs)]
            baseline_flux: float | None = _BASELINE_SOURCE_FLUX_E
        else:
            mags = _gaussian_magnification_curve(n_epochs, peak=5.0)
            source_params = [
                {"flux_e": float(A) * _BASELINE_SOURCE_FLUX_E} for A in mags
            ]
            baseline_flux = _BASELINE_SOURCE_FLUX_E

        print(f"\n  [{i + 1:3d}/{batch_size}] {event_id}  type={event_type}")

        try:
            result = sim.simulate_event(
                event_id=event_id,
                source_params_sequence=source_params,
                timestamps_mjd=timestamps_mjd,
                backgrounds_e_per_s=backgrounds,
                baseline_source_flux_e=baseline_flux,
            )
        except Exception as exc:
            print(f"  [ERROR] {event_id}: {type(exc).__name__}: {exc}")
            continue

        ev_dir = output_dir / event_id
        ev_dir.mkdir(exist_ok=True)

        np.save(ev_dir / "diff_stamps.npy", result.difference_stamps)
        np.save(ev_dir / "sat_stamps.npy", result.saturation_stamps)
        np.save(ev_dir / "cr_stamps.npy", result.cr_stamps)

        epoch_rows: list[dict] = []
        for ep_idx, diff in enumerate(result.difference_stamps):
            sat_ep = (
                result.saturation_stamps[ep_idx]
                if result.saturation_stamps.ndim >= 3
                else result.saturation_stamps
            )
            cr_ep = (
                result.cr_stamps[ep_idx]
                if result.cr_stamps.ndim >= 3
                else result.cr_stamps
            )
            prov = (
                result.provenance[ep_idx]
                if result.provenance and ep_idx < len(result.provenance)
                else None
            )

            row = _compute_pilot_epoch_row(
                event_id=event_id,
                event_type=event_type,
                epoch=ep_idx,
                diff=diff,
                sat=sat_ep,
                cr=cr_ep,
                prov=prov,
                cfg=cfg,
                aperture_radius=aperture_radius,
                n_random_apertures=n_random_apertures,
            )
            row["qa_flags"] = _pilot_qa_flags(
                row,
                null_central_snr_warn=null_central_snr_warn,
                rim_ratio_low=rim_ratio_low,
                rim_ratio_high=rim_ratio_high,
                max_saturation_fraction=max_saturation_fraction,
                max_cr_fraction=max_cr_fraction,
            )
            row["central_signal_dominant"] = _central_signal_dominant(
                row, rim_ratio_low=rim_ratio_low, snr_thresh=null_central_snr_warn,
            )
            epoch_rows.append(row)

            pct_s = (
                f"  pct={row['central_aperture_percentile']:5.1f}%"
                if row["central_aperture_percentile"] is not None
                else ""
            )
            flag_s = f"  [{', '.join(row['qa_flags'])}]" if row["qa_flags"] else ""
            print(
                f"    ep{ep_idx:2d}  mean={row['mean']:+9.2f}  rms={row['std']:8.2f}  "
                f"cSNR={row['central_aperture_snr']:+6.2f}{pct_s}{flag_s}"
            )

        all_rows.extend(epoch_rows)

        # Per-event metadata.json
        prov0 = result.provenance[0] if result.provenance else None
        meta: dict = {
            "event_id": event_id,
            "event_type": event_type,
            "config_sha256": config_sha,
            "master_seed": master_seed,
            "n_epochs": n_epochs,
            "aperture_radius": aperture_radius,
            "reference_processing": cfg.dia.reference_processing,
            "reference_jitter_mode": cfg.dia.reference_jitter_mode,
            "dia_subtraction_method": cfg.dia.subtraction_method,
            "psf_fov_native_pixels": cfg.psf.fov_native_pixels,
            "psf_oversample": cfg.psf.oversample,
            "psf_n_wavelengths": cfg.psf.n_wavelengths,
            "dia_method_provenance": getattr(prov0, "dia_method", None),
            "reference_n_epochs_provenance": getattr(prov0, "reference_n_epochs", None),
            "saturation_pixel_fraction": round(float(result.saturation_stamps.mean()), 6),
            "cr_pixel_fraction": round(float(result.cr_stamps.mean()), 6),
            "epoch_stats": epoch_rows,
        }
        (ev_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        if _MPL:
            ev_scale = _robust_scale(result.difference_stamps)
            for ep_idx, diff in enumerate(result.difference_stamps):
                _save_png_fixed_scale(
                    diff,
                    ev_dir / f"epoch_{ep_idx:02d}_diff.png",
                    title=f"{event_id} ep{ep_idx} ({event_type})",
                    vmax=ev_scale,
                )

    if not all_rows:
        print("\n[WARN] No rows collected; all events failed.")
        return

    # Aggregate summaries
    json_path, csv_path = _save_pilot_summary(all_rows, output_dir)
    print(f"\n  Pilot summary JSON : {json_path}")
    print(f"  Pilot summary CSV  : {csv_path}")

    run_meta: dict = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "master_seed": master_seed,
        "aperture_radius": aperture_radius,
        "n_random_apertures": n_random_apertures,
        "thresholds": thresholds,
    }
    (output_dir / "pilot_run_config.json").write_text(json.dumps(run_meta, indent=2))

    _print_pilot_console_summary(all_rows, thresholds)


# ── main logic ────────────────────────────────────────────────────────────────

def run(
    config_path: Path,
    output_dir: Path,
    n_epochs: int,
    master_seed: int,
    aperture_radius: int = _DEFAULT_APERTURE_RADIUS,
    n_random_apertures: int = _DEFAULT_RANDOM_APERTURES,
    snr_warn: float = _DEFAULT_CENTRAL_SNR_WARN,
    percentile_warn: float = _DEFAULT_CENTRAL_PERCENTILE_WARN,
    capture_dia_inputs: bool = False,
    dia_diagnostic_modes: bool = False,
) -> None:
    # ── load and hash config ──────────────────────────────────────────────
    print(f"Loading config: {config_path}")
    cfg = load_simulation_config(config_path)
    config_sha = get_simulation_config_sha256(cfg)

    print(f"\n[CONFIG] SHA-256: {config_sha[:16]}...")
    print(f"  reference_processing   = {cfg.dia.reference_processing}")
    print(f"  reference_jitter_mode  = {cfg.dia.reference_jitter_mode}")
    print(f"  psf.fov_native_pixels  = {cfg.psf.fov_native_pixels}")
    print(f"  psf.oversample         = {cfg.psf.oversample}")
    print(f"  psf.n_wavelengths      = {cfg.psf.n_wavelengths}")
    print(f"  dia.context_stamp_size = {cfg.dia.context_stamp_size}")
    print(f"  dia.science_stamp_size = {cfg.dia.science_stamp_size}")

    if cfg.dia.reference_processing != "detector_matched":
        print(
            f"\n[WARN] reference_processing={cfg.dia.reference_processing!r} is not "
            "'detector_matched'. DIA residuals are NOT science-safe — use "
            "simulation_science.yaml for trusted QA."
        )

    # ── build simulator ───────────────────────────────────────────────────
    print(f"\nBuilding SceneSimulator (master_seed={master_seed}) ...")
    simulator = SceneSimulator(cfg, master_seed=master_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    # ── fixed timestamps and backgrounds ─────────────────────────────────
    # MJD 60000 ≈ 2023-02-25; one-day cadence for clarity.
    timestamps_mjd = np.array([60000.0 + i for i in range(n_epochs)])
    backgrounds = [_SKY_BG_E_PER_S] * n_epochs

    events = [
        {
            "event_id": "qa_null",
            "is_null": True,
            "label": "NULL (A=1)",
            "source_params_sequence": [
                {"flux_e": _BASELINE_SOURCE_FLUX_E} for _ in range(n_epochs)
            ],
        },
        {
            "event_id": "qa_lensed",
            "is_null": False,
            "label": f"LENSED (Gaussian bell, peak A={5.0:.1f})",
            "source_params_sequence": [
                {"flux_e": float(A) * _BASELINE_SOURCE_FLUX_E}
                for A in _gaussian_magnification_curve(n_epochs, peak=5.0)
            ],
        },
    ]

    # ── metadata skeleton ─────────────────────────────────────────────────
    metadata: dict = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "psf_fov_native_pixels": cfg.psf.fov_native_pixels,
        "psf_oversample": cfg.psf.oversample,
        "psf_n_wavelengths": cfg.psf.n_wavelengths,
        "dia_context_stamp_size": cfg.dia.context_stamp_size,
        "dia_science_stamp_size": cfg.dia.science_stamp_size,
        "dia_reference_processing": cfg.dia.reference_processing,
        "dia_reference_jitter_mode": cfg.dia.reference_jitter_mode,
        "dia_subtraction_method": cfg.dia.subtraction_method,
        "master_seed": master_seed,
        "n_epochs": n_epochs,
        "aperture_radius": aperture_radius,
        "n_random_apertures": n_random_apertures,
        "event_ids": [e["event_id"] for e in events],
        "events": {},
    }

    # Collect diff stamps for the fixed-scale PNG pass after all events.
    all_diff_stamps: dict[str, np.ndarray] = {}
    # Collect AL diagnostic rows across all events (for consolidated summary).
    all_al_diag_rows: list[dict] = []

    # ── simulate each event ───────────────────────────────────────────────
    for ev in events:
        eid = ev["event_id"]
        is_null = ev["is_null"]
        label = ev["label"]
        source_params = ev["source_params_sequence"]

        print(f"\n{'─' * 60}")
        print(f"Simulating: {eid}  [{label}]")

        dia_capture: list[dict] = []
        ctx_mgr = (
            _DiaCaptureContext(dia_capture)
            if capture_dia_inputs
            else contextlib.nullcontext()
        )

        with ctx_mgr:
            result = simulator.simulate_event(
                event_id=eid,
                source_params_sequence=source_params,
                timestamps_mjd=timestamps_mjd,
                backgrounds_e_per_s=backgrounds,
                baseline_source_flux_e=_BASELINE_SOURCE_FLUX_E,
            )

        ev_dir = output_dir / eid
        ev_dir.mkdir(exist_ok=True)

        # DIA-input capture: save arrays before the per-epoch loop.
        dia_epoch_metrics_main: list[dict] = []
        ev_al_diag_rows: list[dict] = []
        if capture_dia_inputs and dia_capture:
            sci_sz = cfg.dia.science_stamp_size
            ctx_sz = cfg.dia.context_stamp_size
            dia_dir = ev_dir / "dia_inputs"
            _save_dia_capture_arrays(dia_capture, dia_dir, sci_sz, ctx_sz)
            for entry in dia_capture:
                dia_epoch_metrics_main.append(
                    _compute_dia_input_epoch_metrics(entry, aperture_radius)
                )
            _print_dia_capture_summary(eid, dia_epoch_metrics_main)

            if dia_diagnostic_modes:
                ev_al_diag_rows = _run_al_diagnostic_suite(
                    capture_entries=dia_capture,
                    event_or_ablation=eid,
                    aperture_radius=aperture_radius,
                    out_dir=ev_dir / "al_diagnostics",
                )
                all_al_diag_rows.extend(ev_al_diag_rows)

        np.save(ev_dir / "diff_stamps.npy", result.difference_stamps)
        np.save(ev_dir / "sat_stamps.npy", result.saturation_stamps)
        np.save(ev_dir / "cr_stamps.npy", result.cr_stamps)

        all_diff_stamps[eid] = result.difference_stamps

        # ── per-epoch loop ────────────────────────────────────────────
        epoch_stats_list = []
        psf_config_hash_seen = None
        null_epoch_snrs: list[float] = []

        for i, (diff, prov) in enumerate(
            zip(result.difference_stamps, result.provenance)
        ):
            pch = getattr(prov, "psf_config_hash", None)
            if pch is not None:
                psf_config_hash_seen = pch

            has_nonfinite = not bool(np.all(np.isfinite(diff)))
            if has_nonfinite:
                print(f"  [WARN] Epoch {i}: non-finite values detected in diff stamp!")

            stats = _stamp_summary(diff)
            rim = _rim_stats(diff)
            central = _central_residual_metrics(diff, aperture_radius)

            # Null-event random-aperture comparison
            rand_ap: dict | None = None
            if is_null:
                rand_ap = _null_random_apertures(
                    diff, aperture_radius, n_random_apertures, seed=42
                )
                null_epoch_snrs.append(central["central_aperture_snr"])

            # Build QA flags for this epoch
            qa_flags: list[str] = []
            if has_nonfinite:
                qa_flags.append("nonfinite_values")
            if is_null:
                snr_val = central["central_aperture_snr"]
                if abs(snr_val) > snr_warn:
                    qa_flags.append("central_null_residual_high_snr")
                    print(
                        f"  [WARN] Epoch {i}: central aperture |SNR|={abs(snr_val):.2f} "
                        f"> threshold {snr_warn}"
                    )
                if rand_ap is not None:
                    pct = rand_ap.get("central_aperture_percentile", float("nan"))
                    if not (pct != pct) and pct > percentile_warn:  # nan-safe >
                        qa_flags.append("central_null_residual_high_percentile")
                        print(
                            f"  [WARN] Epoch {i}: central percentile={pct:.1f}% "
                            f"> threshold {percentile_warn}%"
                        )

            # Print enhanced per-epoch summary
            rand_pct_str = ""
            if rand_ap is not None:
                pct = rand_ap.get("central_aperture_percentile", float("nan"))
                rand_pct_str = f"  pct={pct:5.1f}%"
            print(
                f"  Epoch {i:2d}  mean={stats['mean']:+9.2f}  rms={stats['std']:8.2f}  "
                f"ctr_snr={central['central_aperture_snr']:+6.2f}"
                f"{rand_pct_str}"
                f"  rim_mean={rim['rim_mean']:+9.2f}"
            )

            if _MPL:
                _save_png(
                    diff,
                    ev_dir / f"epoch_{i:02d}_diff.png",
                    title=f"{eid} epoch {i}  mean={stats['mean']:+.1f}",
                )

            if _FITS:
                _save_fits(
                    diff,
                    ev_dir / f"epoch_{i:02d}_diff.fits",
                    header_kws={"EVID": eid, "EPOCH": i},
                )

            dia_metrics_this_epoch = (
                dia_epoch_metrics_main[i]
                if capture_dia_inputs and i < len(dia_epoch_metrics_main)
                else {"dia_input_capture_enabled": False}
            )
            epoch_stats_list.append({
                "epoch": i,
                "has_nonfinite": has_nonfinite,
                "stamp_stats": stats,
                "rim_stats": {k: round(v, 4) for k, v in rim.items()},
                "central_metrics": central,
                "null_random_aperture": rand_ap,
                "qa_flags": qa_flags,
                "dia_input_metrics": dia_metrics_this_epoch,
            })

        # ── null event: repeated same-sign centre check ───────────────
        if is_null and len(null_epoch_snrs) >= 2:
            high_all = all(abs(s) > 3.0 for s in null_epoch_snrs)
            if high_all:
                if all(s > 0 for s in null_epoch_snrs):
                    print(
                        f"  [WARN] Null event: central residual is POSITIVE with "
                        f"|SNR|>3 in all {len(null_epoch_snrs)} epochs — "
                        "possible DIA/template mismatch."
                    )
                elif all(s < 0 for s in null_epoch_snrs):
                    print(
                        f"  [WARN] Null event: central residual is NEGATIVE with "
                        f"|SNR|>3 in all {len(null_epoch_snrs)} epochs — "
                        "possible DIA/template mismatch."
                    )

        # ── provenance summary ────────────────────────────────────────
        prov0 = result.provenance[0] if result.provenance else None
        dia_method = getattr(prov0, "dia_method", None)
        ref_n_epochs_prov = getattr(prov0, "reference_n_epochs", None)
        pch_short = (psf_config_hash_seen[:8] + "...") if psf_config_hash_seen else "N/A"

        print(
            f"  Provenance: dia_method={dia_method!r}  "
            f"reference_n_epochs={ref_n_epochs_prov}  "
            f"psf_config_hash={pch_short}"
        )
        if dia_method == "identity":
            print(
                "  [INFO] dia_method=identity — direct science-reference subtraction "
                "(default for matched-reference simulations).  AL diagnostics may "
                "still be enabled for comparison via --dia-diagnostic-modes."
            )

        if cfg.dia.reference_processing == "detector_matched":
            print("  [OK] reference_processing=detector_matched confirmed.")
        else:
            print("  [WARN] reference_processing is not detector_matched.")

        sat_frac = float(result.saturation_stamps.mean())
        cr_frac = float(result.cr_stamps.mean())
        if sat_frac > 0.0:
            print(f"  [INFO] Saturated pixel fraction: {sat_frac:.4f}")
        if cr_frac > 0.0:
            print(f"  [INFO] CR-flagged pixel fraction: {cr_frac:.4f}")

        metadata["events"][eid] = {
            "label": label,
            "psf_config_hash": psf_config_hash_seen,
            "dia_method": dia_method,
            "reference_n_epochs_provenance": ref_n_epochs_prov,
            "saturation_pixel_fraction": round(sat_frac, 6),
            "cr_pixel_fraction": round(cr_frac, 6),
            "epoch_stats": epoch_stats_list,
        }

    # ── fixed-scale PNG pass ──────────────────────────────────────────────
    if _MPL and all_diff_stamps:
        # Per-event robust scales
        event_scales: dict[str, float] = {
            eid: _robust_scale(diffs)
            for eid, diffs in all_diff_stamps.items()
        }
        # Global scale across all events
        all_flat = np.concatenate([d.ravel() for d in all_diff_stamps.values()])
        global_scale = _robust_scale(all_flat)

        for ev in events:
            eid = ev["event_id"]
            diffs = all_diff_stamps[eid]
            ev_dir = output_dir / eid
            ev_scale = event_scales[eid]

            for i, diff in enumerate(diffs):
                _save_png_fixed_scale(
                    diff,
                    ev_dir / f"epoch_{i:02d}_diff_event_scale.png",
                    title=f"{eid} epoch {i}  [event ±{ev_scale:.2f}]",
                    vmax=ev_scale,
                )
                _save_png_fixed_scale(
                    diff,
                    ev_dir / f"epoch_{i:02d}_diff_global_scale.png",
                    title=f"{eid} epoch {i}  [global ±{global_scale:.2f}]",
                    vmax=global_scale,
                )

            metadata["events"][eid]["fixed_scale_event_vmax"] = round(ev_scale, 4)

        metadata["fixed_scale_global_vmax"] = round(global_scale, 4)

    # ── AL diagnostic summary (consolidated across all events) ────────────
    if dia_diagnostic_modes and all_al_diag_rows:
        print(f"\n{'─' * 60}")
        print("[AL DIAGNOSTIC SUMMARY — MAIN EVENTS]")
        _save_al_diagnostic_summary(all_al_diag_rows, output_dir)

    # ── write metadata JSON ───────────────────────────────────────────────
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # ── final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[SUMMARY]")
    print(f"  Config SHA:    {config_sha[:16]}...")
    print(f"  master_seed:   {master_seed}")
    print(f"  n_epochs:      {n_epochs}")
    print(f"  Aperture r:    {aperture_radius} px  |  random samples: {n_random_apertures}")
    print(f"  Stamps dir:    {output_dir.resolve()}")
    print(f"  Metadata:      {meta_path}")
    print(f"  PNG previews:  {'written (auto + fixed-scale)' if _MPL else 'skipped (matplotlib not available)'}")
    print(f"  FITS outputs:  {'written' if _FITS else 'skipped (astropy not available)'}")

    science_ok = cfg.dia.reference_processing == "detector_matched"
    print(
        f"  Science QA:    "
        f"{'PASS — reference_processing=detector_matched' if science_ok else 'WARN — reference_processing is not detector_matched'}"
    )


def main() -> None:
    args = _build_parser().parse_args()
    config_path = _resolve_config(args.config)

    if args.pilot_batch_size > 0:
        # Use the pilot-specific default output dir unless the user overrode it.
        out_dir = (
            Path(_DEFAULT_PILOT_OUTPUT_DIR)
            if args.output_dir == _DEFAULT_OUTPUT_DIR
            else Path(args.output_dir)
        )
        run_pilot_batch(
            config_path=config_path,
            output_dir=out_dir,
            batch_size=args.pilot_batch_size,
            n_epochs=args.n_epochs,
            master_seed=args.master_seed,
            aperture_radius=args.central_aperture_radius,
            n_random_apertures=args.random_apertures,
            null_central_snr_warn=args.null_central_snr_warn,
            rim_ratio_low=args.rim_ratio_low,
            rim_ratio_high=args.rim_ratio_high,
            max_nonfinite_allowed=args.max_nonfinite_allowed,
            max_saturation_fraction=args.max_saturation_fraction,
            max_cr_fraction=args.max_cr_fraction,
        )
        return

    # --dia-diagnostic-modes implies --capture-dia-inputs.
    dia_diag = args.dia_diagnostic_modes
    capture_dia = args.capture_dia_inputs or dia_diag
    run(
        config_path=config_path,
        output_dir=Path(args.output_dir),
        n_epochs=args.n_epochs,
        master_seed=args.master_seed,
        aperture_radius=args.central_aperture_radius,
        n_random_apertures=args.random_apertures,
        snr_warn=args.central_snr_warn,
        percentile_warn=args.central_percentile_warn,
        capture_dia_inputs=capture_dia,
        dia_diagnostic_modes=dia_diag,
    )
    if args.run_null_ablations:
        cfg = load_simulation_config(config_path)
        run_null_ablations(
            cfg=cfg,
            output_dir=Path(args.output_dir),
            n_epochs=args.n_epochs,
            master_seed=args.master_seed,
            aperture_radius=args.central_aperture_radius,
            n_random_apertures=args.random_apertures,
            capture_dia_inputs=capture_dia,
            dia_diagnostic_modes=dia_diag,
        )


if __name__ == "__main__":
    main()
