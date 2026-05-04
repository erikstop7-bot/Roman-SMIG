#!/usr/bin/env python3
"""plot_h5_lightcurves.py

Extract one high-signal representative event per `EventClass` from a Phase 3
`phase3-contract-v1` HDF5 shard and plot its 1-D aperture-summed DIA residual
curve. Used as a presentation asset for UCLA Research Week 2026.

Data Availability Contract
--------------------------
The shard is inspected at open time. The script adapts to the datasets
that actually exist rather than assuming a fixed layout:

  * ``/science_stamps``            -> required; (N, n_epochs, 64, 64) float32.
                                      These are DIA *difference* stamps, NOT
                                      total-flux stamps.  Expected values are
                                      approximately ``(A(t) − 1) * F₀``
                                      integrated over the aperture, i.e. near
                                      zero at baseline epochs.
  * ``/label__event_class``        -> required; uint8 in {0..4} per labels.py.
  * ``/label__log_tE``, ``log_u0``, ``log_rho``, ``alpha_rad``,
    ``log_q``, ``log_s``          -> optional; used for on-figure annotations.
  * ``/label__t0_mjd_normalized``  -> optional; t0 normalised to [0,1] over
                                      the observation window; used to draw a
                                      vertical marker in the plot.
  * ``/event_id``                  -> optional; used as plot title text.
  * ``/saturation_stamps``,
    ``/cr_stamps``                 -> optional; if present, saturated pixels
                                      in the aperture are zeroed before summing.

If any *optional* dataset is missing, a WARNING is logged and the script
continues with a sensible fallback.  If any *required* dataset is missing,
the script exits with a non-zero code and a clear diagnostic.

Plot mode
---------
The default (and only) mode is ``residual``:

  * Raw central aperture sum of the DIA stamp in native units (approximately
    e-/s for phase3-contract-v1 shards).
  * Y-axis zero line — NOT a baseline=1 line.  DIA baselines are near zero
    after the Phase 3 fix; dividing by the outer-epoch median produces
    ill-conditioned ratios and is deliberately not implemented.
  * t0 vertical marker drawn when ``label__t0_mjd_normalized`` is available.

Representative event selection
------------------------------
Events are ranked by ``max(abs(aperture_sum))`` — the peak absolute DIA
residual — which is correct for signed difference-image values.  Ties are
broken deterministically by lexicographic ``event_id``.

Cadence
-------
The x-axis cadence MUST match the generation config.  Either:

  * ``--phase3-config PATH`` — reads ``cadence_days`` from the generation
    YAML.  Takes precedence over ``--cadence-days`` if both are given.
  * ``--cadence-days VALUE`` — explicit override.
  * If neither is given, defaults to 0.0104 d (smoke config) with a WARNING.

CLI
---
See ``--help``.

Example
-------
    python scripts/presentation_assets/plot_h5_lightcurves.py \\
        --shard-path outputs/phase3_sprint_static/shards/shard_0000.h5 \\
        --phase3-config configs/phase3_sprint_static.yaml \\
        --output-dir outputs/presentation/lightcurves \\
        --seed 20260422
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py
except ImportError as exc:
    print(f"FATAL: h5py is required ({exc})", file=sys.stderr)
    sys.exit(2)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("plot_h5_lightcurves")

# Smoke-config fallback cadence — used only when neither --phase3-config
# nor --cadence-days is provided.
_SMOKE_CADENCE_DAYS: float = 0.0104

# Stable uint8 encoding — MUST match smig/datasets/labels.py `_EVENT_CLASS_ORDER`.
EVENT_CLASS_NAMES: dict[int, str] = {
    0: "PSPL",
    1: "FSPL_STAR",
    2: "PLANETARY_CAUSTIC",
    3: "STELLAR_BINARY",
    4: "HIGH_MAGNIFICATION_CUSP",
}

REQUIRED_DATASETS: tuple[str, ...] = (
    "science_stamps",
    "label__event_class",
)

OPTIONAL_LABELS: tuple[str, ...] = (
    "label__log_tE",
    "label__log_u0",
    "label__log_rho",
    "label__alpha_rad",
    "label__log_q",
    "label__log_s",
    "label__t0_mjd_normalized",
)


# ---------------------------------------------------------------------------
# Cadence resolution
# ---------------------------------------------------------------------------

def _resolve_cadence(args: argparse.Namespace) -> float:
    """Return cadence_days with precedence: --phase3-config > --cadence-days > smoke default."""
    if args.phase3_config is not None:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            _log.error(
                "--phase3-config requires PyYAML. Install it: pip install pyyaml"
            )
            sys.exit(2)
        cfg = yaml.safe_load(Path(args.phase3_config).read_text())
        cadence = float(cfg["cadence_days"])
        if args.cadence_days is not None:
            _log.warning(
                "--phase3-config cadence_days=%.4g overrides "
                "explicit --cadence-days=%.4g.",
                cadence,
                args.cadence_days,
            )
        else:
            _log.info("cadence_days=%.4g from %s", cadence, args.phase3_config)
        return cadence

    if args.cadence_days is not None:
        return float(args.cadence_days)

    _log.warning(
        "Neither --phase3-config nor --cadence-days supplied; "
        "defaulting to smoke-config cadence %.4g d.  "
        "Pass --phase3-config configs/phase3_sprint_static.yaml for the "
        "sprint static baseline (cadence=2.5 d).",
        _SMOKE_CADENCE_DAYS,
    )
    return _SMOKE_CADENCE_DAYS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot DIA residual/excess-flux aperture curves, one per EventClass.  "
            "/science_stamps in phase3-contract-v1 are DIA difference stamps "
            "carrying approximately (A(t)−1)·F₀ excess flux, not total source flux."
        ),
    )
    p.add_argument(
        "--shard-path",
        required=True,
        type=Path,
        help="Path to a phase3-contract-v1 HDF5 shard.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write PNG into (created if missing).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260422,
        help="Deterministic seed for tie-breaking; recorded in output filename.",
    )
    p.add_argument(
        "--aperture-radius",
        type=int,
        default=3,
        help="Half-width in pixels of the central aperture box (default 3 -> 7x7).",
    )
    p.add_argument(
        "--cadence-days",
        type=float,
        default=None,
        help=(
            "Epoch cadence in days for the x-axis.  Overridden by "
            "--phase3-config if both are given.  Defaults to 0.0104 d "
            "(smoke config) with a WARNING if omitted."
        ),
    )
    p.add_argument(
        "--phase3-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a Phase 3 generation YAML config (e.g. "
            "configs/phase3_sprint_static.yaml).  If provided, cadence_days "
            "is read from the config and takes precedence over --cadence-days."
        ),
    )
    p.add_argument(
        "--plot-mode",
        choices=("residual",),
        default="residual",
        help=(
            "'residual' (default and only mode): raw DIA aperture sum in "
            "native units with a zero baseline.  Magnification reconstruction "
            "is not implemented — DIA baselines are near zero after the "
            "Phase 3 fix, making outer-epoch normalisation ill-conditioned."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_shard(shard_path: Path) -> dict:
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    with h5py.File(str(shard_path), "r") as h5:
        attrs = dict(h5.attrs)
        for name in REQUIRED_DATASETS:
            if name not in h5:
                raise KeyError(
                    f"Required dataset '/{name}' missing from {shard_path}"
                )
        # /science_stamps are DIA difference stamps: approx (A(t)-1)*F0 per pixel.
        science = np.asarray(h5["science_stamps"][:], dtype=np.float32)
        event_class = np.asarray(h5["label__event_class"][:], dtype=np.uint8)

        if "event_id" in h5:
            raw_ids = h5["event_id"][:]
            event_ids = [
                (eid.decode() if isinstance(eid, bytes) else str(eid))
                for eid in raw_ids
            ]
        else:
            _log.warning("'/event_id' missing — using row indices as event IDs.")
            event_ids = [f"row_{i:05d}" for i in range(science.shape[0])]

        labels: dict[str, np.ndarray] = {}
        for ds_name in OPTIONAL_LABELS:
            if ds_name in h5:
                labels[ds_name] = np.asarray(h5[ds_name][:], dtype=np.float32)
            else:
                _log.warning(
                    "Optional label '/%s' missing — annotation will be omitted.",
                    ds_name,
                )

        saturation = (
            np.asarray(h5["saturation_stamps"][:], dtype=bool)
            if "saturation_stamps" in h5
            else None
        )
        if saturation is None:
            _log.warning(
                "'/saturation_stamps' missing — aperture sums will include "
                "all central pixels without saturation masking."
            )

    n_epochs = science.shape[1]
    return {
        "attrs": attrs,
        "science": science,
        "event_class": event_class,
        "event_ids": event_ids,
        "labels": labels,
        "saturation": saturation,
        "n_epochs": n_epochs,
    }


# ---------------------------------------------------------------------------
# Aperture photometry
# ---------------------------------------------------------------------------

def _aperture_sum(
    stamps: np.ndarray, half_width: int, sat_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Sum pixels inside a centred box aperture for every epoch.

    stamps    : (n_epochs, H, W) float32 — DIA difference stamp cube.
    half_width: pixels; aperture box is (2*half_width + 1)**2.
    sat_mask  : (n_epochs, H, W) bool, True where pixel is saturated.

    Returns (n_epochs,) float64 in the stamp's native units.  For DIA stamps
    the expected value is approximately (A(t) − 1) * F₀ integrated over the
    aperture, so baseline epochs should yield values near zero.
    """
    _, ny, nx = stamps.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(cy - half_width, 0), min(cy + half_width + 1, ny)
    x0, x1 = max(cx - half_width, 0), min(cx + half_width + 1, nx)
    crop = stamps[:, y0:y1, x0:x1].astype(np.float64)
    if sat_mask is not None:
        crop = np.where(sat_mask[:, y0:y1, x0:x1], 0.0, crop)
    return crop.sum(axis=(1, 2))


# ---------------------------------------------------------------------------
# Event ranking
# ---------------------------------------------------------------------------

def _rank_top_per_class(
    data: dict, aperture_radius: int, rng: np.random.Generator
) -> dict[int, int]:
    """Return {class_uint8 -> row index} for the top-signal event per class.

    Ranking uses max(abs(aperture_sum)) — the peak *absolute* DIA residual —
    so that both positive-excess and negative-residual events are ranked
    fairly.  Deterministic tie-breaking: peak abs signal (descending) then
    event_id (ascending lexicographic).  The RNG is consulted only if two
    events share both keys, which should not happen in practice.
    """
    science: np.ndarray = data["science"]
    event_class: np.ndarray = data["event_class"]
    event_ids: list[str] = data["event_ids"]
    sat: Optional[np.ndarray] = data["saturation"]

    winners: dict[int, int] = {}
    unique_classes = np.unique(event_class).tolist()
    for cls_uint in unique_classes:
        mask = event_class == cls_uint
        idx_in_class = np.flatnonzero(mask)
        if idx_in_class.size == 0:
            continue
        peak_abs = np.empty(idx_in_class.size, dtype=np.float64)
        for j, row in enumerate(idx_in_class):
            sat_row = sat[row] if sat is not None else None
            lc = _aperture_sum(science[row], aperture_radius, sat_row)
            peak_abs[j] = float(np.max(np.abs(lc)))

        eids = np.asarray([event_ids[i] for i in idx_in_class])
        jitter = rng.random(idx_in_class.size) * 1e-12
        order = np.lexsort((jitter, eids, -peak_abs))
        winner_row = int(idx_in_class[order[0]])
        winners[int(cls_uint)] = winner_row
        _log.info(
            "Class %s (%d): %d events, top row=%d id=%s peak_abs=%.3g",
            EVENT_CLASS_NAMES.get(int(cls_uint), str(cls_uint)),
            int(cls_uint),
            idx_in_class.size,
            winner_row,
            event_ids[winner_row],
            float(peak_abs[order[0]]),
        )

    for cls_uint, name in EVENT_CLASS_NAMES.items():
        if cls_uint not in winners:
            _log.warning(
                "Class %s (%d) absent in shard — skipping.", name, cls_uint
            )
    return winners


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _format_label_annotation(data: dict, row: int) -> str:
    """Compact per-event parameter label block for the plot title."""
    labels = data["labels"]
    pieces: list[str] = []
    for key, sym in [
        ("label__log_tE", r"$\log t_E$"),
        ("label__log_u0", r"$\log u_0$"),
        ("label__log_rho", r"$\log\rho$"),
        ("label__alpha_rad", r"$\alpha$"),
        ("label__log_q", r"$\log q$"),
        ("label__log_s", r"$\log s$"),
    ]:
        if key not in labels:
            continue
        val = float(labels[key][row])
        # Per smig/datasets/worker.py UNDEFINED_BINARY_PARAM_SENTINEL = -99.0
        if val <= -90.0 and key in ("label__log_q", "label__log_s"):
            pieces.append(f"{sym}=N/A")
        else:
            pieces.append(f"{sym}={val:.3g}")
    return ", ".join(pieces)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_lightcurves(
    data: dict,
    winners: dict[int, int],
    aperture_radius: int,
    cadence_days: float,
    output_path: Path,
    shard_path: Path,
    seed: int,
) -> None:
    """Plot DIA residual/excess-flux aperture curves, one subplot per EventClass.

    Each subplot shows the raw central aperture sum of the DIA difference stamp
    (/science_stamps) in native float32 units (approximately e-/s).  A
    horizontal zero line marks the expected baseline; a vertical orange line
    marks t0 when label__t0_mjd_normalized is available.

    These are NOT total-flux or magnification curves.  The /science_stamps
    dataset in phase3-contract-v1 stores DIA difference stamps carrying
    approximately (A(t)−1)·F₀ excess flux; baseline epochs are near zero.
    """
    if not winners:
        raise RuntimeError("No winners selected — nothing to plot.")
    ordered = [(cls, winners[cls]) for cls in sorted(winners)]
    n = len(ordered)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])

    science: np.ndarray = data["science"]
    sat: Optional[np.ndarray] = data["saturation"]
    n_epochs: int = data["n_epochs"]
    t_epochs = np.arange(n_epochs) * cadence_days

    labels: dict[str, np.ndarray] = data["labels"]
    t0_arr: Optional[np.ndarray] = labels.get("label__t0_mjd_normalized")

    colors = {
        0: "#1f77b4",  # PSPL
        1: "#2ca02c",  # FSPL_STAR
        2: "#d62728",  # PLANETARY_CAUSTIC
        3: "#9467bd",  # STELLAR_BINARY
        4: "#ff7f0e",  # HIGH_MAGNIFICATION_CUSP
    }

    for ax, (cls_uint, row) in zip(axes, ordered):
        sat_row = sat[row] if sat is not None else None
        lc = _aperture_sum(science[row], aperture_radius, sat_row)

        class_name = EVENT_CLASS_NAMES.get(int(cls_uint), f"class_{cls_uint}")
        eid = data["event_ids"][row]
        annot = _format_label_annotation(data, row)

        ax.plot(
            t_epochs,
            lc,
            marker="o",
            linestyle="-",
            color=colors.get(int(cls_uint), "black"),
            lw=1.3,
            ms=4,
        )
        ax.axhline(0.0, color="grey", linestyle="--", lw=0.7, alpha=0.7)

        if t0_arr is not None:
            t0_norm = float(t0_arr[row])
            if 0.0 <= t0_norm <= 1.0:
                t0_plot = t0_norm * (n_epochs - 1) * cadence_days
                ax.axvline(
                    t0_plot,
                    color="orange",
                    linestyle=":",
                    lw=1.2,
                    alpha=0.9,
                    label=f"t₀ = {t0_plot:.2f} d",
                )
                ax.legend(fontsize=7, loc="upper right")

        ax.set_title(
            f"{class_name}  |  {eid}\n{annot}",
            fontsize=9,
            loc="left",
        )
        ax.set_ylabel("DIA aperture sum (e⁻/s)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f"epoch time (days; Δt = {cadence_days:g} d)")
    fig.suptitle(
        "SMIG v2 — Phase 3 DIA residual/excess-flux aperture curves per EventClass\n"
        f"shard={shard_path.name} | /science_stamps are DIA difference stamps"
        f" ≈(A(t)−1)·F₀ | aperture=({2 * aperture_radius + 1}×"
        f"{2 * aperture_radius + 1}) | seed={seed}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    _log.info("wrote %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    cadence_days = _resolve_cadence(args)

    try:
        data = _load_shard(args.shard_path)
    except (FileNotFoundError, KeyError) as exc:
        _log.error("%s", exc)
        return 2

    attrs = data["attrs"]
    schema_version = str(attrs.get("schema_version", "UNKNOWN"))
    if schema_version != "phase3-contract-v1":
        _log.warning(
            "Shard schema_version=%r (expected 'phase3-contract-v1'). "
            "Plot may still work but fields are not guaranteed.",
            schema_version,
        )

    rng = np.random.default_rng(args.seed)
    winners = _rank_top_per_class(data, args.aperture_radius, rng)
    if not winners:
        _log.error("No events found in shard — nothing to plot.")
        return 1

    out_name = (
        f"dia_residuals__{args.shard_path.stem}"
        f"__ap{args.aperture_radius}_seed{args.seed}.png"
    )
    out_path = args.output_dir / out_name
    _plot_lightcurves(
        data=data,
        winners=winners,
        aperture_radius=args.aperture_radius,
        cadence_days=cadence_days,
        output_path=out_path,
        shard_path=args.shard_path,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
