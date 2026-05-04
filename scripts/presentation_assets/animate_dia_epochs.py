#!/usr/bin/env python3
"""animate_dia_epochs.py

Animate the 30-epoch Difference Image Analysis (DIA) stamp sequence for a
single event from a Phase 3 `phase3-contract-v1` HDF5 shard. Output format
is selectable (GIF or MP4) for inclusion in the UCLA Research Week 2026
video presentation.

Data Availability Contract
--------------------------
This script *inspects* the shard before assuming anything about layout:

  * `/science_stamps`       -> required; (N, n_epochs, 64, 64) float32.
  * Per the Phase 3 contract (docs/phase3_dataset_contract.md:35) the
    64×64 `/science_stamps` is the only image dataset that is guaranteed.
  * A separate 256×256 context-stamp dataset (e.g. `/context_stamps`) is
    *not* part of `phase3-contract-v1`. Some local/experimental shards may
    ship one; we try four plausible keys and use whichever exists:

        /context_stamps
        /reference_stamps
        /ref_stamps
        /science_context_stamps

  * If neither a context stamp nor a FITS side-car directory is given, we
    degrade gracefully and animate only the 64×64 DIA science stamps.

  * `/saturation_stamps` and `/cr_stamps` are overlaid as masks if present.

CLI
---
See ``--help``. Example::

    python scripts/presentation_assets/animate_dia_epochs.py \\
        --shard-path outputs/phase3_smoke/shards/shard_0000.h5 \\
        --event-id 0000000001352640-00000001 \\
        --output-dir outputs/presentation/animations \\
        --format gif

MP4 output requires ``ffmpeg`` on PATH (matplotlib's FFMpegWriter). If the
writer is unavailable, the script logs an error and re-runs as GIF.

Robustness
----------
* ``matplotlib.use("Agg")`` set before pyplot import (headless-safe).
* ``argparse`` for all options.  No hard-coded paths; ``datasets/`` is
  assumed to be user-local and gitignored.
* Deterministic: no random state is consulted.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

try:
    import h5py
except ImportError as exc:
    print(f"FATAL: h5py is required ({exc})", file=sys.stderr)
    sys.exit(2)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("animate_dia_epochs")

_SMOKE_CADENCE_DAYS: float = 0.0104


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
        return cadence
    if args.cadence_days is not None:
        return float(args.cadence_days)
    _log.warning(
        "Neither --phase3-config nor --cadence-days supplied; "
        "defaulting to smoke-config cadence %.4g d.  "
        "Pass --phase3-config configs/phase3_sprint_static.yaml for the sprint "
        "static baseline (cadence=2.5 d).",
        _SMOKE_CADENCE_DAYS,
    )
    return _SMOKE_CADENCE_DAYS

EVENT_CLASS_NAMES: dict[int, str] = {
    0: "PSPL",
    1: "FSPL_STAR",
    2: "PLANETARY_CAUSTIC",
    3: "STELLAR_BINARY",
    4: "HIGH_MAGNIFICATION_CUSP",
}

LABEL_KEYS: tuple[tuple[str, str], ...] = (
    ("label__log_tE", r"$\log t_E$"),
    ("label__log_u0", r"$\log u_0$"),
    ("label__log_rho", r"$\log\rho$"),
    ("label__alpha_rad", r"$\alpha$ [rad]"),
    ("label__log_q", r"$\log q$"),
    ("label__log_s", r"$\log s$"),
)

# Candidate HDF5 keys for a 256x256 context stamp cube, in preference order.
_CONTEXT_CANDIDATES: tuple[str, ...] = (
    "context_stamps",
    "reference_stamps",
    "ref_stamps",
    "science_context_stamps",
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate 30-epoch DIA stamps for one event.",
    )
    p.add_argument(
        "--shard-path",
        required=True,
        type=Path,
        help="Path to a phase3-contract-v1 HDF5 shard.",
    )
    p.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="event_id to animate. If omitted, uses the first row in the shard.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the animation into (created if missing).",
    )
    p.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help=(
            "Output animation format. mp4 requires `ffmpeg` on PATH "
            "(matplotlib.animation.FFMpegWriter)."
        ),
    )
    p.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Animation frames per second (default 4 -> ~7.5 s for 30 epochs).",
    )
    p.add_argument(
        "--cadence-days",
        type=float,
        default=None,
        help=(
            "Epoch cadence in days for caption.  Overridden by --phase3-config "
            "if both are given.  Defaults to 0.0104 d (smoke config) with a "
            "WARNING if omitted."
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
        "--fits-preview-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to a directory of {shard}__{event_id}.fits files "
            "(produced by scripts/export_science_fits.py). Currently informational;"
            " no larger-field imagery is read from FITS previews for animation."
        ),
    )
    return p.parse_args(argv)


def _locate_event(
    h5: "h5py.File",
    event_id: Optional[str],
) -> tuple[int, str]:
    """Return (row_index, resolved_event_id)."""
    raw_ids = h5["event_id"][:] if "event_id" in h5 else None
    if raw_ids is None:
        # No event_id dataset — row 0 by convention.
        if event_id is not None:
            _log.warning(
                "--event-id %r ignored: '/event_id' dataset missing.", event_id
            )
        return 0, "row_00000"
    ids = [
        (eid.decode() if isinstance(eid, bytes) else str(eid))
        for eid in raw_ids
    ]
    if event_id is None:
        return 0, ids[0]
    if event_id not in ids:
        raise KeyError(
            f"event_id={event_id!r} not found in shard (have {ids[:5]}"
            + (f"… {len(ids)} total)" if len(ids) > 5 else ")")
        )
    return ids.index(event_id), event_id


def _load_event(shard_path: Path, event_id: Optional[str]) -> dict:
    with h5py.File(str(shard_path), "r") as h5:
        if "science_stamps" not in h5:
            raise KeyError(
                f"Required dataset '/science_stamps' missing from {shard_path}"
            )
        attrs = dict(h5.attrs)
        row, resolved_id = _locate_event(h5, event_id)

        science_row = np.asarray(
            h5["science_stamps"][row], dtype=np.float32
        )
        sat_row = (
            np.asarray(h5["saturation_stamps"][row], dtype=bool)
            if "saturation_stamps" in h5
            else None
        )
        cr_row = (
            np.asarray(h5["cr_stamps"][row], dtype=bool)
            if "cr_stamps" in h5
            else None
        )

        # Context stamps — try each candidate key. Any candidate present must
        # expose the same (N, n_epochs, H, W) layout; we verify H == W and
        # log the resolved key. If none present, context_row stays None.
        context_row: Optional[np.ndarray] = None
        context_key_used: Optional[str] = None
        for key in _CONTEXT_CANDIDATES:
            if key in h5:
                ctx = np.asarray(h5[key][row], dtype=np.float32)
                if ctx.ndim == 3 and ctx.shape[-1] == ctx.shape[-2]:
                    context_row = ctx
                    context_key_used = key
                    _log.info(
                        "Context stamps resolved from '/%s' (shape=%s).",
                        key, ctx.shape,
                    )
                    break
                _log.warning(
                    "Dataset '/%s' present but shape %s is not a "
                    "square image cube — ignoring.", key, ctx.shape,
                )

        if context_row is None:
            _log.warning(
                "No context-stamp dataset found in %s "
                "(tried %s). Falling back to 64×64-only animation "
                "per Phase 3 contract (docs/phase3_dataset_contract.md).",
                shard_path.name, list(_CONTEXT_CANDIDATES),
            )

        # Labels (all optional for annotation).
        labels: dict[str, float] = {}
        for ds_name, _ in LABEL_KEYS:
            if ds_name in h5:
                labels[ds_name] = float(h5[ds_name][row])
        if "label__event_class" in h5:
            labels["event_class"] = int(h5["label__event_class"][row])

    return {
        "attrs": attrs,
        "row": row,
        "event_id": resolved_id,
        "science": science_row,
        "saturation": sat_row,
        "cr": cr_row,
        "context": context_row,
        "context_key": context_key_used,
        "labels": labels,
    }


def _event_class_name(data: dict) -> str:
    """Return the human-readable event class name for this event."""
    ec = data["labels"].get("event_class", -1)
    return EVENT_CLASS_NAMES.get(int(ec), f"class_{ec}") if ec >= 0 else "UNKNOWN"


def _format_detail_line(data: dict, cadence_days: float, schema_version: str) -> str:
    """Return the compact detail annotation (id, params, cadence) below the class header."""
    pieces = [f"id={data['event_id']}"]
    for key, sym in LABEL_KEYS:
        if key in data["labels"]:
            v = data["labels"][key]
            if v <= -90.0 and key in ("label__log_q", "label__log_s"):
                pieces.append(f"{sym}=N/A")
            else:
                pieces.append(f"{sym}={v:.3g}")
    n_epochs = int(data["science"].shape[0])
    pieces.append(f"Δt={cadence_days:g} d | {n_epochs} epochs | {schema_version}")
    return "  ".join(pieces)


def _robust_vlims(cube: np.ndarray) -> tuple[float, float]:
    """Percentile limits for imshow; robust to CR spikes and cold pixels."""
    finite = cube[np.isfinite(cube)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _build_animation(
    data: dict,
    schema_version: str,
    cadence_days: float,
    fps: int,
) -> animation.FuncAnimation:
    science = data["science"]  # (n_epochs, 64, 64)
    context = data["context"]  # (n_epochs, H, W) or None
    sat = data["saturation"]
    cr = data["cr"]
    n_epochs = science.shape[0]

    side_by_side = context is not None
    if side_by_side:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.7))
        ax_ctx, ax_sci = axes
    else:
        fig, ax_sci = plt.subplots(1, 1, figsize=(5.0, 4.7))
        ax_ctx = None  # type: ignore[assignment]

    vlo_s, vhi_s = _robust_vlims(science)
    im_sci = ax_sci.imshow(
        science[0],
        origin="lower",
        cmap="magma",
        vmin=vlo_s,
        vmax=vhi_s,
        interpolation="nearest",
    )
    ax_sci.set_title("DIA science stamp (64×64)")
    ax_sci.set_xticks([])
    ax_sci.set_yticks([])
    cbar_sci = fig.colorbar(im_sci, ax=ax_sci, fraction=0.046, pad=0.04)
    cbar_sci.set_label("DIA difference flux (e⁻/s; ≈(A(t)−1)·F₀)")

    im_ctx = None
    if side_by_side:
        vlo_c, vhi_c = _robust_vlims(context)
        im_ctx = ax_ctx.imshow(
            context[0],
            origin="lower",
            cmap="magma",
            vmin=vlo_c,
            vmax=vhi_c,
            interpolation="nearest",
        )
        ax_ctx.set_title(
            f"Context stamp ({context.shape[-2]}×{context.shape[-1]})\n"
            f"/{data['context_key']}"
        )
        ax_ctx.set_xticks([])
        ax_ctx.set_yticks([])
        fig.colorbar(im_ctx, ax=ax_ctx, fraction=0.046, pad=0.04)

    # Prominent event-class name at the top, with a smaller detail line below.
    cls_name = _event_class_name(data)
    fig.suptitle(cls_name, fontsize=16, fontweight="bold", y=0.99)
    detail_line = _format_detail_line(data, cadence_days, schema_version)
    title_text = fig.text(
        0.5, 0.95, detail_line, fontsize=7.5, ha="center", va="top",
    )
    caption = ax_sci.text(
        0.02, 0.96, "",
        transform=ax_sci.transAxes,
        color="white",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.5, pad=2.0),
    )

    def update(frame: int):
        im_sci.set_data(science[frame])
        if side_by_side and im_ctx is not None:
            im_ctx.set_data(context[frame])
        n_sat = int(sat[frame].sum()) if sat is not None else 0
        n_cr = int(cr[frame].sum()) if cr is not None else 0
        caption.set_text(
            f"epoch {frame + 1:02d}/{n_epochs:02d}  "
            f"t={frame * cadence_days:.3f} d"
            + (f"\nsat={n_sat} px" if sat is not None else "")
            + (f"  cr={n_cr} px" if cr is not None else "")
        )
        return [im_sci, caption] + ([im_ctx] if im_ctx is not None else [])

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_epochs,
        interval=int(round(1000 / max(1, fps))),
        blit=False,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return anim


def _save_animation(
    anim: animation.FuncAnimation,
    output_path: Path,
    fmt: str,
    fps: int,
) -> Path:
    """Save the animation; fall back to GIF if MP4 writer is missing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4":
        if not animation.FFMpegWriter.isAvailable():
            _log.error(
                "MP4 requested but `ffmpeg` is not available on PATH. "
                "Falling back to GIF."
            )
            fmt = "gif"
            output_path = output_path.with_suffix(".gif")
    try:
        if fmt == "mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(str(output_path), writer=writer, dpi=140)
        else:
            writer = animation.PillowWriter(fps=fps)
            anim.save(str(output_path), writer=writer, dpi=140)
    except Exception as exc:  # broad catch: writer variants raise many types
        _log.error("Animation save failed (%s). Attempting GIF fallback.", exc)
        output_path = output_path.with_suffix(".gif")
        anim.save(
            str(output_path),
            writer=animation.PillowWriter(fps=fps),
            dpi=140,
        )
    return output_path


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        data = _load_event(args.shard_path, args.event_id)
    except (FileNotFoundError, KeyError) as exc:
        _log.error("%s", exc)
        return 2

    schema_version = str(data["attrs"].get("schema_version", "UNKNOWN"))
    if schema_version != "phase3-contract-v1":
        _log.warning(
            "Shard schema_version=%r (expected 'phase3-contract-v1').",
            schema_version,
        )

    cadence_days = _resolve_cadence(args)
    anim = _build_animation(
        data=data,
        schema_version=schema_version,
        cadence_days=cadence_days,
        fps=args.fps,
    )

    suffix = "gif" if args.format == "gif" else "mp4"
    out_name = (
        f"dia__{args.shard_path.stem}__{data['event_id']}_"
        f"{args.fps}fps.{suffix}"
    )
    out_path = args.output_dir / out_name
    final_path = _save_animation(anim, out_path, args.format, args.fps)
    plt.close("all")
    _log.info("wrote %s", final_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
