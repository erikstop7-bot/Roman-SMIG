#!/usr/bin/env python3
"""
scripts/summarize_phase2_qa_batch.py

Reads an existing pilot-batch output directory and prints aggregate QA stats
from the saved .npy arrays and metadata.json files without re-running any
simulation.  Useful for checking stats after generation or re-flagging with
different thresholds.

Usage:
    python3 scripts/summarize_phase2_qa_batch.py outputs/phase2_pilot_qa/

    # Re-apply stricter thresholds without recomputing from raw arrays:
    python3 scripts/summarize_phase2_qa_batch.py outputs/phase2_pilot_qa/ \\
        --null-central-snr-warn 2.0 --rim-ratio-low 0.7 --rim-ratio-high 1.5

    # Force recomputation from raw .npy files and rewrite the summary:
    python3 scripts/summarize_phase2_qa_batch.py outputs/phase2_pilot_qa/ \\
        --recompute --rewrite-summary
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def _load_qa_module():
    """Load generate_phase2_qa_stamps without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "_qa_mod",
        Path(__file__).parent / "generate_phase2_qa_stamps.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Read an existing pilot-batch output directory and print aggregate "
            "QA stats.  By default loads pilot_summary.json and re-applies flags; "
            "use --recompute to rebuild from raw .npy files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("output_dir", help="Pilot-batch output directory to summarize.")
    p.add_argument(
        "--central-aperture-radius", type=int, default=3,
        help="Circular aperture radius in pixels for central residual metrics.",
    )
    p.add_argument(
        "--random-apertures", type=int, default=500,
        help="Number of random off-centre apertures for null-event percentile.",
    )
    p.add_argument(
        "--null-central-snr-warn", type=float, default=3.0,
        help="Flag null epochs with |central_aperture_snr| above this value.",
    )
    p.add_argument(
        "--rim-ratio-low", type=float, default=0.5,
        help="Flag epochs with rim/interior std ratio below this.",
    )
    p.add_argument(
        "--rim-ratio-high", type=float, default=2.0,
        help="Flag epochs with rim/interior std ratio above this.",
    )
    p.add_argument(
        "--max-saturation-fraction", type=float, default=0.0,
        help="Flag epochs with saturation_pixel_fraction above this.",
    )
    p.add_argument(
        "--max-cr-fraction", type=float, default=None,
        help="If set, flag epochs with cr_pixel_fraction above this.",
    )
    p.add_argument(
        "--recompute",
        action="store_true", default=False,
        help=(
            "Ignore pilot_summary.json and recompute all stats from the raw "
            ".npy arrays and metadata.json files in each event subdirectory."
        ),
    )
    p.add_argument(
        "--rewrite-summary",
        action="store_true", default=False,
        help="Rewrite pilot_summary.json and pilot_summary.csv with the recomputed rows.",
    )
    return p


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------

def _scan_event_dirs(output_dir: Path) -> list[dict]:
    """Return a list of event info dicts for subdirs that have diff_stamps.npy."""
    events = []
    for ev_dir in sorted(output_dir.iterdir()):
        if not ev_dir.is_dir():
            continue
        diff_path = ev_dir / "diff_stamps.npy"
        if not diff_path.exists():
            continue
        meta: dict = {}
        meta_path = ev_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        events.append({
            "event_id": meta.get("event_id", ev_dir.name),
            "event_type": meta.get("event_type", "unknown"),
            "diff_path": diff_path,
            "sat_path": ev_dir / "sat_stamps.npy",
            "cr_path": ev_dir / "cr_stamps.npy",
            "meta": meta,
        })
    return events


# ---------------------------------------------------------------------------
# Recompute from raw arrays
# ---------------------------------------------------------------------------

def _recompute_rows(
    events: list[dict],
    qa,
    aperture_radius: int,
    n_random_apertures: int,
    thresholds: dict,
) -> list[dict]:
    """Recompute per-epoch rows from raw .npy files and metadata."""
    all_rows: list[dict] = []

    for ev in events:
        event_id = ev["event_id"]
        event_type = ev["event_type"]
        meta = ev["meta"]

        diff_stamps = np.load(ev["diff_path"])
        sat_stamps = (
            np.load(ev["sat_path"])
            if ev["sat_path"].exists()
            else np.zeros_like(diff_stamps, dtype=np.float32)
        )
        cr_stamps = (
            np.load(ev["cr_path"])
            if ev["cr_path"].exists()
            else np.zeros_like(diff_stamps, dtype=np.float32)
        )

        # Build a minimal cfg-like namespace from metadata fields.
        dia_ns = types.SimpleNamespace(
            reference_processing=meta.get("reference_processing", "unknown"),
            reference_jitter_mode=meta.get("reference_jitter_mode", "unknown"),
        )
        psf_ns = types.SimpleNamespace(
            fov_native_pixels=meta.get("psf_fov_native_pixels"),
            oversample=meta.get("psf_oversample"),
            n_wavelengths=meta.get("psf_n_wavelengths"),
        )
        cfg_ns = types.SimpleNamespace(dia=dia_ns, psf=psf_ns)

        # dia_method per epoch: prefer epoch_stats, fall back to event-level field.
        ep_stats_list = meta.get("epoch_stats", [])

        for ep_idx, diff in enumerate(diff_stamps):
            sat_ep = sat_stamps[ep_idx] if sat_stamps.ndim >= 3 else sat_stamps
            cr_ep = cr_stamps[ep_idx] if cr_stamps.ndim >= 3 else cr_stamps

            dia_method = None
            if ep_idx < len(ep_stats_list):
                ep_s = ep_stats_list[ep_idx]
                dia_method = ep_s.get("dia_method") or meta.get("dia_method_provenance")
            else:
                dia_method = meta.get("dia_method_provenance")

            prov_ns = types.SimpleNamespace(dia_method=dia_method)

            row = qa._compute_pilot_epoch_row(
                event_id=event_id,
                event_type=event_type,
                epoch=ep_idx,
                diff=diff,
                sat=sat_ep,
                cr=cr_ep,
                prov=prov_ns,
                cfg=cfg_ns,
                aperture_radius=aperture_radius,
                n_random_apertures=n_random_apertures,
            )
            row["qa_flags"] = qa._pilot_qa_flags(
                row,
                null_central_snr_warn=thresholds["null_central_snr_warn"],
                rim_ratio_low=thresholds["rim_ratio_low"],
                rim_ratio_high=thresholds["rim_ratio_high"],
                max_saturation_fraction=thresholds["max_saturation_fraction"],
                max_cr_fraction=thresholds["max_cr_fraction"],
            )
            row["central_signal_dominant"] = qa._central_signal_dominant(
                row,
                rim_ratio_low=thresholds["rim_ratio_low"],
                snr_thresh=thresholds["null_central_snr_warn"],
            )
            all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Re-flag existing summary rows
# ---------------------------------------------------------------------------

def _reflag_rows(rows: list[dict], qa, thresholds: dict) -> list[dict]:
    """Re-apply _pilot_qa_flags and _central_signal_dominant to existing summary rows."""
    for row in rows:
        row["qa_flags"] = qa._pilot_qa_flags(
            row,
            null_central_snr_warn=thresholds["null_central_snr_warn"],
            rim_ratio_low=thresholds["rim_ratio_low"],
            rim_ratio_high=thresholds["rim_ratio_high"],
            max_saturation_fraction=thresholds["max_saturation_fraction"],
            max_cr_fraction=thresholds["max_cr_fraction"],
        )
        row["central_signal_dominant"] = qa._central_signal_dominant(
            row,
            rim_ratio_low=thresholds["rim_ratio_low"],
            snr_thresh=thresholds["null_central_snr_warn"],
        )
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        sys.exit(f"ERROR: output_dir not found: {output_dir}")

    qa = _load_qa_module()

    thresholds = {
        "null_central_snr_warn": args.null_central_snr_warn,
        "rim_ratio_low": args.rim_ratio_low,
        "rim_ratio_high": args.rim_ratio_high,
        "max_saturation_fraction": args.max_saturation_fraction,
        "max_cr_fraction": args.max_cr_fraction,
    }

    summary_path = output_dir / "pilot_summary.json"

    if not args.recompute and summary_path.exists():
        print(f"Loading existing pilot_summary.json: {summary_path}")
        try:
            rows = json.loads(summary_path.read_text())
        except Exception as exc:
            sys.exit(f"ERROR: could not parse {summary_path}: {exc}")
        if not rows:
            sys.exit("ERROR: pilot_summary.json is empty.")
        rows = _reflag_rows(rows, qa, thresholds)
        if args.rewrite_summary:
            j, c = qa._save_pilot_summary(rows, output_dir)
            print(f"Rewrote pilot_summary.json : {j}")
            print(f"Rewrote pilot_summary.csv  : {c}")
        qa._print_pilot_console_summary(rows, thresholds)
        return

    # Recompute from raw .npy files
    print(f"Scanning output_dir: {output_dir}")
    events = _scan_event_dirs(output_dir)
    if not events:
        sys.exit(f"ERROR: no event subdirectories with diff_stamps.npy found in {output_dir}")

    print(f"Found {len(events)} event(s). Recomputing stats ...")
    rows = _recompute_rows(
        events, qa,
        aperture_radius=args.central_aperture_radius,
        n_random_apertures=args.random_apertures,
        thresholds=thresholds,
    )
    if not rows:
        sys.exit("ERROR: no epochs extracted from any event directory.")

    if args.rewrite_summary:
        j, c = qa._save_pilot_summary(rows, output_dir)
        print(f"Wrote pilot_summary.json : {j}")
        print(f"Wrote pilot_summary.csv  : {c}")

    qa._print_pilot_console_summary(rows, thresholds)


if __name__ == "__main__":
    main()
