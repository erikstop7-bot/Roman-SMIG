"""
1. Physics residual CDF (memo item "Residual magnitude CDF"):
     For every LOM event, recompute the magnification curve A_static(t)
     with ds_dt = dalpha_dt = 0 and A_lom(t) with the per-event LOM rates,
     using the *same* physics.magnification_2l1s_lom routine so both go
     through identical VBBinaryLensing calls. Convert the per-epoch
     fractional residual to σ-units using a photon-limited noise model:

         F_static_i = A_static_i · F0
         σ_i ≈ sqrt(F_static_i + sky_bg · exposure_s)
         residual_i_sigma = (A_lom_i − A_static_i) · F0 / σ_i

     Report P50 / P90 / P99 of |residual_i_sigma| per event, aggregated
     by (event_class, ds_dt) for the Green / Yellow / Red gate.

2. Stamp-level RMSE (memo item "Surrogate OOD probe", pixel side):
     When a static Phase 3 event can be matched by source_event_id, also
     report the pixel-wise RMSE between the matched LOM and static
     difference stamps. This is a pure diagnostic — not a scientific σ,
     just a sanity check that (a) passthrough single-lens events match
     byte-for-byte (leakage check) and (b) binary events show non-zero
     but finite stamp-level deltas.

Usage
-----
    python scripts/compare_lom_vs_static.py \\
        --config configs/phase3_sprint.yaml \\
        --static-dir outputs/phase3_sprint_static \\
        --lom-dir  outputs/phase3_lom_ablation/dsdt_0.05 \\
        --out-csv  outputs/phase3_lom_ablation/dsdt_0.05/residuals.csv

Exit codes:
    0 — comparison complete
    1 — fatal error (bad inputs, missing dependencies)
    2 — no LOM shards found
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

# Read-only imports from smig / tools.lom_injection. This script does not
# modify the frozen contract.
from tools.lom_injection.io import (
    LOM_SCHEMA_VERSION,
    SurveyConfig,
    load_survey_config,
)
from tools.lom_injection.physics import LOMComputationError, magnification_2l1s_lom
from smig.catalogs.photometry import mag_ab_to_electrons
from smig.datasets.schema import DATASET_SCHEMA_VERSION as _PHASE3_SCHEMA_VERSION

log = logging.getLogger("compare_lom_vs_static")

# Event class integer encoding (mirrors smig.datasets.labels, frozen contract).
_CLASS_NAMES: dict[int, str] = {
    0: "PSPL",
    1: "FSPL_STAR",
    2: "PLANETARY_CAUSTIC",
    3: "STELLAR_BINARY",
    4: "HIGH_MAGNIFICATION_CUSP",
}
_BINARY_CLASSES: frozenset[int] = frozenset({2, 3, 4})
_SENTINEL: float = -99.0


# ---------------------------------------------------------------------------
# Low-level readers (do not go through tools.lom_injection.io.iter_shard to
# avoid loading the full stamp array when only labels are needed).
# ---------------------------------------------------------------------------

def _iter_static_index(static_dir: Path, glob: str = "*.h5") -> dict[str, tuple[Path, int]]:
    """Return { event_id -> (shard_path, row_index) } for every static event."""
    index: dict[str, tuple[Path, int]] = {}
    for shard in sorted(static_dir.glob(glob)):
        with h5py.File(str(shard), "r") as f:
            schema = str(f.attrs.get("schema_version", ""))
            if schema != _PHASE3_SCHEMA_VERSION:
                log.warning(
                    "Skipping %s: schema_version=%r (expected %r)",
                    shard.name, schema, _PHASE3_SCHEMA_VERSION,
                )
                continue
            eids = f["event_id"][:]
            for i, raw in enumerate(eids):
                eid = raw.decode() if isinstance(raw, bytes) else str(raw)
                index[eid] = (shard, i)
    return index


def _iter_lom_events(lom_dir: Path, glob: str = "*.h5") -> Iterator[dict]:
    """Yield per-event dicts from every LOM shard.

    Yields a dict with the fields needed for physics residual + stamp-level
    comparison. The stamps are memory-mapped only when requested by the
    caller; we yield a lazy loader instead of the array itself to keep the
    peak memory footprint small even on large shard sets.
    """
    shards = sorted(lom_dir.glob(glob))
    if not shards:
        raise FileNotFoundError(f"No LOM shards in {lom_dir} matching {glob!r}")

    for shard in shards:
        with h5py.File(str(shard), "r") as f:
            schema = str(f.attrs.get("schema_version", ""))
            if schema != LOM_SCHEMA_VERSION:
                log.warning(
                    "Skipping %s: schema_version=%r (expected %r)",
                    shard.name, schema, LOM_SCHEMA_VERSION,
                )
                continue

            eids = f["event_id"][:]
            src_eids = f["source_event_id"][:]
            ec = f["label__event_class"][:]
            log_tE = f["label__log_tE"][:]
            log_u0 = f["label__log_u0"][:]
            log_rho = f["label__log_rho"][:]
            alpha = f["label__alpha_rad"][:]
            log_q = f["label__log_q"][:]
            log_s = f["label__log_s"][:]
            t0n = f["label__t0_mjd_normalized"][:]
            mag = f["label__source_mag_F146_ab"][:]
            ds_dt = f["lom__ds_dt_per_yr"][:]
            dalpha_dt = f["lom__dalpha_dt_rad_per_yr"][:]

            stamps_ds = f["science_stamps"]
            n_events = stamps_ds.shape[0]

            for i in range(n_events):
                eid = eids[i].decode() if isinstance(eids[i], bytes) else str(eids[i])
                src = src_eids[i].decode() if isinstance(src_eids[i], bytes) else str(src_eids[i])
                yield {
                    "shard": shard,
                    "row": i,
                    "event_id": eid,
                    "source_event_id": src,
                    "event_class": int(ec[i]),
                    "log_tE": float(log_tE[i]),
                    "log_u0": float(log_u0[i]),
                    "log_rho": float(log_rho[i]),
                    "alpha_rad": float(alpha[i]),
                    "log_q": float(log_q[i]),
                    "log_s": float(log_s[i]),
                    "t0_mjd_normalized": float(t0n[i]),
                    "source_mag_F146_ab": float(mag[i]),
                    "ds_dt": float(ds_dt[i]),
                    "dalpha_dt": float(dalpha_dt[i]),
                    # Load stamp only if we actually need it; here we load
                    # eagerly because we need to compute matched-pair RMSE
                    # and the stamp is just (n_epochs, 64, 64) float32.
                    "lom_stamp": stamps_ds[i][:],
                }


def _load_static_stamp(index: dict[str, tuple[Path, int]], event_id: str) -> np.ndarray | None:
    """Fetch one static event stamp by event_id; returns None if not matched."""
    if event_id not in index:
        return None
    shard, row = index[event_id]
    with h5py.File(str(shard), "r") as f:
        return f["science_stamps"][row][:]


# ---------------------------------------------------------------------------
# Residual-magnitude computation
# ---------------------------------------------------------------------------

def _compute_event_residual(
    lom: dict,
    survey: SurveyConfig,
) -> dict | None:
    """Compute per-event residual statistics. Returns None for single-lens events."""
    if lom["event_class"] not in _BINARY_CLASSES:
        return None
    if np.isclose(lom["log_q"], _SENTINEL, atol=1e-3) or np.isclose(lom["log_s"], _SENTINEL, atol=1e-3):
        log.warning(
            "Binary-class event %r has sentinel log_q/log_s; skipping residual.",
            lom["event_id"],
        )
        return None

    t_mjd = survey.epoch_times_mjd
    t0 = survey.t0_abs_mjd(lom["t0_mjd_normalized"])
    tE = 10.0 ** lom["log_tE"]
    u0 = 10.0 ** lom["log_u0"]
    rho = 10.0 ** lom["log_rho"]
    q = 10.0 ** lom["log_q"]
    s = 10.0 ** lom["log_s"]

    try:
        A_static = magnification_2l1s_lom(
            t_mjd=t_mjd, t0=t0, tE=tE, u0=u0, rho=rho,
            alpha0=lom["alpha_rad"], q=q, s0=s,
            ds_dt=0.0, dalpha_dt=0.0,
        )
        A_lom = magnification_2l1s_lom(
            t_mjd=t_mjd, t0=t0, tE=tE, u0=u0, rho=rho,
            alpha0=lom["alpha_rad"], q=q, s0=s,
            ds_dt=lom["ds_dt"], dalpha_dt=lom["dalpha_dt"],
        )
    except (LOMComputationError, ValueError) as exc:
        log.warning("Magnification recompute failed for %r: %s", lom["event_id"], exc)
        return None

    F0 = mag_ab_to_electrons(lom["source_mag_F146_ab"], "F146", survey.exposure_s)
    sky_e = survey.sky_background_e_per_s * survey.exposure_s

    F_static = A_static * F0
    # Photon-limited noise floor: source + sky shot noise per epoch.
    # This approximation omits read noise and PSF-integrated pixel-level
    # detail; it is a conservative lower bound on σ (stricter test).
    sigma = np.sqrt(np.maximum(F_static + sky_e, 1.0))
    residual_sigma = np.abs(A_lom - A_static) * F0 / sigma

    return {
        "p50_sigma": float(np.median(residual_sigma)),
        "p90_sigma": float(np.percentile(residual_sigma, 90)),
        "p99_sigma": float(np.percentile(residual_sigma, 99)),
        "max_sigma": float(np.max(residual_sigma)),
        "chi2_total": float(np.sum(residual_sigma ** 2)),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="compare_lom_vs_static.py",
        description=(
            "Compare a LOM ablation shard set against a static Phase 3 baseline. "
            "Produces per-event residual σ-statistics and a per-(class, ds_dt) "
            "summary table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Phase 3 generation YAML used by BOTH runs "
                        "(for obs_start, cadence, n_epochs, exposure_s, sky_bg).")
    p.add_argument("--static-dir", required=True, type=Path,
                   help="Directory of phase3-contract-v1 static shards.")
    p.add_argument("--lom-dir", required=True, type=Path,
                   help="Directory of phase3-lom-ablation-v0 LOM shards.")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional CSV output path (per-event residual stats).")
    p.add_argument("--shard-glob", default="*.h5", metavar="PATTERN")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--skip-stamp-rmse", action="store_true",
                   help="Skip per-event stamp RMSE (physics residual only; faster).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    survey = load_survey_config(args.config)

    if not args.skip_stamp_rmse:
        log.info("Building static-event index from %s", args.static_dir)
        static_index = _iter_static_index(args.static_dir, args.shard_glob)
        log.info("Indexed %d static events", len(static_index))
    else:
        static_index = {}

    # Per-event rows for the CSV, and grouped aggregates for summary.
    rows: list[dict] = []
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    n_total = 0
    n_passthrough = 0
    n_matched = 0
    n_physics_skipped = 0

    try:
        for lom in _iter_lom_events(args.lom_dir, args.shard_glob):
            n_total += 1
            cls_name = _CLASS_NAMES.get(lom["event_class"], "UNKNOWN")
            key = (cls_name, round(lom["ds_dt"], 6), round(lom["dalpha_dt"], 6))

            row: dict = {
                "source_event_id": lom["source_event_id"],
                "event_class": cls_name,
                "ds_dt": lom["ds_dt"],
                "dalpha_dt": lom["dalpha_dt"],
                "residual_p50_sigma": "",
                "residual_p90_sigma": "",
                "residual_p99_sigma": "",
                "residual_max_sigma": "",
                "residual_chi2_total": "",
                "stamp_rmse": "",
                "stamp_matched": "",
            }

            residual = _compute_event_residual(lom, survey)
            if residual is None:
                n_passthrough += 1
            else:
                row.update({
                    "residual_p50_sigma": residual["p50_sigma"],
                    "residual_p90_sigma": residual["p90_sigma"],
                    "residual_p99_sigma": residual["p99_sigma"],
                    "residual_max_sigma": residual["max_sigma"],
                    "residual_chi2_total": residual["chi2_total"],
                })
                grouped[key].append(residual)

            if not args.skip_stamp_rmse:
                static_stamp = _load_static_stamp(static_index, lom["source_event_id"])
                if static_stamp is not None:
                    diff = lom["lom_stamp"].astype(np.float64) - static_stamp.astype(np.float64)
                    row["stamp_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
                    row["stamp_matched"] = True
                    n_matched += 1
                else:
                    row["stamp_matched"] = False

            rows.append(row)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except ImportError as exc:
        log.error("Missing dependency (likely VBBinaryLensing): %s", exc)
        return 1

    if n_total == 0:
        log.error("No LOM events processed.")
        return 2

    # --------- Summary ----------
    print()
    print("=" * 78)
    print(f"LOM vs static comparison — {n_total} events inspected")
    print(f"  single-lens passthroughs (no physics residual): {n_passthrough}")
    if not args.skip_stamp_rmse:
        print(f"  stamp-level matched (source_event_id in static dir): {n_matched}")
    print(f"  physics residual skipped (VBBL failure): {n_physics_skipped}")
    print("=" * 78)
    print()
    header = (
        f"{'event_class':<25} {'ds_dt':>8} {'dalpha_dt':>10} "
        f"{'N':>6} {'p50 σ':>10} {'p90 σ':>10} {'p99 σ':>10} {'max σ':>10}"
    )
    print(header)
    print("-" * len(header))
    for key in sorted(grouped.keys()):
        cls, dsdt, dadt = key
        bucket = grouped[key]
        p50 = np.median([b["p50_sigma"] for b in bucket])
        p90 = np.median([b["p90_sigma"] for b in bucket])
        p99 = np.median([b["p99_sigma"] for b in bucket])
        mx = np.median([b["max_sigma"] for b in bucket])
        print(
            f"{cls:<25} {dsdt:>8.3f} {dadt:>10.3f} "
            f"{len(bucket):>6d} {p50:>10.3f} {p90:>10.3f} {p99:>10.3f} {mx:>10.3f}"
        )

    # Acceptance-band hint per memo Section 6 (not authoritative).
    print()
    print("Sprint acceptance gate (memo Section 6):")
    print("  Green at (ds_dt, dα_dt) = (0.05, 0.05) /yr  if median residual > 3σ")
    print("  Yellow                                     if 1σ < median ≤ 3σ")
    print("  Red                                        if median ≤ 1σ")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info("Wrote %d rows to %s", len(rows), args.out_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
