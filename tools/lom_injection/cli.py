"""Command-line interface for the SMIG Phase 3 LOM ablation sprint.

Usage
-----
From the repository root::

    python -m tools.lom_injection.cli \\
        --config configs/phase3_smoke.yaml \\
        --input-dir outputs/phase3_smoke \\
        --output-dir outputs/phase3_lom_ablation \\
        --ds-dt 0.05 \\
        --dalpha-dt 0.05

To run multiple (ds_dt, dalpha_dt) settings in sequence (Section 6 sprint)::

    for setting in 0.02 0.05 0.10; do
        python -m tools.lom_injection.cli \\
            --config configs/phase3_smoke.yaml \\
            --input-dir outputs/phase3_smoke \\
            --output-dir outputs/phase3_lom_ablation/dsdt_${setting} \\
            --ds-dt ${setting} --dalpha-dt ${setting}
    done

Environment Variables
---------------------
STPSF_PATH :
    Required by WebbPSF (via STPSF) when SceneSimulator calls the PSF
    provider.  Must point to the STPSF data directory.  Example::

        export STPSF_PATH=/data/stpsf-data

    If unset, SceneSimulator will raise an ImportError or FileNotFoundError
    when initialising the STPSFProvider.  The lom_injection tool itself does
    not read this variable; it is consumed internally by smig.optics.psf.

SMIG_LOG_LEVEL :
    Overrides --log-level if set.  Useful in cluster job scripts.

Exit Codes
----------
0   All shards processed (partial failures within shards are acceptable).
1   Fatal error before processing began (bad config, missing prerequisite,
    equivalence check failed).
2   No input shards found.
"""
from __future__ import annotations
# STPSF / WebbPSF: set before any code path imports smig.optics (via runner).
# Match scripts/build_dataset.py so lom_injection and build_dataset use the same data path.
import os as _os

_stpsf_data = _os.path.expanduser(
    _os.environ.get(
        "STPSF_PATH",
        _os.path.join(_os.path.expanduser("~"), "data", "stpsf-data"),
    )
)
_os.environ["STPSF_PATH"] = _stpsf_data
import stpsf as _stpsf  # noqa: E402
_stpsf.conf.STPSF_PATH = _stpsf_data
import argparse
import logging
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lom_injection",
        description=(
            "SMIG Phase 3 LOM ablation tool. "
            "Reads phase3-contract-v1 HDF5 shards, applies linear lens orbital "
            "motion to binary events via VBBinaryLensing, and writes parallel "
            "phase3-lom-ablation-v0 shards."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required arguments ----
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        metavar="YAML",
        help=(
            "Path to the Phase 3 generation YAML "
            "(e.g. configs/phase3_smoke.yaml). "
            "Must contain: observation_start_mjd, cadence_days, "
            "n_science_epochs, sky_background_e_per_s, simulation_config_path."
        ),
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing phase3-contract-v1 HDF5 shards to process.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "Output directory for phase3-lom-ablation-v0 shards. "
            "Created if absent. Must not be the same as --input-dir."
        ),
    )
    p.add_argument(
        "--ds-dt",
        required=True,
        type=float,
        metavar="FLOAT",
        help=(
            "Rate of change of binary separation (θ_E / yr). "
            "Sprint bracket values: 0.02, 0.05, 0.10. "
            "Positive → separation increasing over time. "
            "Set to 0.0 for a static (no-LOM) control run."
        ),
    )
    p.add_argument(
        "--dalpha-dt",
        required=True,
        type=float,
        metavar="FLOAT",
        help=(
            "Rate of change of binary axis angle (rad / yr). "
            "Sprint bracket values: 0.02, 0.05, 0.10. "
            "Set to 0.0 for a static (no-LOM) control run."
        ),
    )

    # ---- Optional arguments ----
    p.add_argument(
        "--shard-glob",
        default="*.h5",
        metavar="PATTERN",
        help="Glob pattern for selecting input shards within --input-dir.",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Master RNG seed for SceneSimulator.  Default (omitted): use the "
            "master_seed field from the Phase 3 YAML config — the only setting "
            "that produces stochastic state matching the source corpus, so "
            "pixel-level static-vs-LOM comparisons are not confounded by "
            "crowding/jitter/detector noise.  Pass an explicit integer only "
            "for deliberate seed-divergence experiments."
        ),
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("SMIG_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Overridden by SMIG_LOG_LEVEL env var.",
    )
    p.add_argument(
        "--skip-prerequisite-check",
        action="store_true",
        help=(
            "Skip the SceneSimulator.simulate_event introspection check. "
            "Use only in unit-test environments where no rendering is performed."
        ),
    )
    p.add_argument(
        "--skip-equivalence-check",
        action="store_true",
        help=(
            "Skip the static-equivalence cross-check (physics.verify_static_equivalence). "
            "Use only in environments where VBBinaryLensing is absent."
        ),
    )

    return p


def _validate_args(args: argparse.Namespace) -> list[str]:
    """Return a list of human-readable validation errors (empty = OK)."""
    errors: list[str] = []

    if not args.config.exists():
        errors.append(f"--config path does not exist: {args.config}")

    if not args.input_dir.exists():
        errors.append(f"--input-dir does not exist: {args.input_dir}")
    elif args.input_dir.resolve() == args.output_dir.resolve():
        errors.append("--input-dir and --output-dir must not be the same path.")

    return errors


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns an integer exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    log = logging.getLogger("lom_injection.cli")

    errors = _validate_args(args)
    if errors:
        for err in errors:
            log.error("Argument error: %s", err)
        return 1

    log.info(
        "Starting LOM ablation sprint: ds_dt=%.4f θ_E/yr, dalpha_dt=%.4f rad/yr",
        args.ds_dt, args.dalpha_dt,
    )
    log.info("Input:  %s", args.input_dir)
    log.info("Output: %s", args.output_dir)
    log.info("Config: %s", args.config)

    from tools.lom_injection.runner import run_sprint  # noqa: PLC0415

    try:
        results = run_sprint(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            ds_dt=args.ds_dt,
            dalpha_dt=args.dalpha_dt,
            master_seed=args.master_seed,
            glob_pattern=args.shard_glob,
            skip_prerequisite_check=args.skip_prerequisite_check,
            skip_equivalence_check=args.skip_equivalence_check,
        )
    except FileNotFoundError as exc:
        log.error("No input shards found: %s", exc)
        return 2
    except (RuntimeError, ImportError) as exc:
        log.error("Fatal error: %s", exc)
        return 1
    except AssertionError as exc:
        log.error("Static equivalence check failed: %s", exc)
        return 1

    # Print a summary table.
    total_read = sum(r.n_events_read for r in results)
    total_written = sum(r.n_written for r in results)
    total_failed = sum(r.n_failed for r in results)
    total_passthrough = sum(r.n_single_lens_passthrough for r in results)

    log.info("=" * 60)
    log.info(
        "Sprint summary: %d shards processed.", len(results)
    )
    log.info("  Events read:               %6d", total_read)
    log.info("  Events written:            %6d", total_written)
    log.info("  Single-lens passthroughs:  %6d", total_passthrough)
    log.info("  Events failed / skipped:   %6d", total_failed)
    if total_read > 0:
        log.info(
            "  Success rate:              %5.1f%%",
            100.0 * total_written / total_read,
        )
    log.info("Output schema: phase3-lom-ablation-v0")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
