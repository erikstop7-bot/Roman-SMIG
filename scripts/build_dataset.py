"""Phase 3.5 dataset builder CLI.

Usage
-----
    python scripts/build_dataset.py --config configs/phase3_smoke.yaml [--resume]

Prerequisites
-------------
    pip install -e '.[phase2]'   # galsim, webbpsf, pandas required by workers

The ``if __name__ == "__main__"`` guard is required for Windows (spawn-based
ProcessPoolExecutor) to prevent the build from re-running in every worker
subprocess.
"""
import argparse
import logging
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_dataset.py",
        description=(
            "Build a Phase 3.5 SMIG Roman WFI dataset shard set.\n\n"
            "Prerequisites: pip install -e '.[phase2]' "
            "(installs galsim, webbpsf, pandas)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Phase 3.5 PhaseConfig YAML file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted build. Skips already-committed events "
            "and cleans up orphaned .tmp shard files."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    prog_group = parser.add_mutually_exclusive_group()
    prog_group.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        default=None,
        help=(
            "Always show progress bar / periodic log, even for small runs "
            "or when stderr is not a TTY."
        ),
    )
    prog_group.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Suppress all progress output.",
    )
    parser.add_argument(
        "--progress-min-events",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Auto-mode threshold: show progress only when remaining events >= N "
            "(default: 100).  Ignored when --progress or --no-progress is given."
        ),
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=1.0,
        metavar="S",
        help=(
            "Seconds between periodic progress lines when stderr is not a TTY "
            "(default: 1.0)."
        ),
    )
    return parser.parse_args(argv)


def _configure_stpsf_from_env() -> None:
    """Apply STPSF_PATH if provided, without importing stpsf at module import time."""
    stpsf_path = os.environ.get("STPSF_PATH")
    if not stpsf_path:
        return
    os.environ["STPSF_PATH"] = stpsf_path
    try:
        import stpsf
    except ImportError:
        logging.getLogger(__name__).warning(
            "STPSF_PATH is set but 'stpsf' is not importable; continuing without applying "
            "stpsf.conf.STPSF_PATH. Install stpsf or unset STPSF_PATH to suppress this warning."
        )
        return
    stpsf.conf.STPSF_PATH = stpsf_path


def _silence_psf_backend_loggers() -> None:
    """Reduce poppy/webbpsf/stpsf log verbosity for production runs.

    The PSF backends emit one INFO line per monochromatic propagation
    ("Initialized OpticalSystem", "Propagating wavelength = ...").  At
    n_wavelengths=10 and 60 PSF calls per event this produces ~600 log lines
    per event with no operator value once the cache is warm, and slows
    throughput when stderr is piped or written to a slow terminal.

    Production-meaningful warnings/errors continue to surface because we set
    the level to WARNING, not ERROR.
    """
    for name in ("poppy", "webbpsf", "stpsf"):
        logging.getLogger(name).setLevel(logging.WARNING)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _silence_psf_backend_loggers()
    _configure_stpsf_from_env()

    from smig.datasets.config import PhaseConfig
    from smig.datasets.generator import DatasetBuilder

    cfg = PhaseConfig.from_yaml(args.config)
    builder = DatasetBuilder(cfg, resume=args.resume)
    builder.build(
        show_progress=args.show_progress,
        progress_min_events=args.progress_min_events,
        progress_interval_s=args.progress_interval_s,
    )
    sys.exit(0)
