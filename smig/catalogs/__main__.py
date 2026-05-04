"""
smig/catalogs/__main__.py
==========================
CLI entry point: ``python -m smig.catalogs``

Usage
-----
::

    python -m smig.catalogs --l 1.0 --b -3.0 --fov 0.28 --provider synthetic

Prints the head and column dtypes of the projected DataFrame for the
requested field.  Intended for smoke-testing the full catalog → adapter →
renderer-compatible DataFrame pipeline without requiring a real catalog.

Supported --provider values
---------------------------
- ``synthetic`` — :class:`~smig.catalogs.synthetic.SyntheticCatalogProvider`
  (default; no external data required)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from smig.catalogs.adapter import project_to_sca_dataframe
from smig.catalogs.sampler import sample_field
from smig.catalogs.synthetic import SyntheticCatalogProvider

_DEFAULT_EXPOSURE_S: float = 139.8
_DEFAULT_N_STARS: int = 50
_DEFAULT_SCA_ID: int = 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m smig.catalogs",
        description="Smoke-test the Phase 3.1 catalog → DataFrame pipeline.",
    )
    p.add_argument("--l", dest="l_deg", type=float, required=True,
                   help="Galactic longitude of field centre (degrees).")
    p.add_argument("--b", dest="b_deg", type=float, required=True,
                   help="Galactic latitude of field centre (degrees).")
    p.add_argument("--fov", dest="fov_deg", type=float, default=0.28,
                   help="Square FOV full width in degrees (default: 0.28).")
    p.add_argument("--provider", choices=["synthetic"], default="synthetic",
                   help="Catalog provider to use (default: synthetic).")
    p.add_argument("--n-stars", type=int, default=_DEFAULT_N_STARS,
                   help="Number of synthetic stars (default: 50).")
    p.add_argument("--exposure-s", type=float, default=_DEFAULT_EXPOSURE_S,
                   help="Exposure time in seconds (default: 139.8).")
    p.add_argument("--sca-id", type=int, default=_DEFAULT_SCA_ID,
                   help="SCA identifier (default: 1).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.provider == "synthetic":
        provider = SyntheticCatalogProvider(n_stars=args.n_stars)
    else:
        print(f"Unknown provider: {args.provider}", file=sys.stderr)
        sys.exit(1)

    stars = sample_field(
        provider, args.l_deg, args.b_deg, args.fov_deg, rng, use_cache=False
    )

    df = project_to_sca_dataframe(
        stars,
        sca_id=args.sca_id,
        field_center_l_deg=args.l_deg,
        field_center_b_deg=args.b_deg,
        exposure_s=args.exposure_s,
    )

    print(f"Projected {len(df)} stars into SCA{args.sca_id:02d} pixel space.")
    print()
    print("Head (5 rows):")
    print(df.head().to_string())
    print()
    print("Column dtypes:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
