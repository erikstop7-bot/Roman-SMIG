#!/usr/bin/env python3
"""
scripts/export_science_fits.py
==============================
Export science stamps from SMIG Phase 3 HDF5 shards to viewable FITS files.

Reads ``/science_stamps`` (shape N × n_epochs × 64 × 64, float32) from each
shard and writes one multi-extension FITS per event:

    Primary HDU : empty (conformant header only)
    Extension 1 : 3-D image cube (n_epochs × 64 × 64, float32)

Output directory layout::

    <output-dir>/fits_preview/
        shard_0000__<event_id>.fits
        shard_0001__<event_id>.fits
        ...

FITS header keywords written to every extension
------------------------------------------------
EVENTID   event_id string
SHARD     shard integer index
EPOCHS    number of science epochs (NAXIS3)
NAXIS1    64  (pixel columns)
NAXIS2    64  (pixel rows)
SMIGVER   smig_version from shard attrs
EVCLASS   compact event class string (see mapping below)

Event class mapping (HDF5 label__event_class uint8 → EVCLASS value)
---------------------------------------------------------------------
Precedence: HDF5 ``/label__event_class`` is source-of-truth; manifest.json is
NOT consulted by this exporter (HDF5-first policy).

uint8  EventClass enum name          EVCLASS value
-----  ---------------------------   -------------
0      PSPL                          PSPL
1      FSPL_STAR                     FSPL
2      PLANETARY_CAUSTIC             PLANETARY
3      STELLAR_BINARY                STELLARBIN
4      HIGH_MAGNIFICATION_CUSP       HMC

If the dataset is absent or the value is unrecognized, EVCLASS is set to
"UNKNOWN" and a warning is printed to stderr.

Usage
-----
::

    python scripts/export_science_fits.py --input-dir outputs/phase3_smoke
    python scripts/export_science_fits.py \\
        --input-dir outputs/phase3_smoke \\
        --output-dir /tmp/fits_preview

Exit codes
----------
* ``0`` — all shards exported successfully.
* ``1`` — at least one shard failed (summary printed to stderr).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# uint8 HDF5 value → compact FITS-safe EVCLASS string (8-char FITS keyword limit
# applies to the key, not the value; values may be longer but are kept short for
# readability in QA tools).  Order matches _EVENT_CLASS_ORDER in labels.py —
# DO NOT reorder.
_EVCLASS_MAP: dict[int, str] = {
    0: "PSPL",        # PSPL
    1: "FSPL",        # FSPL_STAR
    2: "PLANETARY",   # PLANETARY_CAUSTIC
    3: "STELLARBIN",  # STELLAR_BINARY
    4: "HMC",         # HIGH_MAGNIFICATION_CUSP
}


def _export_shard(shard_path: Path, output_dir: Path) -> list[Path]:
    """Export all events from one HDF5 shard to FITS files.

    Returns list of written FITS paths.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required: pip install h5py") from exc

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("astropy>=5.0 is required: pip install 'astropy>=5.0'") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with h5py.File(str(shard_path), "r") as h5:
        shard_id = int(h5.attrs.get("shard_id", -1))
        smig_ver = str(h5.attrs.get("smig_version", "unknown"))
        stamps = h5["science_stamps"][:]          # (N, n_epochs, 64, 64) float32
        eids_raw = h5["event_id"][:]

        # Load event class labels (uint8) if present; None signals missing dataset.
        ev_class_raw = (
            h5["label__event_class"][:] if "label__event_class" in h5 else None
        )

        n_events, n_epochs, ny, nx = stamps.shape

        for i in range(n_events):
            eid = eids_raw[i]
            if isinstance(eid, bytes):
                eid = eid.decode("utf-8")
            eid_str = str(eid)

            cube = stamps[i]  # (n_epochs, 64, 64)

            # Resolve EVCLASS from HDF5 label (HDF5-first; manifest not consulted).
            if ev_class_raw is not None:
                ev_uint8 = int(ev_class_raw[i])
                ev_class_str = _EVCLASS_MAP.get(ev_uint8)
                if ev_class_str is None:
                    ev_class_str = "UNKNOWN"
                    print(
                        f"WARNING: event_id={eid_str!r} has unrecognized "
                        f"label__event_class={ev_uint8}; writing EVCLASS=UNKNOWN",
                        file=sys.stderr,
                    )
            else:
                ev_class_str = "UNKNOWN"
                print(
                    f"WARNING: event_id={eid_str!r}: label__event_class dataset "
                    "absent from shard; writing EVCLASS=UNKNOWN",
                    file=sys.stderr,
                )

            # Primary HDU: conformant empty header
            primary = fits.PrimaryHDU()
            primary.header["ORIGIN"] = ("SMIG v2", "SMIG Phase 3.5 dataset builder")
            primary.header["SHARD"] = (shard_id, "HDF5 shard index")
            primary.header["EVENTID"] = (eid_str[:68], "SMIG event identifier")
            primary.header["EVCLASS"] = (ev_class_str, "Microlensing event class")

            # Image HDU: (n_epochs, 64, 64) cube
            img = fits.ImageHDU(data=cube)
            img.header["EVENTID"] = (eid_str[:68], "SMIG event identifier")
            img.header["SHARD"] = (shard_id, "HDF5 shard index")
            img.header["EPOCHS"] = (n_epochs, "Number of science epochs (NAXIS3)")
            img.header["NAXIS1"] = (nx, "Stamp width in pixels")
            img.header["NAXIS2"] = (ny, "Stamp height in pixels")
            img.header["SMIGVER"] = (smig_ver[:68], "smig package version")
            img.header["BUNIT"] = ("electrons", "Science stamp units")
            img.header["EVCLASS"] = (ev_class_str, "Microlensing event class")

            # Sanitize event_id for filename (replace path-unsafe chars)
            safe_eid = eid_str.replace("/", "_").replace(":", "_")
            fits_name = f"shard_{shard_id:04d}__{safe_eid}.fits"
            fits_path = output_dir / fits_name

            hdul = fits.HDUList([primary, img])
            hdul.writeto(str(fits_path), overwrite=True)
            written.append(fits_path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_science_fits",
        description="Export Phase 3 science stamps from HDF5 shards to FITS files.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Phase 3 output directory containing shards/ subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for FITS files. "
            "Defaults to <input-dir>/fits_preview/."
        ),
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    shards_dir = input_dir / "shards"
    if not shards_dir.exists():
        print(
            f"ERROR: shards directory not found: {shards_dir}", file=sys.stderr
        )
        return 1

    output_dir = args.output_dir if args.output_dir else input_dir / "fits_preview"

    shard_files = sorted(shards_dir.glob("shard_*.h5"))
    if not shard_files:
        print(f"ERROR: no shard_*.h5 files found in {shards_dir}", file=sys.stderr)
        return 1

    all_written: list[Path] = []
    errors: list[str] = []

    for shard_path in shard_files:
        try:
            written = _export_shard(shard_path, output_dir)
            all_written.extend(written)
            print(f"  {shard_path.name}: exported {len(written)} event(s)")
        except Exception as exc:
            errors.append(f"{shard_path.name}: {exc}")
            print(f"  ERROR {shard_path.name}: {exc}", file=sys.stderr)

    print(
        f"\nExported {len(all_written)} FITS file(s) to {output_dir}"
    )

    if errors:
        print(f"\nFAILED — {len(errors)} shard(s) had errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
