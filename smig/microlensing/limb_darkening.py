"""Claret 2000 H-band linear LD coefficient interpolator for Roman F146 proxy.

Grid: data/microlensing/claret2000_ld.csv (Teff × log_g × [Fe/H]).
The H-band is a proxy for Roman F146 (see docs/phase3_2_design.md §6.2).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from smig.microlensing.errors import ClaretGridError
from smig.microlensing.event import SourceProperties

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "microlensing" / "claret2000_ld.csv"


@lru_cache(maxsize=None)
def _load_grid() -> dict[str, RegularGridInterpolator]:
    """Load the Claret LD CSV and build one RegularGridInterpolator per band."""
    rows: dict[str, list] = {}
    with _DATA_PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            band = row["band"]
            rows.setdefault(band, []).append(
                (float(row["Teff_K"]), float(row["log_g"]),
                 float(row["FeH"]), float(row["a_linear"]))
            )

    interpolators: dict[str, RegularGridInterpolator] = {}
    for band, entries in rows.items():
        arr = np.array(entries)
        teff_vals = np.unique(arr[:, 0])
        logg_vals = np.unique(arr[:, 1])
        feh_vals  = np.unique(arr[:, 2])

        grid = np.full((len(teff_vals), len(logg_vals), len(feh_vals)), np.nan)
        t_idx = {v: i for i, v in enumerate(teff_vals)}
        g_idx = {v: i for i, v in enumerate(logg_vals)}
        f_idx = {v: i for i, v in enumerate(feh_vals)}
        for teff, logg, feh, a in entries:
            grid[t_idx[teff], g_idx[logg], f_idx[feh]] = a

        interpolators[band] = RegularGridInterpolator(
            (teff_vals, logg_vals, feh_vals),
            grid,
            method="linear",
            bounds_error=True,   # raises ValueError if out of grid
        )
    return interpolators


def _nearest_neighbor(teff_K: float, log_g: float, feh: float, band: str) -> float:
    """Return LD coef of nearest grid point (Euclidean in normalised coordinates)."""
    interps = _load_grid()
    if band not in interps:
        raise ClaretGridError(teff_K, log_g, feh, band)
    interp = interps[band]
    t_grid, g_grid, f_grid = interp.grid
    ti = int(np.argmin(np.abs(t_grid - teff_K)))
    gi = int(np.argmin(np.abs(g_grid - log_g)))
    fi = int(np.argmin(np.abs(f_grid - feh)))
    return float(interp.values[ti, gi, fi])


def get_coefficient(
    source_props: SourceProperties,
    band: str,
    strict: bool = True,
) -> tuple[float, bool]:
    """Return (a_linear, was_fallback) for the given source and photometric band.

    Args:
        source_props: Source stellar parameters.
        band:         Photometric band key (must be present in the LD CSV, e.g. "H").
        strict:       If True and out of grid, raise ClaretGridError.
                      If False and out of grid, return nearest-neighbour with was_fallback=True.

    Returns:
        (a_linear, was_fallback) where a_linear is the Claret 2000 linear LD coefficient
        and was_fallback indicates whether a nearest-neighbour fallback was used.
    """
    interps = _load_grid()
    if band not in interps:
        raise ClaretGridError(source_props.teff_K, source_props.log_g,
                              source_props.metallicity_feh, band)

    try:
        coef = float(interps[band](
            [[source_props.teff_K, source_props.log_g, source_props.metallicity_feh]]
        )[0])
        return coef, False
    except ValueError:
        # Out of grid bounds
        if strict:
            raise ClaretGridError(
                source_props.teff_K, source_props.log_g,
                source_props.metallicity_feh, band
            )
        coef = _nearest_neighbor(source_props.teff_K, source_props.log_g,
                                 source_props.metallicity_feh, band)
        return coef, True
