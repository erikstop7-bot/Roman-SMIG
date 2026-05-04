"""
smig/sensor/calibration/ipc_kernels.py
=======================================
Synthetic HDF5 IPC kernel generator and bilinear-interpolation loader.

Leaf module — must not import any sibling sensor module.

HDF5 Layout
-----------
Each SCA is stored under ``/sca_{id}/`` with:

- ``kernels``: float64 array of shape ``(ny, nx, 9, 9)``
- ``field_x``: float64 array of shape ``(nx,)`` — normalised x-coords [0, 1]
- ``field_y``: float64 array of shape ``(ny,)`` — normalised y-coords [0, 1]

``ny`` and ``nx`` define the spatial calibration grid.  Each grid point
stores one 9x9 IPC kernel.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def generate_synthetic_ipc_hdf5(
    path: Path,
    sca_ids: tuple[int, ...] = (1,),
    grid_ny: int = 4,
    grid_nx: int = 4,
    kernel_size: int = 9,
    base_alpha: float = 0.02,
    rng: np.random.Generator | None = None,
) -> None:
    """Generate a synthetic HDF5 IPC kernel calibration file.

    Creates spatially-varying 9x9 kernels on a regular grid for each
    requested SCA.  The variation is a smooth gradient in the centre-
    pixel alpha from ``base_alpha`` (top-left) to ``1.5 * base_alpha``
    (bottom-right), with small random perturbations per grid point.

    Parameters
    ----------
    path:
        Output HDF5 file path.  Parent directory must exist.
    sca_ids:
        SCA identifiers to generate.
    grid_ny, grid_nx:
        Number of calibration grid points along y and x.
    kernel_size:
        Side length of each square kernel (must be odd).
    base_alpha:
        Baseline IPC coupling coefficient at the top-left grid corner.
    rng:
        NumPy random generator for reproducible perturbations.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if rng is None:
        rng = np.random.default_rng(0)

    field_y = np.linspace(0.0, 1.0, grid_ny)
    field_x = np.linspace(0.0, 1.0, grid_nx)

    with h5py.File(path, "w") as f:
        for sca_id in sca_ids:
            grp = f.create_group(f"sca_{sca_id}")
            kernels = np.zeros((grid_ny, grid_nx, kernel_size, kernel_size),
                               dtype=np.float64)
            centre = kernel_size // 2

            for iy in range(grid_ny):
                for ix in range(grid_nx):
                    # Smooth spatial gradient + small random perturbation.
                    frac_y = field_y[iy]
                    frac_x = field_x[ix]
                    alpha = base_alpha * (1.0 + 0.5 * (frac_x + frac_y) / 2.0)
                    alpha += rng.normal(0.0, base_alpha * 0.01)
                    alpha = max(alpha, 0.0)

                    k = np.zeros((kernel_size, kernel_size), dtype=np.float64)
                    # Nearest-neighbour coupling (horizontal/vertical).
                    k[centre - 1, centre] = alpha
                    k[centre + 1, centre] = alpha
                    k[centre, centre - 1] = alpha
                    k[centre, centre + 1] = alpha
                    # Diagonal coupling (weaker).
                    diag = alpha * 0.1
                    k[centre - 1, centre - 1] = diag
                    k[centre - 1, centre + 1] = diag
                    k[centre + 1, centre - 1] = diag
                    k[centre + 1, centre + 1] = diag
                    # Centre pixel: remainder so kernel sums to 1.
                    k[centre, centre] = 1.0 - k.sum()
                    kernels[iy, ix] = k

            grp.create_dataset("kernels", data=kernels)
            grp.create_dataset("field_x", data=field_x)
            grp.create_dataset("field_y", data=field_y)


def load_interpolated_kernel(
    path: Path,
    sca_id: int,
    field_position: tuple[float, float],
) -> np.ndarray:
    """Load and bilinearly interpolate an IPC kernel from an HDF5 file.

    Parameters
    ----------
    path:
        Path to the HDF5 calibration file.
    sca_id:
        SCA identifier (must match a ``/sca_{id}/`` group in the file).
    field_position:
        ``(x, y)`` normalised coordinates within the SCA (0.0–1.0).

    Returns
    -------
    np.ndarray
        Single interpolated 9x9 kernel, normalised to sum to 1.0.

    Raises
    ------
    ValueError
        If ``field_position`` coordinates are outside [0, 1] or the
        SCA group is missing.
    """
    fx, fy = field_position
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        raise ValueError(
            f"field_position must be in [0, 1], got ({fx}, {fy})"
        )

    with h5py.File(path, "r") as f:
        grp_name = f"sca_{sca_id}"
        if grp_name not in f:
            raise ValueError(
                f"SCA group '{grp_name}' not found in {path}. "
                f"Available groups: {list(f.keys())}"
            )
        grp = f[grp_name]
        # TODO: Memory optimization - slice specific neighborhood instead of
        # loading full (ny, nx, 9, 9) array into RAM.  Replace the line below
        # with h5py slicing: grp["kernels"][iy0:iy1+1, ix0:ix1+1, ...] once
        # the bracket indices are computed before opening the file.
        kernels = grp["kernels"][:]      # (ny, nx, K, K) — full array loaded
        field_x = grp["field_x"][:]      # (nx,)
        field_y = grp["field_y"][:]      # (ny,)

    # Find the bounding grid indices for bilinear interpolation.
    ix0, tx = _find_bracket(field_x, fx)
    iy0, ty = _find_bracket(field_y, fy)

    ix1 = min(ix0 + 1, len(field_x) - 1)
    iy1 = min(iy0 + 1, len(field_y) - 1)

    # Bilinear interpolation of the 9x9 kernels.
    k00 = kernels[iy0, ix0]
    k01 = kernels[iy0, ix1]
    k10 = kernels[iy1, ix0]
    k11 = kernels[iy1, ix1]

    kernel = (
        (1.0 - ty) * ((1.0 - tx) * k00 + tx * k01)
        + ty * ((1.0 - tx) * k10 + tx * k11)
    )

    # Safety check: interpolated kernel must have a positive, finite sum.
    kernel_sum = kernel.sum()
    if not np.isfinite(kernel_sum):
        raise ValueError(
            f"Interpolated IPC kernel for SCA {sca_id} at field position "
            f"({fx}, {fy}) has a non-finite sum ({kernel_sum}).  "
            "Check the calibration HDF5 file for NaN/Inf values."
        )
    if kernel_sum <= 0.0:
        raise ValueError(
            f"Interpolated IPC kernel for SCA {sca_id} at field position "
            f"({fx}, {fy}) has a non-positive sum ({kernel_sum}).  "
            "The kernel cannot be normalised; check the calibration HDF5 file."
        )

    # Renormalize to exactly 1.0.
    kernel /= kernel_sum
    return kernel


def _find_bracket(coords: np.ndarray, val: float) -> tuple[int, float]:
    """Find the lower grid index and fractional offset for interpolation.

    Parameters
    ----------
    coords:
        Monotonically increasing 1D coordinate array.
    val:
        Query value (must be within the range of ``coords``).

    Returns
    -------
    tuple[int, float]
        ``(index, fraction)`` where ``index`` is the lower bounding index
        and ``fraction`` is in [0, 1].
    """
    if len(coords) < 2:
        return 0, 0.0
    idx = np.searchsorted(coords, val, side="right") - 1
    idx = int(np.clip(idx, 0, len(coords) - 2))
    span = coords[idx + 1] - coords[idx]
    if span == 0.0:
        return idx, 0.0
    frac = (val - coords[idx]) / span
    frac = float(np.clip(frac, 0.0, 1.0))
    return idx, frac
