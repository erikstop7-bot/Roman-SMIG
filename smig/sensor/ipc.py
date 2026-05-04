"""
smig/sensor/ipc.py
==================
Field-dependent inter-pixel capacitance (IPC) convolution.

Leaf module — must not import any sibling sensor module.
Topology rule: ``ipc`` may import ``calibration``, but ``calibration``
must NEVER import ``ipc``.

## Kernel shape contract

The unified ``build_kernel()`` method returns a single 9x9 ``np.ndarray``
normalised to sum = 1.0.  Two paths:

1. **Analytic (uniform)**: ``config.ipc_kernel_path is None`` — build a
   symmetric kernel from ``ipc_alpha_center``.
2. **HDF5 (field-dependent)**: bilinear interpolation via the calibration
   loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from smig.config.schemas import IPCConfig
from smig.sensor.calibration.ipc_kernels import load_interpolated_kernel


class FieldDependentIPC:
    """Field-dependent inter-pixel capacitance (IPC) convolution.

    Applies a spatially-varying IPC kernel to convert collected charge to
    electrically-sensed signal.  Loads a 9x9 kernel from HDF5 when a
    calibration path is provided, otherwise constructs a uniform analytic
    kernel from ``ipc_alpha_center``.

    Parameters
    ----------
    config:
        IPC sub-configuration extracted from DetectorConfig.
    sca_id:
        Science camera array identifier (1-18).  Used as the HDF5 dataset
        lookup key when ``config.ipc_kernel_path`` is provided.
    field_position:
        ``(x, y)`` fractional position within the SCA focal plane
        (0.0-1.0 in each axis), used to select the local kernel from the
        field-dependent kernel map.
    """

    def __init__(
        self,
        config: IPCConfig,
        sca_id: int,
        field_position: tuple[float, float] | None = None,
    ) -> None:
        self._config = config
        self._sca_id = sca_id
        # Default to SCA centre when no field position is supplied.
        self._field_position: tuple[float, float] = (
            field_position if field_position is not None else (0.5, 0.5)
        )
        self._kernel = self.build_kernel()

    def build_kernel(self) -> np.ndarray:
        """Build or load a normalised 9x9 IPC kernel.

        If ``config.ipc_kernel_path`` is ``None``, constructs a uniform
        analytic kernel with ``ipc_alpha_center`` coupling to the 4
        nearest neighbours and weaker diagonal coupling.  Otherwise,
        loads and bilinearly interpolates from the HDF5 calibration file.

        Returns
        -------
        np.ndarray
            9x9 float64 kernel normalised to sum = 1.0.
        """
        if (
            self._config.ipc_field_dependent
            and self._config.ipc_kernel_path is not None
        ):
            kernel = load_interpolated_kernel(
                path=Path(self._config.ipc_kernel_path),
                sca_id=self._sca_id,
                field_position=self._field_position,
            )
            self._validate_kernel_shape(kernel)
            return kernel

        # Analytic uniform kernel construction (used when ipc_field_dependent
        # is False OR when no HDF5 calibration path is provided).
        ks = self._config.ipc_kernel_size
        kernel = np.zeros((ks, ks), dtype=np.float64)
        c = ks // 2
        alpha = self._config.ipc_alpha_center

        # Horizontal/vertical coupling (symmetric).
        kernel[c - 1, c] = alpha
        kernel[c + 1, c] = alpha
        kernel[c, c - 1] = alpha
        kernel[c, c + 1] = alpha

        # Diagonal coupling: configurable fraction of the orthogonal coupling.
        # The schema validator guarantees alpha <= 1/(4 + 4*diag_frac), so the
        # centre pixel (1 - sum of neighbours) is always >= 0.
        diag = alpha * self._config.ipc_diagonal_fraction
        kernel[c - 1, c - 1] = diag
        kernel[c - 1, c + 1] = diag
        kernel[c + 1, c - 1] = diag
        kernel[c + 1, c + 1] = diag

        # Centre pixel: remainder ensures sum = 1.
        kernel[c, c] = 1.0 - kernel.sum()

        return kernel

    def _validate_kernel_shape(self, kernel: np.ndarray) -> None:
        """Validate that a loaded IPC kernel has the expected shape.

        Parameters
        ----------
        kernel:
            Kernel array returned by the HDF5 loader.

        Raises
        ------
        ValueError
            If the kernel is not 2D or does not match the configured
            kernel size.
        """
        ks = self._config.ipc_kernel_size
        if kernel.ndim != 2:
            raise ValueError(
                f"IPC kernel must be 2-D, got ndim={kernel.ndim}."
            )
        if kernel.shape != (ks, ks):
            raise ValueError(
                f"IPC kernel shape {kernel.shape!r} does not match "
                f"configured size ({ks}, {ks})."
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply IPC convolution to a charge image.

        Uses ``scipy.signal.fftconvolve`` with ``mode='same'`` for
        efficient convolution with the 9x9 kernel.  The image is
        reflect-padded before convolution and cropped afterwards to
        avoid flux leakage at the boundaries.

        Parameters
        ----------
        image:
            2D array (ny, nx) of electron counts.

        Returns
        -------
        np.ndarray
            IPC-convolved image.

        Raises
        ------
        ValueError
            If ``image`` is not 2-D.
        """
        if image.ndim != 2:
            raise ValueError(
                f"FieldDependentIPC.apply() requires a 2-D image, "
                f"got ndim={image.ndim}."
            )
        pad = self._kernel.shape[0] // 2
        padded = np.pad(image, pad, mode="reflect")
        result = fftconvolve(padded, self._kernel, mode="same")
        return result[pad:-pad, pad:-pad]

    def deconvolve(self, image: np.ndarray, n_iterations: int = 4) -> np.ndarray:
        """Iterative Jansson-Van Cittert IPC deconvolution.

        This method is for testing and analysis only and is not part of
        the forward simulation hot path.

        Iteratively estimates the pre-IPC image by correcting the
        residual between the re-convolved estimate and the observed image.

        Parameters
        ----------
        image:
            2D array (ny, nx) of IPC-convolved electron counts.
        n_iterations:
            Number of Van Cittert iterations (default: 4).

        Returns
        -------
        np.ndarray
            Estimated pre-IPC image.

        Raises
        ------
        ValueError
            If ``image`` is not 2-D.
        """
        if image.ndim != 2:
            raise ValueError(
                f"FieldDependentIPC.deconvolve() requires a 2-D image, "
                f"got ndim={image.ndim}."
            )
        pad = self._kernel.shape[0] // 2
        estimate = image.copy()
        for _ in range(n_iterations):
            padded = np.pad(estimate, pad, mode="reflect")
            reconvolved = fftconvolve(padded, self._kernel, mode="same")
            reconvolved = reconvolved[pad:-pad, pad:-pad]
            estimate += image - reconvolved
        return estimate
