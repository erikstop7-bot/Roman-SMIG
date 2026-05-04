"""
smig/sensor/charge_diffusion.py
================================
Charge diffusion and brighter-fatter effect (BFE).

Leaf module — must not import any sibling sensor module.

Static diffusion is a fixed Gaussian kernel applied via
``scipy.ndimage.gaussian_filter``.  The brighter-fatter effect is an
iterative, signal-dependent charge redistribution to the 4 nearest
neighbours (Jacobi-style, 3 iterations).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from smig.config.schemas import ChargeDiffusionConfig


class ChargeDiffusionModel:
    """Charge diffusion and brighter-fatter effect (BFE) model.

    Models lateral charge migration in the HgCdTe detector layer and the
    signal-dependent PSF broadening (brighter-fatter effect).

    Static diffusion is applied via a fixed Gaussian kernel computed from
    ``config.pixel_pitch_um * config.diffusion_length_factor``.  BFE is
    applied as a signal-dependent perturbation kernel using only the
    *current* accumulated charge in ``image``.

    Parameters
    ----------
    config:
        Charge diffusion sub-configuration built by the orchestrator from
        ``DetectorConfig.geometry.pixel_pitch_um``,
        ``DetectorConfig.electrical.full_well_electrons``, and
        ``DetectorConfig.charge_diffusion`` tuning parameters.
    """

    def __init__(self, config: ChargeDiffusionConfig) -> None:
        self._config = config
        # diffusion_length_factor is the dimensionless ratio of diffusion
        # length to pixel pitch.  gaussian_filter expects sigma in pixel
        # (array-index) units, so:
        #   sigma_pixels = (pixel_pitch_um * factor) / pixel_pitch_um = factor
        self._sigma = config.diffusion_length_factor

    def apply_static_diffusion(self, image: np.ndarray) -> np.ndarray:
        """Apply a static Gaussian diffusion kernel.

        Uses ``scipy.ndimage.gaussian_filter`` with ``mode='reflect'``.
        Post-hoc renormalization enforces strict flux conservation despite
        edge truncation.

        Parameters
        ----------
        image:
            2D array (ny, nx) of electron counts.

        Returns
        -------
        np.ndarray
            Diffused image with conserved total flux.
        """
        original_sum = image.sum()
        result = gaussian_filter(image, sigma=self._sigma, mode="reflect")
        # Post-hoc renormalization for strict flux conservation.
        # Epsilon guard prevents divide-by-zero when the image is all zeros.
        if original_sum > 0.0:
            result *= original_sum / (result.sum() + 1e-12)
        return result

    def apply_bfe(self, image: np.ndarray) -> np.ndarray:
        """Apply the brighter-fatter effect via iterative Jacobi redistribution.

        For each pixel, a charge fraction proportional to the local
        fill-fraction is redistributed to the 4 nearest neighbours:

            f = Q / Q_FW
            delta_Q = bfe_coupling_coeff * f * Q

        The centre pixel loses 4 * delta_Q and each orthogonal neighbour
        gains delta_Q.  Boundaries use reflection (``np.pad`` mode
        ``'reflect'``).  Three Jacobi iterations are performed (updates
        computed from a snapshot of the previous iteration's state).

        Parameters
        ----------
        image:
            2D array (ny, nx) of electron counts (post-static-diffusion).

        Returns
        -------
        np.ndarray
            BFE-adjusted image.  Charge is clipped to >= 0.
        """
        coupling = self._config.bfe_coupling_coeff
        full_well = self._config.full_well_electrons

        if coupling == 0.0:
            return image.copy()

        current = image.copy()
        ny, nx = current.shape

        for _ in range(3):
            # Snapshot for Jacobi-style update (all updates from previous state).
            prev = current.copy()

            # Charge fraction: cap at 1.0 to prevent over-saturation artefacts
            # when charge slightly exceeds full_well due to BFE redistribution.
            f = np.clip(prev / full_well, 0.0, 1.0)
            delta_q = coupling * f * prev  # shape (ny, nx)

            # Pad with reflection for boundary handling.
            padded_delta = np.pad(delta_q, pad_width=1, mode="reflect")

            # Accumulate neighbour contributions from the padded array.
            # padded_delta indices are offset by 1 relative to current.
            neighbour_sum = (
                padded_delta[0:ny, 1 : nx + 1]      # top
                + padded_delta[2 : ny + 2, 1 : nx + 1]  # bottom
                + padded_delta[1 : ny + 1, 0:nx]     # left
                + padded_delta[1 : ny + 1, 2 : nx + 2]  # right
            )

            # Centre pixel loses charge to 4 neighbours, gains from 4 neighbours.
            current = prev - 4.0 * delta_q + neighbour_sum

        # Clip to prevent unphysical negative electrons.
        np.clip(current, 0.0, None, out=current)

        # Post-hoc renormalization: reflection padding at boundaries can
        # introduce a tiny flux imbalance; correct it to conserve charge.
        # Epsilon guard prevents divide-by-zero when the image is all zeros.
        original_sum = image.sum()
        if original_sum > 0.0:
            current *= original_sum / (current.sum() + 1e-12)
        return current

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply charge diffusion and BFE to a charge image.

        Applies static Gaussian diffusion followed by the brighter-fatter
        effect, in accordance with the fixed signal chain order.

        Parameters
        ----------
        image:
            2D array (ny, nx) of electron counts.

        Returns
        -------
        np.ndarray
            Diffused + BFE-adjusted image.

        Raises
        ------
        ValueError
            If ``image`` is not 2-D.
        """
        if image.ndim != 2:
            raise ValueError(
                f"ChargeDiffusionModel.apply() requires a 2-D image, "
                f"got ndim={image.ndim}."
            )
        result = self.apply_static_diffusion(image)
        result = self.apply_bfe(result)
        return result
