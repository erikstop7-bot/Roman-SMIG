"""
smig/sensor/noise/cosmic_rays.py
=================================
Clustered cosmic-ray hit injector stub.

Leaf module — must not import any sibling sensor module.
"""

from __future__ import annotations

import numpy as np

from smig.config.schemas import DetectorConfig


class ClusteredCosmicRayInjector:
    """Clustered cosmic-ray hit injector.

    Simulates cosmic-ray strikes as spatially clustered charge deposits in the
    detector, with energies and morphologies drawn from empirical distributions.

    This stub is a no-op.

    Parameters
    ----------
    config:
        Full detector configuration (geometry and electrical parameters required).
    rng:
        NumPy random generator for reproducible noise realizations.
    """

    def __init__(self, config: DetectorConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Inject cosmic-ray hits into a 2D image.

        Parameters
        ----------
        image:
            2D array (ny, nx) of electron counts.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            ``(modified_image, cr_mask, cosmic_ray_hit_count)``.
            ``cr_mask`` is a boolean array of the same shape as ``image``,
            True at ALL pixels in the cosmic-ray cluster (primary impact site
            plus all pixels with deposited charge from the same event,
            including morphologically extended tracks).  A single event may
            set multiple pixels True.
            ``cosmic_ray_hit_count`` is the count of distinct CR *events*
            (primary impact sites), not affected pixels.  A single event with
            a 5-pixel morphology contributes 1 to this count.
            Stub: returns ``(copy of input, all-False mask, 0)``.

        Raises
        ------
        ValueError
            If ``image.ndim != 2``.  This method operates on 2D images only;
            use ``inject_into_ramp`` for 3D MULTIACCUM ramps (not yet
            implemented).

        # TODO: Implement physical model — sample hit positions, energies, and
        # cluster morphologies from empirical distributions, then deposit charge.
        """
        if image.ndim != 2:
            raise ValueError(
                f"apply() requires a 2-D image array, got ndim={image.ndim}. "
                "For 3-D MULTIACCUM ramps use inject_into_ramp() (not yet implemented)."
            )
        return image.copy(), np.zeros(image.shape, dtype=bool), 0

    def inject_into_ramp(
        self,
        ramp: np.ndarray,
        read_idx: int,
        *,
        epoch_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Inject cosmic-ray hits into a single read of a 3D MULTIACCUM ramp.

        Intended to replace ``apply()`` once ``simulate_ramp`` returns a 3D
        array ``(n_reads, ny, nx)``.  CR injection must be applied per-read
        so that the deposited charge accumulates correctly in subsequent reads.

        Parameters
        ----------
        ramp:
            3D array (n_reads, ny, nx) representing the full ramp buffer.
        read_idx:
            Index of the read within the ramp at which the CR event occurs.
        epoch_index:
            Zero-based epoch index; used to seed per-epoch reproducibility.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            ``(modified_ramp, cr_mask, cosmic_ray_hit_count)`` where
            ``cr_mask`` is a 2D boolean array (ny, nx).

        Raises
        ------
        NotImplementedError
            Always.  Implement once simulate_ramp returns a 3D ramp.
            See detector.py FIXME CR-1.
        """
        raise NotImplementedError(
            "inject_into_ramp is not yet implemented. "
            "Implement alongside the 3D MULTIACCUM ramp in readout.py. "
            "See detector.py FIXME CR-1."
        )

    def _inject_single_event(
        self,
        image: np.ndarray,
        y0: int,
        x0: int,
        energy_electrons: float,
        morphology: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deposit a single cosmic-ray event at a specified location.

        Deterministic injection hook used for unit testing and for composing
        the stochastic ``apply()`` / ``inject_into_ramp()`` implementations.

        Parameters
        ----------
        image:
            2D array (ny, nx) of electron counts to modify in-place copy.
        y0:
            Row index of the primary impact site.
        x0:
            Column index of the primary impact site.
        energy_electrons:
            Total energy deposited by the CR event, in electrons.
        morphology:
            2D array of fractional energy weights summing to 1.0; defines
            the spatial extent of the cluster relative to ``(y0, x0)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(modified_image, cr_mask)`` where ``cr_mask`` is True at ALL
            pixels that received deposited charge from this event.

        Raises
        ------
        NotImplementedError
            Always.  Implement alongside the physical injection model.
        """
        raise NotImplementedError(
            "_inject_single_event is not yet implemented. "
            "Implement as the deterministic core of apply() / inject_into_ramp()."
        )
