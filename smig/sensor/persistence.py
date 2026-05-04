"""
smig/sensor/persistence.py
===========================
Two-component exponential persistence (residual image) model stub.

Leaf module — must not import any sibling sensor module.
"""

from __future__ import annotations

import numpy as np

from smig.config.schemas import PersistenceConfig


class DynamicPersistence:
    """Two-component exponential persistence (residual image) model.

    Tracks trapped charge from previous epochs and injects a persistence signal
    into subsequent frames.  The decay follows::

        P(t) = amp_short * exp(-t / tau_short_s)
               + amp_long  * exp(-t / tau_long_s)

    This stub is a no-op pass-through.

    Parameters
    ----------
    config:
        Persistence sub-configuration extracted from DetectorConfig.
    """

    def __init__(self, config: PersistenceConfig) -> None:
        self._config = config
        # Per-pixel trap-state map lazily allocated on first apply() call.
        # Stores trapped charge (electrons) from previous epochs.
        self._trap_state: np.ndarray | None = None

    def apply(self, image: np.ndarray, delta_time_s: float = 0.0) -> np.ndarray:
        """Apply persistence injection to a charge image.

        Parameters
        ----------
        image:
            2D array (ny, nx) of electron counts.
        delta_time_s:
            Time elapsed since the previous epoch in seconds, derived from
            the MJD timestamps by the orchestrator.  Used to compute the
            exponential decay of trapped charge from prior saturation events.

        Returns
        -------
        np.ndarray
            Image with persistence signal added (stub: copy of input).

        # TODO: Implement physical model — two-component exponential decay from
        # self._trap_state updated after each epoch.
        """
        # Lazy allocation: dimensions only known at first call.
        if self._trap_state is None:
            self._trap_state = np.zeros(image.shape, dtype=np.float64)
        elif self._trap_state.shape != image.shape:
            raise ValueError(
                f"DynamicPersistence: input image shape changed unexpectedly.  "
                f"Cached trap-state shape is {self._trap_state.shape!r} but "
                f"received {image.shape!r}.  Each DynamicPersistence instance "
                "must be used with a single fixed detector geometry."
            )
        # TODO: Apply exponential decay using delta_time_s and update
        # self._trap_state based on current image charge levels.
        return image.copy()
