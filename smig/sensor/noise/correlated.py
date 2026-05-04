"""
smig/sensor/noise/correlated.py
================================
Correlated noise stubs: 1/f (pink) noise and random telegraph signal (RTS).

Leaf module — must not import any sibling sensor module.
OneOverFNoise and RTSNoise are independent classes; neither imports the other.
"""

from __future__ import annotations

import numpy as np

from smig.config.schemas import DetectorConfig


class OneOverFNoise:
    """1/f (pink) correlated noise generator.

    Generates spatially and temporally correlated noise with a 1/f power
    spectral density, characteristic of MOSFET amplifiers in the H4RG-10
    readout circuit.

    This stub is a no-op pass-through.

    Parameters
    ----------
    config:
        Full detector configuration (electrical and readout parameters required).
    rng:
        NumPy random generator for reproducible noise realizations.
    """

    def __init__(self, config: DetectorConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Inject 1/f correlated noise into an image.

        Parameters
        ----------
        image:
            2D array (ny, nx) to add noise to.

        Returns
        -------
        np.ndarray
            Image with 1/f noise added (stub: copy of input).

        # TODO: Implement physical model — generate 1/f noise via FFT-based
        # method (e.g., Timmer & König 1995) and add to image.
        """
        return image.copy()


class RTSNoise:
    """Random telegraph signal (RTS) noise generator.

    Simulates discrete switching noise from individual charge traps in H4RG-10
    detector pixels.

    This stub is a no-op pass-through.

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
        # Per-pixel two-state Markov chain states, lazily allocated on first
        # apply() call.  Persists across epochs to model trap-switching history.
        self._pixel_states: np.ndarray | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Inject RTS noise into an image.

        Parameters
        ----------
        image:
            2D array (ny, nx) to add noise to.

        Returns
        -------
        np.ndarray
            Image with RTS noise added (stub: copy of input).

        # TODO: Implement physical model — sample a two-state Markov chain
        # per pixel using self._rng, update self._pixel_states, and add the
        # switching amplitude to the image.
        """
        # Lazy allocation: dimensions only known at first call.
        if self._pixel_states is None:
            self._pixel_states = np.zeros(image.shape, dtype=np.int8)
        elif self._pixel_states.shape != image.shape:
            raise ValueError(
                f"RTSNoise: input image shape changed unexpectedly.  "
                f"Cached pixel-state shape is {self._pixel_states.shape!r} but "
                f"received {image.shape!r}.  Each RTSNoise instance "
                "must be used with a single fixed detector geometry."
            )
        return image.copy()
