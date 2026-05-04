"""
smig/sensor/readout.py
=======================
MULTIACCUM (up-the-ramp) readout simulator.

Leaf module — must not import any sibling sensor module, with one deliberate
exception: NonLinearityModel is injected by the orchestrator (H4RG10Detector)
so that nonlinearity can be applied per-read within the ramp.

ARCHITECTURAL NOTE — one-way dependency:
    readout.py → nonlinearity.py  (this file, intentional)
    nonlinearity.py → readout.py  (FORBIDDEN — must remain zero)

NL is applied per-read inside the ramp loop; this is the only reason for
the dependency.  The coupling is bounded: only NonLinearityModel.apply() is
called from within this module.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from smig.config.schemas import ReadoutConfig
from smig.sensor.nonlinearity import NonLinearityModel


class MultiAccumSimulator:
    """MULTIACCUM (up-the-ramp) readout simulator.

    Builds a non-destructive read ramp from an accumulated charge image,
    simulating the H4RG-10's sample-up-the-ramp readout mode.  Returns a
    3D ramp of shape ``(n_reads, ny, nx)`` with nonlinearity applied
    per-read and Gaussian read noise added after each nonlinearity
    evaluation.

    Parameters
    ----------
    config:
        Readout sub-configuration extracted from DetectorConfig.
    dark_current_e_per_s:
        Dark current rate in electrons per second from ``ElectricalConfig``.
    read_noise_cds_electrons:
        Correlated double-sampling read noise in electrons from
        ``ElectricalConfig``.
    nonlinearity:
        Nonlinearity model injected by the orchestrator.  Applied per-read
        inside the ramp loop.
    rng:
        NumPy random generator for all stochastic sampling.  Must be an
        explicitly injected, deterministic generator; ``None`` is not allowed.
        The orchestrator derives this from a child of the event-level seed.
    """

    def __init__(
        self,
        config: ReadoutConfig,
        dark_current_e_per_s: float,
        read_noise_cds_electrons: float,
        nonlinearity: NonLinearityModel | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            raise ValueError(
                "MultiAccumSimulator requires an explicitly injected RNG "
                "(rng must not be None).  Pass a deterministic generator "
                "derived from the event-level seed via SeedSequence.spawn()."
            )
        self._config = config
        self._dark_current_e_per_s = dark_current_e_per_s
        self._read_noise_cds_electrons = read_noise_cds_electrons
        self._nonlinearity = nonlinearity
        self._rng = rng

    # ------------------------------------------------------------------
    # Ramp construction
    # ------------------------------------------------------------------

    def simulate_ramp(
        self,
        ideal_image_e: np.ndarray,
        cr_injector: Callable[..., tuple[np.ndarray, np.ndarray, int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a 3D up-the-ramp sample cube.

        The photon signal (``ideal_image_e``) is treated as the total
        accumulated charge over the full exposure and distributed uniformly
        across ``n_reads - 1`` frame intervals.  Each interval adds
        Poisson-sampled photon charge and Poisson-sampled dark current.
        Nonlinearity is applied to the accumulated charge state at each
        read, followed by Gaussian read noise.

        Saturation is detected on the **physical accumulated charge before
        read noise** (a pixel is physically saturated when it has collected
        ≥ Q_sat electrons, independent of the noisy measured value).

        Parameters
        ----------
        ideal_image_e:
            2D array ``(ny, nx)`` of total ideal electron counts for the
            full exposure (photon signal only, no dark current).
        cr_injector:
            Optional callable for per-read cosmic-ray injection.  When
            provided, it is called after each dark/photon accumulation step
            and before saturation flagging, with signature::

                cr_injector(accumulated: np.ndarray, read_idx: int)
                    -> (modified_accumulated, cr_mask_2d, hit_count)

            The injector deposits charge directly into ``accumulated`` (the
            pre-NL, pre-noise physical charge buffer) so that subsequent
            saturation checks and read-noise addition see the correct charge.

            IMPORTANT: ``readout.py`` must NOT import ``cosmic_rays``
            directly (star-topology constraint).  Pass the injector from
            ``H4RG10Detector`` as a Callable, never import the class here.

            Stub behaviour: when ``cr_injector`` is ``None`` (current
            default), per-read CR injection is skipped and the 2D ``apply()``
            call in the orchestrator's post-ramp path remains in effect
            (see FIXME CR-1 in detector.py).

        Returns
        -------
        ramp : np.ndarray
            3D float64 array ``(n_reads, ny, nx)`` of measured signal
            values (NL-corrected + read noise, electrons).
        sat_reads : np.ndarray
            3D bool array ``(n_reads, ny, nx)``; ``True`` where the
            accumulated physical charge met or exceeded the saturation
            flagging threshold at that read.  Use ``.any(axis=0)`` to
            obtain a 2D per-pixel saturation flag.

        Raises
        ------
        ValueError
            If ``ideal_image_e`` is not 2-D or does not have 2 spatial axes.
        """
        if ideal_image_e.ndim != 2:
            raise ValueError(
                f"simulate_ramp requires a 2-D input array, "
                f"got ndim={ideal_image_e.ndim}."
            )
        ny, nx = ideal_image_e.shape
        n_reads = self._config.n_ramp_reads
        frame_time = self._config.frame_time_s

        # Per-frame-interval photon increment: total photon signal / (n-1) frames.
        # n_ramp_reads >= 2 is enforced by schema, so (n_reads - 1) >= 1.
        # Floor to zero: FFT-based IPC/charge-diffusion convolutions can produce
        # tiny sub-zero ringing artifacts that cause poisson() to raise.
        photon_per_interval = np.maximum(ideal_image_e / (n_reads - 1), 0.0)

        # Scalar dark current increment per frame interval (electrons).
        dark_per_interval = self._dark_current_e_per_s * frame_time

        # Per-read noise std: CDS noise / sqrt(2) gives per-sample std.
        per_read_noise_std = self._read_noise_cds_electrons / np.sqrt(2.0)

        # Saturation flagging level from the public NL property (electrons).
        if self._nonlinearity is not None:
            Q_sat = self._nonlinearity.saturation_flagging_threshold_e
        else:
            Q_sat = np.inf

        ramp = np.empty((n_reads, ny, nx), dtype=np.float64)
        sat_reads = np.zeros((n_reads, ny, nx), dtype=bool)
        # accumulated tracks physical charge (pre-NL, pre-noise).
        accumulated = np.zeros((ny, nx), dtype=np.float64)

        for i in range(n_reads):
            if i > 0:
                # Poisson-sample photon signal increment (per pixel).
                accumulated += self._rng.poisson(photon_per_interval)
                # Poisson-sample dark current increment (scalar rate, per-pixel draw).
                accumulated += self._rng.poisson(dark_per_interval, size=(ny, nx))

                # Per-read CR injection hook (star-topology: injector passed
                # as Callable from orchestrator, never imported here).
                if cr_injector is not None:
                    accumulated, _cr_mask, _cr_count = cr_injector(accumulated, i)

            # Saturation flag on physical charge BEFORE noise (physically correct).
            sat_reads[i] = accumulated >= Q_sat

            # Measured value: apply NL to physical charge, then add read noise.
            if self._nonlinearity is not None:
                measured = self._nonlinearity.apply(accumulated)
            else:
                measured = accumulated.copy()

            measured += self._rng.normal(0.0, per_read_noise_std, size=(ny, nx))
            ramp[i] = measured

        return ramp, sat_reads

    # ------------------------------------------------------------------
    # Ramp collapse
    # ------------------------------------------------------------------

    def fit_slope(
        self,
        ramp_cube: np.ndarray,
        sat_reads: np.ndarray | None = None,
    ) -> np.ndarray:
        """Estimate the count rate for each pixel via OLS along the ramp.

        Uses ordinary least-squares linear regression over the ``n_reads``
        axis.  Reads at or above the saturation threshold and all subsequent
        reads are excluded from the fit on a per-pixel basis.

        Parameters
        ----------
        ramp_cube:
            3D array ``(n_reads, ny, nx)`` of measured signal values as
            returned by ``simulate_ramp``.
        sat_reads:
            Optional 3D bool array ``(n_reads, ny, nx)`` of pre-noise
            saturation flags as returned by ``simulate_ramp``.  When
            provided, saturation exclusion is based on physical charge
            (more accurate than thresholding the noisy ramp).  When
            ``None``, falls back to ``ramp_cube >= Q_sat``.

        Returns
        -------
        np.ndarray
            2D array ``(ny, nx)`` of fitted count rates in electrons per
            second.  Pixels where fewer than 2 good reads remain (e.g.
            saturated from the first science read) return 0.0.

        Raises
        ------
        ValueError
            If ``ramp_cube`` is not 3-D.
        """
        if ramp_cube.ndim != 3:
            raise ValueError(
                f"fit_slope requires a 3-D ramp cube, got ndim={ramp_cube.ndim}."
            )
        n_reads, ny, nx = ramp_cube.shape
        frame_time = self._config.frame_time_s
        t = np.arange(n_reads, dtype=np.float64) * frame_time  # (n_reads,)

        # Saturation flagging threshold for fallback (when sat_reads not provided).
        if self._nonlinearity is not None:
            Q_sat = self._nonlinearity.saturation_flagging_threshold_e
        else:
            Q_sat = np.inf

        # n_good[y, x]: number of reads to include for pixel (y, x).
        # Reads 0 .. n_good-1 are valid; reads n_good .. n_reads-1 are excluded.
        if sat_reads is not None:
            sat_mask = sat_reads                  # pre-noise physical saturation
        else:
            sat_mask = ramp_cube >= Q_sat         # fallback: threshold noisy ramp
        any_sat = sat_mask.any(axis=0)            # (ny, nx)
        first_sat = np.argmax(sat_mask, axis=0)   # (ny, nx); 0 where never sat
        # n_good: index of first bad read (all reads valid → n_reads)
        n_good = np.where(any_sat, first_sat, n_reads).astype(np.int64)  # (ny, nx)

        # Accumulate OLS sums over reads (loop over n_reads; avoids large
        # (n_reads, ny, nx) temporaries beyond ramp_cube itself).
        sum_t = np.zeros((ny, nx), dtype=np.float64)
        sum_y = np.zeros((ny, nx), dtype=np.float64)
        sum_tt = np.zeros((ny, nx), dtype=np.float64)
        sum_ty = np.zeros((ny, nx), dtype=np.float64)
        n_used = np.zeros((ny, nx), dtype=np.float64)

        for i in range(n_reads):
            valid_i = i < n_good        # (ny, nx) bool
            ti = t[i]
            y_i = ramp_cube[i]          # (ny, nx) view — no copy

            sum_t += np.where(valid_i, ti, 0.0)
            sum_y += np.where(valid_i, y_i, 0.0)
            sum_tt += np.where(valid_i, ti * ti, 0.0)
            sum_ty += np.where(valid_i, ti * y_i, 0.0)
            n_used += valid_i.astype(np.float64)

        # OLS slope = (n*Σty - Σt*Σy) / (n*Σtt - (Σt)²)
        # np.divide with where= avoids the spurious "invalid value encountered
        # in divide" RuntimeWarning that np.where would emit because both
        # branches are evaluated before selection.
        numerator = n_used * sum_ty - sum_t * sum_y
        denom = n_used * sum_tt - sum_t ** 2
        slope = np.zeros_like(numerator)
        np.divide(numerator, denom, out=slope, where=denom > 0.0)
        return slope
