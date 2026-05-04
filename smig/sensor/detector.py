"""
smig/sensor/detector.py
========================
Star-topology orchestrator for the Roman WFI H4RG-10 detector simulation.

H4RG10Detector is the sole orchestrator in the signal chain.  Leaf modules
(ipc, charge_diffusion, etc.) are owned by the detector and must not import
each other.

Fixed signal chain order (CLAUDE.md §Architectural Constraints):
    charge diffusion → IPC → persistence → MULTIACCUM readout (with per-read NL)
    → noise injection (1/f, RTS, cosmic rays)
"""

from __future__ import annotations

import gc
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from smig.config.schemas import ChargeDiffusionConfig, DetectorConfig
from smig.config.utils import get_config_sha256
from smig.provenance.schema import ProvenanceRecord, sanitize_rng_state
from smig.sensor.charge_diffusion import ChargeDiffusionModel
from smig.sensor.ipc import FieldDependentIPC
from smig.sensor.noise.correlated import OneOverFNoise, RTSNoise
from smig.sensor.noise.cosmic_rays import ClusteredCosmicRayInjector
from smig.sensor.nonlinearity import NonLinearityModel
from smig.sensor.persistence import DynamicPersistence
from smig.sensor.readout import MultiAccumSimulator

# MJD epoch: 1858-11-17 00:00:00 UTC
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorOutput:
    """Per-epoch detector simulation result.

    Note: ``frozen=True`` freezes attribute *bindings* (``self.rate_image``
    cannot be rebound to a different array), but the ``np.ndarray`` contents
    remain mutable.  This is a deliberate trade-off: deep immutability of large
    arrays is prohibitively expensive.

    Attributes
    ----------
    rate_image:
        2D float64 array (ny, nx) of electron counts after the full signal chain.
    saturation_mask:
        2D bool array (ny, nx); True where charge met or exceeded the saturation
        flag threshold.
    cr_mask:
        2D bool array (ny, nx); True at ALL pixels in the cosmic-ray cluster
        (primary impact site plus all pixels with deposited charge from the
        same event, including morphologically extended tracks).  A single
        event may set multiple pixels True.
    provenance_data:
        Dict containing exactly the 12 ``ProvenanceRecord`` constructor kwargs
        that are not supplied by ``process_event``:
        ``git_commit``, ``container_digest``, ``python_version``,
        ``numpy_version``, ``config_sha256``, ``random_state``,
        ``ipc_applied``, ``persistence_applied``, ``nonlinearity_applied``,
        ``charge_diffusion_applied``,
        ``saturated_pixel_count``, ``cosmic_ray_hit_count``.
    """

    rate_image: np.ndarray
    saturation_mask: np.ndarray
    cr_mask: np.ndarray
    provenance_data: dict[str, Any]


@dataclass(frozen=True)
class EventOutput:
    """Multi-epoch event simulation result.

    Note: ``frozen=True`` freezes attribute bindings; ``np.ndarray`` contents
    remain mutable.  See ``DetectorOutput`` docstring.

    Attributes
    ----------
    rate_cube:
        3D float64 array (n_epochs, ny, nx).
    saturation_cube:
        3D bool array (n_epochs, ny, nx).
    cr_cube:
        3D bool array (n_epochs, ny, nx).
    persistence_peak_map:
        2D float64 array (ny, nx); maximum persistence signal seen across all
        epochs (stub: all zeros).
    provenance_records:
        One ``ProvenanceRecord`` per epoch, in epoch-index order.
    peak_memory_mb:
        Peak resident memory consumed during ``process_event`` in megabytes
        (stub: 0.0).
    """

    rate_cube: np.ndarray
    saturation_cube: np.ndarray
    cr_cube: np.ndarray
    persistence_peak_map: np.ndarray
    provenance_records: list[ProvenanceRecord]
    peak_memory_mb: float


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class H4RG10Detector:
    """Star-topology orchestrator for the Roman WFI H4RG-10 detector.

    Owns all leaf-module instances and drives the fixed signal chain:
    charge diffusion → IPC → persistence → MULTIACCUM readout (with per-read NL)
    → noise injection.

    Parameters
    ----------
    config:
        Fully validated, immutable detector configuration.
    rng:
        NumPy random generator consumed by noise leaf modules for reproducible
        stochastic realizations.

    Notes
    -----
    ``config_sha256`` is computed once at construction time.  Because
    ``DetectorConfig`` is frozen, the hash cannot change for the lifetime of
    this object.
    """

    def __init__(self, config: DetectorConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._config_sha256 = get_config_sha256(config)
        self._git_commit: str | None = os.environ.get("GIT_COMMIT_SHA")
        self._container_digest: str | None = os.environ.get("IMAGE_DIGEST")

        # Leaf module instances (star topology: detector owns all, none import peers).
        # Exception: readout.py imports nonlinearity.py (one-way, documented in readout.py).
        #
        # ChargeDiffusionConfig is built here explicitly so that ChargeDiffusionModel
        # receives only the fields it needs, enforcing the leaf interface boundary.
        _cd_config = ChargeDiffusionConfig(
            pixel_pitch_um=config.geometry.pixel_pitch_um,
            full_well_electrons=config.electrical.full_well_electrons,
            diffusion_length_factor=config.charge_diffusion.diffusion_length_factor,
            bfe_coupling_coeff=config.charge_diffusion.bfe_coupling_coeff,
        )
        self._charge_diffusion = ChargeDiffusionModel(_cd_config)
        # sca_id is passed from the IPC config; the HDF5 kernel loader uses it
        # as its dataset key when ipc_kernel_path is set.
        # field_position: use config.ipc.field_position if that field is declared
        # in IPCConfig in the future; otherwise FieldDependentIPC defaults to (0.5, 0.5).
        self._ipc = FieldDependentIPC(
            config.ipc,
            sca_id=config.ipc.sca_id,
            field_position=getattr(config.ipc, "field_position", None),
        )
        self._persistence = DynamicPersistence(config.persistence)
        # NonLinearityModel must be created before MultiAccumSimulator: it is
        # injected into the readout simulator so NL can be applied per-read
        # when the 3D MULTIACCUM ramp is implemented.
        self._nonlinearity = NonLinearityModel(
            config.nonlinearity,
            full_well_electrons=config.electrical.full_well_electrons,
        )

        # Spawn independent, deterministic child RNGs for all stochastic modules.
        # Must happen before any child RNG is consumed.
        # Indices: [0] 1/f noise, [1] RTS noise, [2] cosmic rays, [3] readout.
        # SeedSequence.spawn is positionally stable: spawn(4)[0:3] == spawn(3)[0:3],
        # so existing noise-module reproducibility is preserved when adding index 3.
        _seed_seq = rng.bit_generator.seed_seq
        if _seed_seq is None:
            # Fallback for bit generators not backed by a SeedSequence (e.g.
            # custom BitGenerators or legacy wrappers). Sample entropy directly
            # from the parent generator to build a deterministic SeedSequence.
            _seed_seq = np.random.SeedSequence(
                int(rng.integers(0, 2**63, dtype=np.int64))
            )
        _child_seeds = _seed_seq.spawn(4)

        self._readout = MultiAccumSimulator(
            config.readout,
            dark_current_e_per_s=config.electrical.dark_current_e_per_s,
            read_noise_cds_electrons=config.electrical.read_noise_cds_electrons,
            nonlinearity=self._nonlinearity,
            rng=np.random.default_rng(_child_seeds[3]),
        )
        self._onef_noise = OneOverFNoise(config, np.random.default_rng(_child_seeds[0]))
        self._rts_noise = RTSNoise(config, np.random.default_rng(_child_seeds[1]))
        self._cr_injector = ClusteredCosmicRayInjector(
            config, np.random.default_rng(_child_seeds[2])
        )

        # Precompute once at construction time; config is frozen so this cannot drift.
        self._saturation_threshold: float = (
            config.nonlinearity.saturation_flag_threshold
            * config.electrical.full_well_electrons
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_epoch(
        self,
        ideal_image_e: np.ndarray,
        epoch_index: int,
        epoch_time_mjd: float,
        prev_epoch_time_mjd: float | None = None,
    ) -> DetectorOutput:
        """Process a single epoch through the detector signal chain.

        The current implementation is a structural stub: leaf module calls are
        wired in the correct order, but each is a no-op returning a copy.
        Physics will be enabled in subsequent phases.

        Note on cosmic-ray injection: ``cr_injector.apply()`` is a 2D-only
        placeholder.  When ``simulate_ramp`` is upgraded to return a 3D ramp
        ``(n_reads, ny, nx)``, CR injection must be replaced with per-read
        calls to ``inject_into_ramp``.  See FIXME CR-1 below.

        Parameters
        ----------
        ideal_image_e:
            2D array (ny, nx) of ideal electron counts (no detector effects
            applied).  Must match ``config.geometry.ny`` × ``config.geometry.nx``.
        epoch_index:
            Zero-based index of this epoch within the event.
        epoch_time_mjd:
            Modified Julian Date of this epoch's observation midpoint.  Used to
            derive a deterministic, physically consistent UTC timestamp for
            provenance (via MJD epoch 1858-11-17), and to compute the
            inter-epoch time delta for persistence.
        prev_epoch_time_mjd:
            MJD of the immediately preceding epoch, or ``None`` for the first
            epoch.  The inter-epoch time delta is derived exclusively from
            ``epoch_time_mjd - prev_epoch_time_mjd``; there is no separate
            ``delta_time_s`` parameter to prevent duplicate formulas.

        Returns
        -------
        DetectorOutput

        Raises
        ------
        ValueError
            If ``ideal_image_e`` is not 2-D, does not match the configured
            geometry, contains non-finite values, contains negative electrons,
            or if ``epoch_index`` is negative.
        """
        if ideal_image_e.ndim != 2:
            raise ValueError(
                f"ideal_image_e must be 2-D, got ndim={ideal_image_e.ndim}."
            )
        ny = self._config.geometry.ny
        nx = self._config.geometry.nx
        if ideal_image_e.shape != (ny, nx):
            raise ValueError(
                f"ideal_image_e.shape {ideal_image_e.shape!r} does not match "
                f"configured geometry ({ny}, {nx})."
            )
        if not np.all(np.isfinite(ideal_image_e)):
            raise ValueError(
                "ideal_image_e contains non-finite values (NaN or Inf)."
            )
        if np.any(ideal_image_e < 0):
            raise ValueError(
                "ideal_image_e contains negative electron counts."
            )
        if epoch_index < 0:
            raise ValueError(
                f"epoch_index must be >= 0, got {epoch_index}."
            )
        if not np.isfinite(epoch_time_mjd):
            raise ValueError(
                f"epoch_time_mjd must be finite, got {epoch_time_mjd}."
            )
        if prev_epoch_time_mjd is not None:
            if not np.isfinite(prev_epoch_time_mjd):
                raise ValueError(
                    f"prev_epoch_time_mjd must be finite, got {prev_epoch_time_mjd}."
                )
            if prev_epoch_time_mjd > epoch_time_mjd:
                raise ValueError(
                    f"prev_epoch_time_mjd ({prev_epoch_time_mjd}) must be <= "
                    f"epoch_time_mjd ({epoch_time_mjd}); zero-delta is allowed."
                )

        # Capture child-RNG states BEFORE any epoch draws so the epoch can be
        # perfectly replayed from this initial state.
        # BREAKING CHANGE: random_state is now a structured dict with one entry
        # per child generator, not a flat parent-generator state dict.
        rng_state: dict[str, Any] = {
            "parent": sanitize_rng_state(self._rng.bit_generator.state),
            "readout": sanitize_rng_state(self._readout._rng.bit_generator.state),
            "one_over_f": sanitize_rng_state(self._onef_noise._rng.bit_generator.state),
            "rts": sanitize_rng_state(self._rts_noise._rng.bit_generator.state),
            "cosmic_rays": sanitize_rng_state(self._cr_injector._rng.bit_generator.state),
        }

        # Inter-epoch time delta — single source of truth, derived from MJD only.
        # No separate delta_time_s parameter exists on this method.
        if prev_epoch_time_mjd is not None:
            delta_time_s = (epoch_time_mjd - prev_epoch_time_mjd) * 86400.0
        else:
            delta_time_s = 0.0

        # Cast to float64 and begin the signal chain.
        image = ideal_image_e.astype(np.float64, copy=True)

        # --- Fixed signal chain (stubs: each leaf returns input.copy()) ---
        # Charge diffusion before IPC: diffusion acts on the physical charge
        # distribution in the semiconductor layer; IPC acts on the electrical
        # sensing of that charge. Swapping these is physically incorrect.
        image = self._charge_diffusion.apply(image)
        image = self._ipc.apply(image)
        image = self._persistence.apply(image, delta_time_s=delta_time_s)
        # Build 3D MULTIACCUM ramp (n_reads, ny, nx) with NL applied per-read.
        # sat_reads: pre-noise physical saturation flag (accurate; not contaminated
        # by read noise that could push clipped values below Q_sat).
        ramp_cube, sat_reads = self._readout.simulate_ramp(image)

        # Saturation mask derived from pre-noise accumulated charge.
        saturation_mask = sat_reads.any(axis=0)
        saturated_pixel_count = int(saturation_mask.sum())

        # Collapse 3D ramp to 2D rate image (e-/s) via OLS slope fit.
        # Pass sat_reads for accurate per-pixel saturation exclusion.
        image = self._readout.fit_slope(ramp_cube, sat_reads=sat_reads)
        del ramp_cube, sat_reads
        gc.collect()

        # Noise injection (2D rate image; stubs pass through unchanged).
        # FIXME CR-1: When CR injection is upgraded to per-read, replace this
        # call with inject_into_ramp inside simulate_ramp.
        image = self._onef_noise.apply(image)
        image = self._rts_noise.apply(image)
        image, cr_mask, cr_hit_count = self._cr_injector.apply(image)

        provenance_data: dict[str, Any] = {
            "git_commit": self._git_commit,
            "container_digest": self._container_digest,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "config_sha256": self._config_sha256,
            "random_state": sanitize_rng_state(rng_state),
            "ipc_applied": True,
            # DynamicPersistence.apply() is called unconditionally in the signal
            # chain, so this flag is always True.  If a config toggle is added
            # later (e.g. config.persistence.enabled), derive this flag from it.
            "persistence_applied": True,
            "nonlinearity_applied": True,   # NL applied per-read inside simulate_ramp
            "charge_diffusion_applied": True,
            "saturated_pixel_count": saturated_pixel_count,
            "cosmic_ray_hit_count": cr_hit_count,
        }

        return DetectorOutput(
            rate_image=image,
            saturation_mask=saturation_mask,
            cr_mask=cr_mask,
            provenance_data=provenance_data,
        )

    def process_event(
        self,
        event_id: str,
        ideal_cube_e: np.ndarray,
        timestamps_mjd: np.ndarray,
    ) -> EventOutput:
        """Process a multi-epoch event through the detector signal chain.

        Iterates over the epoch dimension of ``ideal_cube_e``, calling
        ``process_epoch`` for each slice.  Explicitly frees per-epoch
        intermediate data after extracting results to stay within the 32 GB
        memory budget.

        Parameters
        ----------
        event_id:
            Unique event identifier recorded in every provenance record.
        ideal_cube_e:
            3D array (n_epochs, ny, nx) of ideal electron counts.
        timestamps_mjd:
            1D array (n_epochs,) of Modified Julian Dates, one per epoch.

        Returns
        -------
        EventOutput

        Raises
        ------
        ValueError
            If ``ideal_cube_e`` is not 3-D, if ``timestamps_mjd`` length does
            not match the epoch dimension, or if ``timestamps_mjd`` contains
            non-finite values.
        """
        if ideal_cube_e.ndim != 3:
            raise ValueError(
                f"ideal_cube_e must be 3-D, got ndim={ideal_cube_e.ndim}."
            )
        n_epochs = ideal_cube_e.shape[0]
        if timestamps_mjd.shape != (n_epochs,):
            raise ValueError(
                f"timestamps_mjd.shape {timestamps_mjd.shape!r} does not match "
                f"n_epochs {n_epochs}."
            )
        if not np.all(np.isfinite(timestamps_mjd)):
            raise ValueError(
                "timestamps_mjd contains non-finite values (NaN or Inf)."
            )
        if len(timestamps_mjd) > 1 and not np.all(np.diff(timestamps_mjd) >= 0):
            raise ValueError(
                "timestamps_mjd must be non-decreasing."
            )

        # Validate spatial geometry against config BEFORE the epoch loop.
        cfg_ny = self._config.geometry.ny
        cfg_nx = self._config.geometry.nx
        if ideal_cube_e.shape[1:] != (cfg_ny, cfg_nx):
            raise ValueError(
                f"ideal_cube_e spatial shape {ideal_cube_e.shape[1:]!r} does not "
                f"match configured geometry ({cfg_ny}, {cfg_nx})."
            )

        ny, nx = ideal_cube_e.shape[1], ideal_cube_e.shape[2]
        rate_cube = np.empty((n_epochs, ny, nx), dtype=np.float64)
        saturation_cube = np.zeros((n_epochs, ny, nx), dtype=bool)
        cr_cube = np.zeros((n_epochs, ny, nx), dtype=bool)
        persistence_peak_map = np.zeros((ny, nx), dtype=np.float64)
        provenance_records: list[ProvenanceRecord] = []

        for i in range(n_epochs):
            prev_mjd = float(timestamps_mjd[i - 1]) if i > 0 else None
            epoch_output = self.process_epoch(
                ideal_image_e=ideal_cube_e[i],
                epoch_index=i,
                epoch_time_mjd=float(timestamps_mjd[i]),
                prev_epoch_time_mjd=prev_mjd,
            )

            rate_cube[i] = epoch_output.rate_image
            saturation_cube[i] = epoch_output.saturation_mask
            cr_cube[i] = epoch_output.cr_mask

            record = ProvenanceRecord(
                event_id=event_id,
                epoch_index=i,
                timestamp_utc=_MJD_EPOCH + timedelta(days=float(timestamps_mjd[i])),
                **epoch_output.provenance_data,
            )
            provenance_records.append(record)

            # Aggressive GC required to prevent 3D ramp cubes from exceeding
            # 32GB limit during long events.
            del epoch_output
            gc.collect()

        return EventOutput(
            rate_cube=rate_cube,
            saturation_cube=saturation_cube,
            cr_cube=cr_cube,
            persistence_peak_map=persistence_peak_map,
            provenance_records=provenance_records,
            peak_memory_mb=0.0,
        )
