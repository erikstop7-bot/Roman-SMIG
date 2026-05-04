"""HDF5 I/O for the SMIG Phase 3 LOM ablation sprint.

Reading (phase3-contract-v1)
----------------------------
``iter_shard`` opens a Phase 3 HDF5 shard produced by the frozen pipeline and
yields one ``Phase3EventRecord`` per event.  The contract datasets are read in
the order defined by ``smig.datasets.schema.LABEL_DATASET_NAMES`` (imported
read-only; this file never modifies smig/).

Denormalization
---------------
``denormalize`` converts stored label values back to physical parameters using
the exact inverse of the transforms in smig.datasets.worker:

    tE  = 10 ** log_tE
    u0  = 10 ** log_u0          # base is 10, matching log10 in worker
    rho = 10 ** log_rho
    q   = 10 ** log_q           (or None when log_q == SENTINEL = −99.0)
    s   = 10 ** log_s           (or None when log_s == SENTINEL = −99.0)
    t0  = obs_start + t0_norm * (n_epochs − 1) * cadence_days

The t0 denominator ``(n_epochs − 1) * cadence_days`` is exactly the
``window_span_days`` computed in smig.datasets.worker line ~243:
``window_span_days = (n_science_epochs - 1) * cadence_days``.  The u0 base
is 10 (log10), not e (natural log), matching
``math.log10(abs(event.u0))`` in smig.datasets.worker._assemble_label_vector.
Both must stay bit-identical with the worker's forward transform to avoid
systematic denormalization bias.

Sentinel Handling
-----------------
The sentinel value −99.0 (``smig.datasets.worker.UNDEFINED_BINARY_PARAM_SENTINEL``)
marks undefined binary parameters in single-lens events.  Detection uses
``np.isclose(val, −99.0, atol=1e-3)`` rather than a fixed tolerance of 0.5
to survive float32 round-trips without false positives.  (The nearest valid
log_q for a real binary event is approximately −4, which is ≫ 1 away from
the sentinel.)

Sentinel Pairing Invariant
--------------------------
log_q and log_s are **always set together** by the Phase 3 worker: both are
set to the sentinel for single-lens events, or both are set to physical values
for binary events.  ``denormalize`` enforces this invariant; a row where one
is a sentinel and the other is not indicates shard corruption and raises
``SentinelPairingError``.

Epoch Times
-----------
All events in a Phase 3 shard share the same uniform cadence grid:

    epoch_times_mjd[i] = obs_start_mjd + i * cadence_days,  i = 0 … n_epochs−1

This grid is **identical across all events within a shard** — t0 is the event
peak within the grid, not an epoch boundary.

Survey Config YAML
------------------
The Phase 3 YAML (e.g. configs/phase3_smoke.yaml) supplies:

    observation_start_mjd  float   MJD of the first science epoch
    cadence_days           float   Uniform cadence between epochs (days)
    n_science_epochs       int     Number of epochs per event
    sky_background_e_per_s float   Sky background (electrons per second)
    simulation_config_path str     Path to simulation.yaml (resolved relative to
                                   the YAML's own directory first, then CWD)

The ``simulation_config_path`` sub-config supplies:

    detector.readout.exposure_time_s  float  Exposure duration (seconds)

Writing (phase3-lom-ablation-v0)
---------------------------------
``LomShardWriter`` is a context manager that writes a parallel HDF5 shard with:

- All 12 standard label datasets from phase3-contract-v1 (same dtypes)
- Two new LOM datasets:
    lom__ds_dt_per_yr        float32  (N,)   ds/dt in θ_E / yr per event
    lom__dalpha_dt_rad_per_yr float32  (N,)   dα/dt in rad / yr per event
- Row-ordering datasets:
    shard_row_index          uint32   (N,)   source shard row index (passthrough)
    lom_row_index            uint32   (N,)   sequential position in this output shard
- Provenance root attributes:
    schema_version           "phase3-lom-ablation-v0"
    source_schema_version    "phase3-contract-v1"
    event_id_transform       "prefix:lom_"
    rows_expected / rows_written / rows_failed  (written on close)
- Atomic write discipline: .tmp file + os.replace (mirrors smig.datasets.writer)

The source Phase 3 shards are opened in read-only mode and never modified.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import yaml

# Read-only imports from smig — no smig file is modified.
from smig.datasets.schema import (
    DATASET_SCHEMA_VERSION as _PHASE3_SCHEMA_VERSION,
    HDF5_COMPRESSION,
    LABEL_DATASET_NAMES,
    SCIENCE_STAMP_SHAPE,
    science_stamp_chunks,
)
from smig.config.utils import load_simulation_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOM_SCHEMA_VERSION: str = "phase3-lom-ablation-v0"

# Sentinel written by smig.datasets.worker.UNDEFINED_BINARY_PARAM_SENTINEL.
_SENTINEL: float = -99.0

# float32 round-trip can shift values by ≲ 1e-4; atol=1e-3 is tight enough
# to be unambiguous (nearest valid log_q for a real binary is ≈ −4, far away)
# while robust against floating-point representation noise.
_SENTINEL_ATOL: float = 1e-3

_EVENT_ID_DTYPE = h5py.string_dtype(encoding="utf-8")

_SMIG_VERSION: str
try:
    import importlib.metadata as _im
    _SMIG_VERSION = _im.version("smig")
except Exception:
    _SMIG_VERSION = "unknown"

# LOM-specific label datasets added by this ablation schema.
LOM_LABEL_DATASET_NAMES: tuple[str, ...] = (
    "lom__ds_dt_per_yr",
    "lom__dalpha_dt_rad_per_yr",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SentinelPairingError(ValueError):
    """Raised when log_q and log_s sentinels are mismatched in a shard row.

    The Phase 3 worker always sets both to the sentinel (single-lens) or
    neither (binary).  A mismatch indicates shard corruption.
    """

    def __init__(self, event_id: str, log_q: float, log_s: float) -> None:
        self.event_id = event_id
        self.log_q = log_q
        self.log_s = log_s
        super().__init__(
            f"Sentinel pairing violated for event {event_id!r}: "
            f"log_q={log_q} sentinel={_is_sentinel(log_q)}, "
            f"log_s={log_s} sentinel={_is_sentinel(log_s)}. "
            "Both must be sentinel (single-lens) or neither (binary)."
        )


# ---------------------------------------------------------------------------
# Survey config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SurveyConfig:
    """Observation-level parameters from the Phase 3 generation YAML.

    These are NOT stored in the HDF5 shards and must be supplied explicitly.

    Attributes
    ----------
    observation_start_mjd :
        MJD of the first science epoch (index 0).
    cadence_days :
        Uniform inter-epoch cadence in days.
    n_science_epochs :
        Number of science epochs per event.  Must match the ``n_epochs`` HDF5
        file attribute; ``iter_shard`` validates this at read time.
    sky_background_e_per_s :
        Sky background in electrons per second (constant over the survey window).
    exposure_s :
        Exposure duration in seconds (from simulation_config readout block).
    """

    observation_start_mjd: float
    cadence_days: float
    n_science_epochs: int
    sky_background_e_per_s: float
    exposure_s: float

    @property
    def epoch_times_mjd(self) -> np.ndarray:
        """Uniform cadence grid shared by all events in the survey window."""
        return (
            self.observation_start_mjd
            + np.arange(self.n_science_epochs, dtype=float) * self.cadence_days
        )

    @property
    def window_span_days(self) -> float:
        """Survey window duration used to normalise t0 (days).

        Mirrors smig.datasets.worker: ``window_span_days = (n_science_epochs - 1) * cadence_days``.
        """
        return (self.n_science_epochs - 1) * self.cadence_days

    def t0_abs_mjd(self, t0_normalized: float) -> float:
        """Inverse of the t0_mjd_normalized transform applied at generation time.

        Exact inverse of smig.datasets.worker lines ~244–246::

            t0_abs_mjd = observation_start_mjd + t0_rng.uniform(0, window_span_days)
            t0_norm    = (t0_abs_mjd - observation_start_mjd) / window_span_days

        Parameters
        ----------
        t0_normalized :
            ``label__t0_mjd_normalized`` value from the HDF5 shard (∈ [0, 1)).

        Returns
        -------
        float
            Absolute MJD of the event peak time.
        """
        return self.observation_start_mjd + float(t0_normalized) * self.window_span_days


def load_survey_config(config_path: str | Path) -> SurveyConfig:
    """Load survey observation parameters from a Phase 3 generation YAML.

    Parameters
    ----------
    config_path :
        Path to the Phase 3 YAML config (e.g. ``configs/phase3_smoke.yaml``).
        Required keys: ``observation_start_mjd``, ``cadence_days``,
        ``n_science_epochs``, ``sky_background_e_per_s``,
        ``simulation_config_path``.

    Returns
    -------
    SurveyConfig

    Raises
    ------
    FileNotFoundError
        If the YAML file or the referenced simulation config does not exist.
    KeyError
        If a required key is missing from the YAML.
    """
    config_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    obs_start = float(raw["observation_start_mjd"])
    cadence = float(raw["cadence_days"])
    n_epochs = int(raw["n_science_epochs"])
    sky_bg = float(raw["sky_background_e_per_s"])

    sim_cfg_str = raw["simulation_config_path"]
    sim_cfg_path = Path(sim_cfg_str)
    if not sim_cfg_path.is_absolute():
        # Resolve relative to the config file's parent first, then CWD.
        candidate = config_path.parent / sim_cfg_path
        sim_cfg_path = candidate if candidate.exists() else Path.cwd() / sim_cfg_path

    sim_cfg = load_simulation_config(sim_cfg_path)
    exposure_s = float(sim_cfg.detector.readout.exposure_time_s)

    return SurveyConfig(
        observation_start_mjd=obs_start,
        cadence_days=cadence,
        n_science_epochs=n_epochs,
        sky_background_e_per_s=sky_bg,
        exposure_s=exposure_s,
    )


# ---------------------------------------------------------------------------
# Event record types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Phase3EventRecord:
    """Raw per-event data read from a phase3-contract-v1 HDF5 shard.

    Labels are in their HDF5 stored form (log-space, sentinel-encoded).
    Use ``denormalize`` to convert to physical parameters.
    """

    event_id: str
    starfield_seed: int
    shard_row_index: int            # source shard position (for provenance links)
    # 12 label fields (stored, log-space)
    event_class: int                # uint8: PSPL=0 FSPL_STAR=1 PC=2 SB=3 HMC=4
    log_tE: float
    log_u0: float
    log_rho: float
    alpha_rad: float
    log_q: float                    # −99.0 sentinel for single-lens events
    log_s: float                    # −99.0 sentinel for single-lens events
    t0_mjd_normalized: float        # ∈ [0, 1)
    source_mag_F146_ab: float
    lens_mass_msun: float
    source_distance_kpc: float
    lens_distance_kpc: float
    # Pre-rendered science stamps (carried through for passthrough / reference)
    stamps: np.ndarray              # (n_epochs, 64, 64) float32


@dataclasses.dataclass(frozen=True)
class DenormalizedParams:
    """Physical parameters reconstructed from a Phase3EventRecord.

    Attributes
    ----------
    t0_mjd :
        Absolute peak time (MJD), reconstructed via ``SurveyConfig.t0_abs_mjd``.
    tE_days :
        Einstein ring crossing time (days).
    u0 :
        Impact parameter (θ_E units).
    rho :
        Source radius (θ_E units).
    alpha_rad :
        Binary axis angle at t0 (radians; passed through directly from label).
    q :
        Binary mass ratio m2/m1, or ``None`` for single-lens events.
    s :
        Binary separation (θ_E units), or ``None`` for single-lens events.
    source_mag_F146_ab :
        Unlensed source AB magnitude in F146 (direct passthrough from label).
    """

    t0_mjd: float
    tE_days: float
    u0: float
    rho: float
    alpha_rad: float
    q: float | None   # None for PSPL / FSPL_STAR
    s: float | None   # None for PSPL / FSPL_STAR
    source_mag_F146_ab: float

    @property
    def is_binary(self) -> bool:
        """True for PLANETARY_CAUSTIC, STELLAR_BINARY, HIGH_MAGNIFICATION_CUSP."""
        return self.q is not None


def _is_sentinel(val: float) -> bool:
    """True if val is the −99.0 undefined-binary-param sentinel.

    Uses np.isclose with atol=1e-3 to survive float32 round-trips.
    The nearest valid log_q for a real binary event is ≈ −4 (mass ratio 1e-4),
    which is ≫ 1 away from −99, so atol=1e-3 has no false-positive risk.
    """
    return bool(np.isclose(float(val), _SENTINEL, atol=_SENTINEL_ATOL, rtol=0.0))


def denormalize(record: Phase3EventRecord, survey: SurveyConfig) -> DenormalizedParams:
    """Convert stored label values to physical parameters.

    Applies the exact inverse of the transforms in smig.datasets.worker:

        tE  = 10 ** log_tE
        u0  = 10 ** log_u0          (base 10, matching math.log10 in worker)
        rho = 10 ** log_rho
        q   = 10 ** log_q           (None when log_q ≈ −99 sentinel)
        s   = 10 ** log_s           (None when log_s ≈ −99 sentinel)
        t0  = obs_start + t0_norm * (n_epochs − 1) * cadence_days

    Sentinel Pairing Invariant
    --------------------------
    If log_q is a sentinel, log_s must also be a sentinel (and vice versa).
    Violation raises SentinelPairingError — it indicates a corrupted shard row
    that must not be silently propagated into the LOM corpus.

    Parameters
    ----------
    record :
        Raw HDF5 record from ``iter_shard``.
    survey :
        Survey config providing obs_start_mjd, cadence_days, n_science_epochs.

    Returns
    -------
    DenormalizedParams

    Raises
    ------
    SentinelPairingError
        If exactly one of log_q / log_s is a sentinel.
    """
    q_is_sentinel = _is_sentinel(record.log_q)
    s_is_sentinel = _is_sentinel(record.log_s)

    if q_is_sentinel != s_is_sentinel:
        raise SentinelPairingError(record.event_id, record.log_q, record.log_s)

    q: float | None = None
    s: float | None = None
    if not q_is_sentinel:
        q = 10.0 ** float(record.log_q)
        s = 10.0 ** float(record.log_s)

    return DenormalizedParams(
        t0_mjd=survey.t0_abs_mjd(record.t0_mjd_normalized),
        tE_days=10.0 ** float(record.log_tE),
        u0=10.0 ** float(record.log_u0),
        rho=10.0 ** float(record.log_rho),
        alpha_rad=float(record.alpha_rad),
        q=q,
        s=s,
        source_mag_F146_ab=float(record.source_mag_F146_ab),
    )


# ---------------------------------------------------------------------------
# Shard reader
# ---------------------------------------------------------------------------

def iter_shard(
    shard_path: str | Path,
    survey: SurveyConfig,
) -> Iterator[tuple[Phase3EventRecord, DenormalizedParams]]:
    """Yield (raw record, denormalized params) for every event in a Phase 3 shard.

    Parameters
    ----------
    shard_path :
        Path to a phase3-contract-v1 HDF5 shard.
    survey :
        Survey config for denormalization and n_epochs validation.

    Yields
    ------
    (Phase3EventRecord, DenormalizedParams)

    Raises
    ------
    FileNotFoundError
        If shard_path does not exist.
    ValueError
        If schema_version ≠ "phase3-contract-v1" or n_epochs ≠ survey.n_science_epochs.
    SentinelPairingError
        If any event row has mismatched sentinels (propagated from ``denormalize``).
    """
    shard_path = Path(shard_path)
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    with h5py.File(str(shard_path), "r") as f:
        schema = str(f.attrs.get("schema_version", "<missing>"))
        if schema != _PHASE3_SCHEMA_VERSION:
            raise ValueError(
                f"Expected schema_version={_PHASE3_SCHEMA_VERSION!r}, "
                f"got {schema!r} in {shard_path}"
            )
        shard_n_epochs = int(f.attrs["n_epochs"])
        if shard_n_epochs != survey.n_science_epochs:
            raise ValueError(
                f"Shard n_epochs={shard_n_epochs} does not match "
                f"survey.n_science_epochs={survey.n_science_epochs} in {shard_path}"
            )

        n_events = f["science_stamps"].shape[0]

        # Load all 1-D label arrays at once; each is shape (N,), small in memory.
        labels: dict[str, np.ndarray] = {ds: f[ds][:] for ds in LABEL_DATASET_NAMES}
        event_ids: list[str] = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in f["event_id"][:]
        ]
        starfield_seeds: np.ndarray = f["starfield_seed"][:]
        shard_row_indices: np.ndarray = f["shard_row_index"][:]

        for i in range(n_events):
            # Load one stamp row at a time to honour the (1, n_epochs, 64, 64) chunk layout.
            stamps: np.ndarray = f["science_stamps"][i]

            record = Phase3EventRecord(
                event_id=event_ids[i],
                starfield_seed=int(starfield_seeds[i]),
                shard_row_index=int(shard_row_indices[i]),
                event_class=int(labels["label__event_class"][i]),
                log_tE=float(labels["label__log_tE"][i]),
                log_u0=float(labels["label__log_u0"][i]),
                log_rho=float(labels["label__log_rho"][i]),
                alpha_rad=float(labels["label__alpha_rad"][i]),
                log_q=float(labels["label__log_q"][i]),
                log_s=float(labels["label__log_s"][i]),
                t0_mjd_normalized=float(labels["label__t0_mjd_normalized"][i]),
                source_mag_F146_ab=float(labels["label__source_mag_F146_ab"][i]),
                lens_mass_msun=float(labels["label__lens_mass_msun"][i]),
                source_distance_kpc=float(labels["label__source_distance_kpc"][i]),
                lens_distance_kpc=float(labels["label__lens_distance_kpc"][i]),
                stamps=stamps,
            )
            yield record, denormalize(record, survey)


def read_all_shards(
    shard_dir: str | Path,
    survey: SurveyConfig,
    glob_pattern: str = "*.h5",
) -> Iterator[tuple[Phase3EventRecord, DenormalizedParams]]:
    """Iterate over all Phase 3 shards in a directory, sorted by filename.

    Parameters
    ----------
    shard_dir :
        Directory containing phase3-contract-v1 HDF5 files.
    survey :
        Survey config for denormalization.
    glob_pattern :
        Glob pattern applied inside shard_dir (default ``"*.h5"``).

    Yields
    ------
    (Phase3EventRecord, DenormalizedParams)

    Raises
    ------
    FileNotFoundError
        If no files match the pattern.
    """
    shard_dir = Path(shard_dir)
    shard_paths = sorted(shard_dir.glob(glob_pattern))
    if not shard_paths:
        raise FileNotFoundError(
            f"No HDF5 shards matching {glob_pattern!r} found in {shard_dir}"
        )
    for shard_path in shard_paths:
        yield from iter_shard(shard_path, survey)


# ---------------------------------------------------------------------------
# LOM shard writer
# ---------------------------------------------------------------------------

class LomShardWriterError(ValueError):
    """Raised on dtype / shape / finite validation failure in LomShardWriter."""


class LomShardWriter:
    """Context manager for writing one LOM ablation HDF5 shard atomically.

    Layout extends phase3-contract-v1 with two LOM datasets and a
    ``lom_row_index`` ordering tracker::

        /lom__ds_dt_per_yr          float32  (N,)   per-event ds/dt (θ_E / yr)
        /lom__dalpha_dt_rad_per_yr  float32  (N,)   per-event dα/dt (rad / yr)
        /lom_row_index              uint32   (N,)   sequential output row index
        /shard_row_index            uint32   (N,)   source shard row (passthrough)
        /source_event_id            str      (N,)   original event_id (before prefix)

    Root attribute ``event_id_transform = "prefix:lom_"`` records the naming
    convention so downstream tools can reconstruct the source event_id without
    string parsing heuristics.

    Partial-Failure Metadata
    ------------------------
    On close, three provenance attributes are written:

        rows_expected   (int)  pre-allocated count passed to the constructor
        rows_written    (int)  actual committed rows
        rows_failed     (int)  events skipped by the runner (set via resize)

    These attributes are present even when the shard is full (rows_failed == 0).

    Write Discipline
    ----------------
    Opens a ``.tmp`` file in the same directory as the target; on successful
    close atomically renames to the final path.  On exception the ``.tmp`` is
    deleted.  Mirrors smig.datasets.writer.ShardWriter exactly.

    Parameters
    ----------
    path :
        Final target path (e.g. ``output/lom_shard_0000.h5``).
    shard_id :
        Integer shard identifier stored in root attrs.
    expected_n :
        Pre-allocated row count.
    n_epochs :
        Number of science epochs per event.
    ds_dt :
        Shard-level ds/dt default (θ_E / yr).  Also stored as a root attr for
        quick introspection without loading the per-event dataset.
    dalpha_dt :
        Shard-level dα/dt default (rad / yr).
    smig_version :
        Override for the smig version string in root attrs.
    """

    def __init__(
        self,
        path: Path,
        shard_id: int,
        expected_n: int,
        n_epochs: int,
        ds_dt: float,
        dalpha_dt: float,
        smig_version: str = _SMIG_VERSION,
    ) -> None:
        self._target_path = Path(path)
        self._tmp_path = self._target_path.with_suffix(
            self._target_path.suffix + ".tmp"
        )
        self._shard_id = shard_id
        self._expected_n = expected_n
        self._n_epochs = n_epochs
        self._ds_dt = ds_dt
        self._dalpha_dt = dalpha_dt
        self._smig_version = smig_version
        self._f: h5py.File | None = None
        self._rows_written: int = 0
        self._rows_failed: int = 0

    def __enter__(self) -> "LomShardWriter":
        self._f = h5py.File(str(self._tmp_path), "w")
        f = self._f

        # Root attributes — written now; rows_written / rows_failed added on close.
        f.attrs["schema_version"] = LOM_SCHEMA_VERSION
        f.attrs["source_schema_version"] = _PHASE3_SCHEMA_VERSION
        f.attrs["shard_id"] = self._shard_id
        f.attrs["n_epochs"] = self._n_epochs
        f.attrs["smig_version"] = self._smig_version
        f.attrs["writer_backend"] = "h5py"
        f.attrs["lom_ds_dt_per_yr"] = float(self._ds_dt)
        f.attrs["lom_dalpha_dt_rad_per_yr"] = float(self._dalpha_dt)
        # Programmatic audit trail: downstream tools can reconstruct source_event_id
        # from event_id without string heuristics.
        f.attrs["event_id_transform"] = "prefix:lom_"
        f.attrs["rows_expected"] = self._expected_n

        n = self._expected_n
        epochs = self._n_epochs
        comp, comp_opts = HDF5_COMPRESSION

        # Science stamps (re-rendered with LOM magnification)
        f.create_dataset(
            "science_stamps",
            shape=(n, epochs, *SCIENCE_STAMP_SHAPE),
            maxshape=(None, epochs, *SCIENCE_STAMP_SHAPE),
            dtype=np.float32,
            chunks=science_stamp_chunks(epochs),
            compression=comp,
            compression_opts=comp_opts,
        )

        # Standard Phase 3 label datasets (same dtypes as phase3-contract-v1)
        for ds_name in LABEL_DATASET_NAMES:
            dtype = np.uint8 if ds_name == "label__event_class" else np.float32
            f.create_dataset(
                ds_name,
                shape=(n,),
                maxshape=(None,),
                dtype=dtype,
                compression=comp,
                compression_opts=comp_opts,
            )

        # LOM-specific per-event datasets
        for ds_name in LOM_LABEL_DATASET_NAMES:
            f.create_dataset(
                ds_name,
                shape=(n,),
                maxshape=(None,),
                dtype=np.float32,
                compression=comp,
                compression_opts=comp_opts,
            )

        # Metadata datasets — no compression (small, frequently accessed)
        f.create_dataset("event_id", shape=(n,), maxshape=(None,), dtype=_EVENT_ID_DTYPE)
        f.create_dataset("source_event_id", shape=(n,), maxshape=(None,), dtype=_EVENT_ID_DTYPE)
        f.create_dataset("starfield_seed", shape=(n,), maxshape=(None,), dtype=np.uint64)
        # shard_row_index: source shard position (passthrough from Phase 3 record)
        f.create_dataset("shard_row_index", shape=(n,), maxshape=(None,), dtype=np.uint32)
        # lom_row_index: sequential position within this output shard
        f.create_dataset("lom_row_index", shape=(n,), maxshape=(None,), dtype=np.uint32)

        return self

    def write_row(
        self,
        lom_row_index: int,
        record: "Phase3EventRecord",
        new_stamps: np.ndarray,
        ds_dt: float,
        dalpha_dt: float,
    ) -> None:
        """Write one LOM ablation event row.

        Parameters
        ----------
        lom_row_index :
            Sequential zero-based output row index in this shard.  Written to
            the ``lom_row_index`` dataset (NOT to ``shard_row_index``, which
            carries the source shard position from ``record.shard_row_index``).
        record :
            Source Phase3EventRecord providing labels, event_id, starfield_seed,
            and the original ``shard_row_index`` to be passed through.
        new_stamps :
            Re-rendered science stamps with LOM magnification applied (binary
            events), or original stamps passed through (single-lens nulls).
            Must be float32, shape (n_epochs, 64, 64), all finite.
        ds_dt :
            Per-event ds/dt (θ_E / yr).  0.0 for single-lens passthrough rows.
        dalpha_dt :
            Per-event dα/dt (rad / yr).  0.0 for single-lens passthrough rows.

        Raises
        ------
        LomShardWriterError
            If new_stamps fails dtype, shape, or finite validation.
        """
        if new_stamps.dtype != np.float32:
            raise LomShardWriterError(
                f"new_stamps.dtype must be float32, got {new_stamps.dtype} "
                f"(event_id={record.event_id!r})"
            )
        expected_shape = (self._n_epochs, 64, 64)
        if new_stamps.shape != expected_shape:
            raise LomShardWriterError(
                f"new_stamps.shape must be {expected_shape}, "
                f"got {new_stamps.shape} (event_id={record.event_id!r})"
            )
        if not np.isfinite(new_stamps).all():
            raise LomShardWriterError(
                f"new_stamps contain NaN or Inf (event_id={record.event_id!r})"
            )

        assert self._f is not None
        f = self._f

        f["science_stamps"][lom_row_index] = new_stamps

        # Standard labels — direct passthrough from source record.
        f["label__event_class"][lom_row_index] = np.uint8(record.event_class)
        f["label__log_tE"][lom_row_index] = np.float32(record.log_tE)
        f["label__log_u0"][lom_row_index] = np.float32(record.log_u0)
        f["label__log_rho"][lom_row_index] = np.float32(record.log_rho)
        f["label__alpha_rad"][lom_row_index] = np.float32(record.alpha_rad)
        f["label__log_q"][lom_row_index] = np.float32(record.log_q)
        f["label__log_s"][lom_row_index] = np.float32(record.log_s)
        f["label__t0_mjd_normalized"][lom_row_index] = np.float32(record.t0_mjd_normalized)
        f["label__source_mag_F146_ab"][lom_row_index] = np.float32(record.source_mag_F146_ab)
        f["label__lens_mass_msun"][lom_row_index] = np.float32(record.lens_mass_msun)
        f["label__source_distance_kpc"][lom_row_index] = np.float32(record.source_distance_kpc)
        f["label__lens_distance_kpc"][lom_row_index] = np.float32(record.lens_distance_kpc)

        # LOM per-event values
        f["lom__ds_dt_per_yr"][lom_row_index] = np.float32(ds_dt)
        f["lom__dalpha_dt_rad_per_yr"][lom_row_index] = np.float32(dalpha_dt)

        # Metadata
        f["event_id"][lom_row_index] = f"lom_{record.event_id}"
        f["source_event_id"][lom_row_index] = record.event_id
        f["starfield_seed"][lom_row_index] = np.uint64(record.starfield_seed)
        # shard_row_index: carry the SOURCE shard position (not output position)
        f["shard_row_index"][lom_row_index] = np.uint32(record.shard_row_index)
        # lom_row_index: output ordering within this shard
        f["lom_row_index"][lom_row_index] = np.uint32(lom_row_index)

        self._rows_written += 1

    def resize(self, actual_n: int, rows_failed: int = 0) -> None:
        """Trim all datasets to ``actual_n`` rows and record the failure count.

        Must be called before the context manager exits if any events were
        skipped or failed (i.e. ``actual_n < expected_n``).  The runner is
        responsible for computing ``rows_failed`` and passing it here.

        Parameters
        ----------
        actual_n :
            True number of committed rows (rows_written).
        rows_failed :
            Number of events that were attempted but skipped due to physics
            errors or single-lens passthrough decisions.  Stored as a root attr.
        """
        self._rows_failed = rows_failed
        assert self._f is not None
        if actual_n == self._expected_n:
            return
        f = self._f
        f["science_stamps"].resize((actual_n, self._n_epochs, *SCIENCE_STAMP_SHAPE))
        for ds_name in (*LABEL_DATASET_NAMES, *LOM_LABEL_DATASET_NAMES):
            f[ds_name].resize((actual_n,))
        for ds_name in (
            "event_id", "source_event_id",
            "starfield_seed", "shard_row_index", "lom_row_index",
        ):
            f[ds_name].resize((actual_n,))

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        f = self._f
        self._f = None

        if exc_type is None and f is not None:
            # Write partial-failure metadata before flushing.
            f.attrs["rows_written"] = self._rows_written
            f.attrs["rows_failed"] = self._rows_failed
            f.flush()
            f.close()
            fd = os.open(str(self._tmp_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(self._tmp_path), str(self._target_path))
            dir_fd = os.open(str(self._target_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        else:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            if self._tmp_path.exists():
                self._tmp_path.unlink()

        return False  # propagate exceptions
