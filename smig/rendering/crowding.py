"""
smig/rendering/crowding.py
===========================
Phase 2 crowded-field scene composition for the SMIG v2 pipeline.

Provides :class:`CrowdedFieldRenderer`, which assembles a static neighbor
background by rendering all stars from a neighbor catalog (absolute detector
coordinates) into a single-stamp image, with explicit dictionary-based
caching keyed on a stable tuple of PSF-array hash, stamp centre, stamp size,
and pixel scale.

Phase 2 module — requires ``galsim`` and ``pandas``.
Install with: ``pip install -e '.[phase2]'``.

Catalog provenance
------------------
Phase 2 tests use **synthetic uniform-random catalogs** generated at call
time.  File-based catalogs derived from Galaxia population-synthesis grids
are a **Phase 3 deliverable**.

Architecture boundary
---------------------
This module must never import from ``smig.sensor``.  The sensor chain
receives plain ``np.ndarray`` photon maps from the rendering layer; GalSim
objects never cross the rendering→sensor boundary.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import galsim as _galsim_type
    import pandas as _pd_type

_GALSIM_AVAILABLE: bool = False
_galsim: Any = None
try:
    import galsim as _galsim  # type: ignore[assignment]
    _GALSIM_AVAILABLE = True
except ImportError:
    pass

_PANDAS_AVAILABLE: bool = False
_pd: Any = None
try:
    import pandas as _pd  # type: ignore[assignment]
    _PANDAS_AVAILABLE = True
except ImportError:
    pass

_REQUIRED_COLUMNS: tuple[str, ...] = ("x_pix", "y_pix", "flux_e", "mag_w146")
_POSITION_COLUMNS: tuple[str, ...] = ("x_pix", "y_pix")


class CrowdedFieldRenderer:
    """Render a static crowded-field background from a neighbor star catalog.

    The catalog stores stars in **absolute detector pixel coordinates**.  At
    render time, per-star offsets relative to a caller-specified stamp centre
    are computed and the neighbor flux is convolved with the supplied PSF
    using a single GalSim FFT (all DeltaFunctions are summed before
    convolution for efficiency).

    Results are cached in an explicit per-instance dictionary keyed on a
    stable ``(psf_fingerprint, stamp_center, stamp_size, pixel_scale)``
    tuple.  A second call with identical arguments returns a copy of the
    cached ``np.ndarray`` without recomputation.  Callers should supply a
    ``psf_fingerprint`` — a deterministic hashable value derived from the
    PSF's configuration inputs (e.g. ``(psf_config_hash, sca_id,
    field_position, jitter_seed)``) — so the key is stable across Python
    objects and process restarts.  When ``psf_fingerprint`` is omitted,
    ``id(psf)`` is used as a fallback (only correct when the same Python
    object is consistently reused, as in unit tests).

    Parameters
    ----------
    neighbor_catalog:
        :class:`pandas.DataFrame` with columns ``x_pix``, ``y_pix``,
        ``flux_e``, and ``mag_w146``.  Validated at construction.  Position
        columns must be floating-point.  No NaN values are permitted in any
        required column.
    stamp_size:
        Side length of the output stamp in pixels (square).
    pixel_scale:
        Arcseconds per pixel.  Default is 0.11 (Roman WFI native plate scale).
    brightness_cap_mag:
        If provided, stars with ``mag_w146 > brightness_cap_mag`` are
        completely excluded from rendering.  ``mag_w146`` is metadata only;
        ``flux_e`` is authoritative for flux values.

    Raises
    ------
    ImportError
        If ``galsim`` or ``pandas`` is not installed.
    ValueError
        If *neighbor_catalog* fails validation (missing columns, NaN values,
        or non-float position dtype).
    """

    def __init__(
        self,
        neighbor_catalog: "_pd_type.DataFrame",
        stamp_size: int = 64,
        pixel_scale: float = 0.11,
        brightness_cap_mag: float | None = None,
    ) -> None:
        if not _GALSIM_AVAILABLE:
            raise ImportError(
                "galsim is required for CrowdedFieldRenderer. "
                "Install the Phase 2 extras: pip install -e '.[phase2]'"
            )
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for CrowdedFieldRenderer. "
                "Install the Phase 2 extras: pip install -e '.[phase2]'"
            )

        self._catalog = self._validate_catalog(neighbor_catalog)
        self._stamp_size = stamp_size
        self._pixel_scale = pixel_scale
        self._brightness_cap_mag = brightness_cap_mag

        # Explicit instance dict — avoids @functools.lru_cache self-leak.
        self._static_field_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Catalog validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_catalog(df: "_pd_type.DataFrame") -> "_pd_type.DataFrame":
        """Validate *df* and return a clean copy.

        Enforces
        --------
        1. All required columns are present.
        2. No NaN or Inf values in any required column.
        3. Position columns (``x_pix``, ``y_pix``) are floating-point dtype.
        4. ``flux_e`` is non-negative.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"neighbor_catalog is missing required columns: {missing}. "
                f"Required: {list(_REQUIRED_COLUMNS)}"
            )

        nan_cols = [c for c in _REQUIRED_COLUMNS if df[c].isna().any()]
        if nan_cols:
            raise ValueError(
                f"neighbor_catalog contains NaN values in columns: {nan_cols}."
            )

        inf_cols = [
            c for c in _REQUIRED_COLUMNS
            if _pd.api.types.is_float_dtype(df[c]) and np.isinf(df[c]).any()
        ]
        if inf_cols:
            raise ValueError(
                f"neighbor_catalog contains Inf values in columns: {inf_cols}."
            )

        if (df["flux_e"] < 0.0).any():
            raise ValueError(
                "neighbor_catalog contains negative values in column 'flux_e'."
            )

        for col in _POSITION_COLUMNS:
            if not _pd.api.types.is_float_dtype(df[col]):
                raise ValueError(
                    f"neighbor_catalog column '{col}' must be float dtype, "
                    f"got {df[col].dtype}."
                )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_static_field(
        self,
        psf: "_galsim_type.InterpolatedImage",
        stamp_center_detector_pix: tuple[float, float],
        psf_fingerprint: tuple | str | None = None,
    ) -> np.ndarray:
        """Render all catalog neighbors into a stamp centred on
        *stamp_center_detector_pix*.

        All neighbor fluxes are rendered as :class:`galsim.DeltaFunction`
        profiles.  For efficiency, all shifted deltas are summed into a
        single :class:`galsim.Sum` object before a single PSF convolution,
        so the FFT is performed once regardless of catalog size.

        Stars fainter than ``brightness_cap_mag`` (when set) are excluded.

        The result is stored in ``self._static_field_cache`` and a copy is
        returned immediately on subsequent calls with identical arguments.

        Parameters
        ----------
        psf:
            Polychromatic PSF as a :class:`galsim.InterpolatedImage`.
        stamp_center_detector_pix:
            ``(x, y)`` absolute detector-pixel coordinates of the stamp
            centre.  Offsets are computed as
            ``dx = star.x_pix - stamp_center_x``.
        psf_fingerprint:
            A deterministic, hashable value that uniquely identifies the
            PSF's physics — typically
            ``(psf_config_hash, sca_id, field_position, jitter_seed)``.
            Using this avoids both expensive array serialisation and the
            memory-address instability of ``id(psf)``.  When ``None``
            (default), ``id(psf)`` is used as a fallback, which is only
            correct when the same Python object is consistently reused
            (acceptable in unit tests; **not** correct in production where
            ``STPSFProvider`` creates a new object per call).

        Returns
        -------
        np.ndarray
            2-D ``float64`` array of shape ``(stamp_size, stamp_size)``
            containing rendered neighbor flux in electrons.  Never returns
            a GalSim Image object.
        """
        # Build a deterministic cache key. Use the caller-supplied fingerprint
        # when available; fall back to id(psf) only for tests where the same
        # Python object is reused across calls.
        psf_key = psf_fingerprint if psf_fingerprint is not None else id(psf)
        cache_key: tuple = (
            psf_key,
            stamp_center_detector_pix,
            self._stamp_size,
            self._pixel_scale,
        )
        if cache_key in self._static_field_cache:
            # Return a copy so callers cannot corrupt the cached array in-place.
            return self._static_field_cache[cache_key].copy()

        # --- Filter by brightness cap (mag_w146 metadata only) ---
        cat = self._catalog
        if self._brightness_cap_mag is not None:
            cat = cat[cat["mag_w146"] <= self._brightness_cap_mag]

        # --- Allocate output image ---
        field_image = _galsim.Image(
            self._stamp_size,
            self._stamp_size,
            scale=self._pixel_scale,
        )

        if len(cat) > 0:
            center_x, center_y = stamp_center_detector_pix
            # Extract columns as contiguous numpy arrays to avoid pandas Series overhead.
            x_arr = cat["x_pix"].to_numpy()
            y_arr = cat["y_pix"].to_numpy()
            flux_arr = cat["flux_e"].to_numpy()
            # Build shifted DeltaFunction profiles for every star.
            profiles: list[Any] = []
            for x, y, flux in zip(x_arr, y_arr, flux_arr):
                dx_pix = float(x) - center_x
                dy_pix = float(y) - center_y
                delta = _galsim.DeltaFunction(flux=float(flux))
                # shift() takes arcsec; convert from pixels.
                profiles.append(
                    delta.shift(dx_pix * self._pixel_scale, dy_pix * self._pixel_scale)
                )

            # Single FFT via Sum → Convolve — O(1) FFTs regardless of N stars.
            total_profile = _galsim.Sum(profiles)
            convolved = _galsim.Convolve([total_profile, psf])
            convolved.drawImage(image=field_image, method="auto")

        result = np.array(field_image.array, dtype=np.float64)
        self._static_field_cache[cache_key] = result
        return result

    def count_neighbors_rendered(self) -> int:
        """Return the number of catalog stars that pass the brightness-cap filter.

        Replicates the filter applied inside :meth:`render_static_field` so the
        orchestrator can record the post-cap neighbor count in provenance without
        accessing private attributes.

        Returns
        -------
        int
            Number of stars in the catalog with ``mag_w146 <= brightness_cap_mag``
            (or all stars when no cap is set).
        """
        cat = self._catalog
        if self._brightness_cap_mag is not None:
            cat = cat[cat["mag_w146"] <= self._brightness_cap_mag]
        return int(len(cat))
