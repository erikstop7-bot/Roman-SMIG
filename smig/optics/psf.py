"""
smig/optics/psf.py
==================
Field-varying polychromatic PSF provider for the Roman Space Telescope WFI.

Phase 2 module — requires ``galsim`` (mandatory) and ``webbpsf`` (optional).
Install with: ``pip install -e '.[phase2]'``.

Design notes
------------
* ``galsim`` and ``webbpsf`` are probed at module import time via guarded
  ``try/except ImportError`` blocks.  ``ImportError`` is only *raised* inside
  ``STPSFProvider.__init__`` so the class can be inspected on a base-only
  install without crashing.
* ``._backend`` is ``'webbpsf'`` when ``webbpsf`` is importable, else
  ``'analytic'``.
* ``webbpsf.roman.WFI()`` is lazily instantiated on the first PSF computation
  call (instantiation costs ~10 s; doing it in ``__init__`` would stall every
  ``STPSFProvider`` construction even if no PSFs are ever needed).
* Monochromatic cache keys exclude jitter parameters; polychromatic keys include
  them. This ensures ``get_psf_at_wavelength`` results are reusable across
  different jitter configurations.
* Field positions are clamped to ``[0, 1]`` and quantized to 4 decimal places
  before being embedded in cache keys to prevent float-formatting instability.
* HDF5 disk cache failures (file lock, concurrent access, full disk) are caught
  silently so cache unavailability never crashes a simulation run.
* Edge taper: when ``psf_edge_taper_pixels`` is set in :class:`PSFConfig`, a 2-D
  cosine window is multiplied into the raw array after computation (before jitter)
  on both backends.  Taper trades tiny wing accuracy for less rectangular FFT
  ringing; it is not a substitute for larger arrays.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from smig.config.optics_schemas import PSFConfig

if TYPE_CHECKING:
    import galsim as _galsim_type

# ---------------------------------------------------------------------------
# Optional backend probes — ImportError deferred to STPSFProvider.__init__
# ---------------------------------------------------------------------------

_GALSIM_AVAILABLE: bool = False
_galsim: Any = None
try:
    import galsim as _galsim  # type: ignore[assignment]
    _GALSIM_AVAILABLE = True
except ImportError:
    pass

_WEBBPSF_AVAILABLE: bool = False
_webbpsf: Any = None
try:
    import webbpsf as _webbpsf  # type: ignore[assignment]
    _WEBBPSF_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_ROMAN_PRIMARY_DIAMETER_M: float = 2.4       # Roman primary mirror diameter (m)
_NATIVE_PIXEL_SCALE_ARCSEC: float = 0.11     # WFI native plate scale (arcsec/pixel)
_FIELD_POSITION_QUANTIZE_DP: int = 4         # Decimal places for cache key
_MEMORY_CACHE_MAX: int = 200                 # Max PSF arrays in in-process cache

# ---------------------------------------------------------------------------
# Edge-taper helper
# ---------------------------------------------------------------------------


def _cosine_taper_1d(n: int, w: int) -> np.ndarray:
    """Return a 1-D cosine taper of length *n* with ramp width *w* pixels.

    Values are 0 at indices 0 and n-1, ramp to 1 over the first/last *w*
    pixels, and are 1 in the interior.  *w* is clamped to ``n // 2`` so the
    two ramp regions never overlap.
    """
    t = np.ones(n, dtype=np.float64)
    w = min(w, n // 2)
    if w <= 0:
        return t
    ramp = np.arange(w, dtype=np.float64)
    vals = 0.5 * (1.0 - np.cos(np.pi * ramp / w))
    t[:w] = vals
    t[n - w:] = vals[::-1]
    return t

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


# SMIG uses "W146" as the wide-band label; STPSF expects the STScI name "F146".
_STPSF_FILTER_ALIAS: dict[str, str] = {
    "W146": "F146",
}


def _resolve_stpsf_filter_name(filter_name: str) -> str:
    """Map a SMIG filter label to the name expected by the STPSF/WebbPSF WFI object.

    Returns the input unchanged when it is already a valid STPSF filter name.
    The mapping is only consulted in the WebbPSF code path.
    """
    return _STPSF_FILTER_ALIAS.get(filter_name, filter_name)


def _normalize_sca_id(sca_id: str | int) -> str:
    """Normalize an SCA identifier to canonical ``'SCA{n:02d}'`` format.

    Accepts: ``'SCA01'``, ``'SCA1'``, ``1``, ``'1'``, etc.
    Returns: ``'SCA01'`` … ``'SCA18'``.

    Raises
    ------
    ValueError
        If the numeric part is outside the range 1–18.
    """
    if isinstance(sca_id, int):
        n = sca_id
    else:
        s = str(sca_id).strip().upper()
        n = int(s[3:]) if s.startswith("SCA") else int(s)
    if not (1 <= n <= 18):
        raise ValueError(f"SCA ID must be in 1–18, got {n!r}.")
    return f"SCA{n:02d}"


def _quantize_field_position(
    field_position: tuple[float, float],
) -> tuple[float, float]:
    """Clamp each coordinate to ``[0, 1]`` and round to 4 decimal places.

    Prevents float-formatting instability from creating unbounded cache entries
    for imperceptibly different field positions.
    """
    x = round(max(0.0, min(1.0, float(field_position[0]))), _FIELD_POSITION_QUANTIZE_DP)
    y = round(max(0.0, min(1.0, float(field_position[1]))), _FIELD_POSITION_QUANTIZE_DP)
    return (x, y)


# ---------------------------------------------------------------------------
# Bounded in-process cache
# ---------------------------------------------------------------------------


class _BoundedCache:
    """Thread-safe bounded cache using an access-ordered LRU eviction policy.

    Evicts the least-recently-accessed entry when the limit is exceeded.
    Both ``get`` and ``put`` promote the key to the end of the ordering, so
    frequently accessed keys are protected from premature eviction.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            if key in self._data:
                self.hits += 1
                self._data.move_to_end(key)
                return self._data[key].copy()
            self.misses += 1
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


# ---------------------------------------------------------------------------
# STPSFProvider
# ---------------------------------------------------------------------------


class STPSFProvider:
    """Field-varying polychromatic PSF provider for the Roman WFI.

    Computes wavelength-dependent, field-position-varying PSFs using either
    the WebbPSF backend (when installed via Phase 2 extras) or an analytic
    Airy+Gaussian fallback.

    The ``._backend`` attribute is ``'webbpsf'`` or ``'analytic'`` and is set
    at construction time based on import availability.

    Parameters
    ----------
    config:
        PSF sub-configuration from :class:`~smig.config.optics_schemas.PSFConfig`.

    Raises
    ------
    ImportError
        If ``galsim`` is not installed.  Install Phase 2 extras::

            pip install -e '.[phase2]'
    """

    def __init__(self, config: PSFConfig) -> None:
        # Defer the ImportError to construction time (not module import time).
        if not _GALSIM_AVAILABLE:
            raise ImportError(
                "galsim is required for STPSFProvider. "
                "Install the Phase 2 extras: pip install -e '.[phase2]'"
            )

        self._config = config
        self._backend: str = "webbpsf" if _WEBBPSF_AVAILABLE else "analytic"

        # Log-spaced wavelength grid spanning the filter bandpass.
        self._wavelengths_um: np.ndarray = np.geomspace(
            config.wavelength_range_um[0],
            config.wavelength_range_um[1],
            config.n_wavelengths,
        )

        # Oversampled pixel scale used for GalSim image creation.
        self._pixel_scale_arcsec: float = (
            _NATIVE_PIXEL_SCALE_ARCSEC / config.oversample
        )

        # Canonical hash of the full PSFConfig for cache key derivation.
        self._config_hash: str = hashlib.sha256(
            config.model_dump_json(round_trip=True).encode("utf-8")
        ).hexdigest()

        # Lazy WebbPSF instrument — expensive (~10 s) to instantiate.
        self._instrument: Any = None
        self._instrument_lock = threading.Lock()

        # Bounded in-process LRU cache.
        self._cache = _BoundedCache(maxsize=_MEMORY_CACHE_MAX)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def cache_hits(self) -> int:
        """Number of in-process cache hits (memory cache only)."""
        return self._cache.hits

    @property
    def cache_misses(self) -> int:
        """Number of in-process cache misses (memory cache only)."""
        return self._cache.misses

    @property
    def psf_config_hash(self) -> str:
        """SHA-256 hex digest of the full PSFConfig used to construct this provider."""
        return self._config_hash

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_psf_at_wavelength(
        self,
        sca_id: str | int,
        field_position: tuple[float, float],
        wavelength_um: float,
    ) -> np.ndarray:
        """Compute a normalized monochromatic PSF at a single wavelength.

        The result is a 2-D ``float64`` array where the sum over all pixels
        equals 1.0 (total-flux normalization).  Results are cached to memory
        (and optionally to an HDF5 disk cache when
        ``config.cache_dir`` is set).

        Jitter parameters are **not** included in the monochromatic cache key;
        these pre-jitter kernels are reusable across different jitter
        configurations.

        Parameters
        ----------
        sca_id:
            SCA identifier: string (``'SCA01'``) or integer (``1``–``18``).
        field_position:
            ``(x, y)`` fractional position on the SCA focal plane, each in
            ``[0, 1]``.  ``(0.5, 0.5)`` is the detector centre.
        wavelength_um:
            Wavelength in micrometres.

        Returns
        -------
        np.ndarray
            2-D ``float64`` PSF array, ``sum() == 1.0 ± 1e-6``.
        """
        # Resolve backend first so cache keys are stable (backend may switch to
        # 'analytic' on first instrument init when STPSF data files are absent).
        self._ensure_backend_resolved()

        sca = _normalize_sca_id(sca_id)
        qfp = _quantize_field_position(field_position)
        cache_key = self._mono_cache_key(sca, qfp, wavelength_um)

        # 1. In-process memory cache.
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # 2. Disk cache.
        disk_hit = self._load_from_disk(cache_key)
        if disk_hit is not None:
            self._cache.put(cache_key, disk_hit)
            return disk_hit

        # 3. Compute.
        if self._backend == "webbpsf":
            array = self._compute_webbpsf_mono(sca, qfp, wavelength_um)
        else:
            array = self._compute_analytic_mono(qfp, wavelength_um)

        # Apply edge taper before normalization (taper-off path is a no-op).
        array = self._apply_edge_taper(array)

        # Normalize to sum = 1.0.
        total = float(array.sum())
        if total <= 0.0:
            raise ValueError(
                f"PSF at wavelength={wavelength_um} μm, sca={sca!r}, "
                f"field={qfp} produced non-positive total flux ({total:.6g}). "
                "Cannot normalize."
            )
        array = array / total

        # Persist.
        self._cache.put(cache_key, array)
        self._save_to_disk(cache_key, array, {
            "sca_id": sca,
            "field_x": qfp[0],
            "field_y": qfp[1],
            "wavelength_um": wavelength_um,
            "oversample": self._config.oversample,
            "filter_name": self._config.filter_name,
            "backend": self._backend,
            "type": "monochromatic",
        })

        return array

    def get_psf(
        self,
        sca_id: str | int,
        field_position: tuple[float, float],
        source_sed: str = "flat",
        jitter_seed: int | None = None,
    ) -> "_galsim_type.InterpolatedImage":
        """Compute a polychromatic, jitter-convolved PSF.

        Assembles the polychromatic PSF by weighted summation of monochromatic
        PSFs over the log-spaced wavelength grid, then convolves with a
        jitter kernel for a specific, seed-driven realization.

        Parameters
        ----------
        sca_id:
            SCA identifier: string (``'SCA01'``) or integer (``1``–``18``).
        field_position:
            ``(x, y)`` fractional position on the SCA focal plane, each in
            ``[0, 1]``.
        source_sed:
            Source SED type.  Currently only ``'flat'`` (equal weight across
            the bandpass) is supported.
        jitter_seed:
            Seed for the jitter realization.  Same seed → same jitter.
            Different seed → different sub-pixel centroid offset.
            ``None`` → deterministic default derived from
            ``config_hash``.

        Returns
        -------
        galsim.InterpolatedImage
            Polychromatic PSF with pixel scale
            ``0.11 / oversample`` arcsec per pixel.

        Raises
        ------
        ValueError
            If ``source_sed`` is not supported.
        """
        if source_sed != "flat":
            raise ValueError(
                f"Unsupported SED type {source_sed!r}. Only 'flat' is supported."
            )

        self._ensure_backend_resolved()

        sca = _normalize_sca_id(sca_id)
        qfp = _quantize_field_position(field_position)
        cache_key = self._poly_cache_key(sca, qfp, source_sed, jitter_seed)

        # Memory cache.
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._to_interpolated_image(cached)

        # Disk cache.
        disk_hit = self._load_from_disk(cache_key)
        if disk_hit is not None:
            self._cache.put(cache_key, disk_hit)
            return self._to_interpolated_image(disk_hit)

        # Compute monochromatic PSFs and combine.
        mono_psfs = [
            self.get_psf_at_wavelength(sca, qfp, wl)
            for wl in self._wavelengths_um
        ]

        # Flat SED: equal wavelength weights, normalized to sum = 1.0.
        weights = np.ones(self._config.n_wavelengths, dtype=np.float64)
        weights /= weights.sum()

        poly_psf = np.zeros_like(mono_psfs[0])
        for w, mono in zip(weights, mono_psfs):
            poly_psf += w * mono

        # Normalize.
        total = float(poly_psf.sum())
        if total > 0.0:
            poly_psf = poly_psf / total

        # Apply jitter (Gaussian blur + seed-dependent sub-pixel shift).
        if self._config.jitter_rms_mas > 0.0:
            poly_psf = self._apply_jitter(poly_psf, jitter_seed)
            total = float(poly_psf.sum())
            if total > 0.0:
                poly_psf = poly_psf / total

        # Persist.
        self._cache.put(cache_key, poly_psf)
        self._save_to_disk(cache_key, poly_psf, {
            "sca_id": sca,
            "field_x": qfp[0],
            "field_y": qfp[1],
            "source_sed": source_sed,
            "jitter_seed": str(jitter_seed),
            "backend": self._backend,
            "type": "polychromatic",
        })

        return self._to_interpolated_image(poly_psf)

    # ------------------------------------------------------------------
    # Cache key computation
    # ------------------------------------------------------------------

    def _mono_cache_key(
        self,
        sca: str,
        qfp: tuple[float, float],
        wavelength_um: float,
    ) -> str:
        """SHA-256 cache key for a pre-jitter monochromatic PSF.

        Excludes jitter parameters so the same kernel is reused across
        different ``jitter_seed`` and ``jitter_rms_mas`` values.
        """
        payload = (
            f"mono|sca={sca}|"
            f"fp=({qfp[0]:.4f},{qfp[1]:.4f})|"
            f"wl={wavelength_um:.6f}|"
            f"os={self._config.oversample}|"
            f"filter={self._config.filter_name}|"
            f"cfg={self._config_hash}|"
            f"backend={self._backend}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _poly_cache_key(
        self,
        sca: str,
        qfp: tuple[float, float],
        source_sed: str,
        jitter_seed: int | None,
    ) -> str:
        """SHA-256 cache key for the final polychromatic jitter-convolved PSF.

        Includes all parameters that affect the final output, including
        ``n_wavelengths`` and ``wavelength_range_um`` so that grid changes
        invalidate the cache.
        """
        seed_str = str(jitter_seed) if jitter_seed is not None else "none"
        lo, hi = self._config.wavelength_range_um
        payload = (
            f"poly|sca={sca}|"
            f"fp=({qfp[0]:.4f},{qfp[1]:.4f})|"
            f"sed={source_sed}|"
            f"jseed={seed_str}|"
            f"jitter_rms_mas={self._config.jitter_rms_mas:.6f}|"
            f"nwl={self._config.n_wavelengths}|"
            f"wlrange=({lo:.6f},{hi:.6f})|"
            f"cfg={self._config_hash}|"
            f"backend={self._backend}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------
    # WebbPSF backend
    # ------------------------------------------------------------------

    def _ensure_backend_resolved(self) -> None:
        """Finalize self._backend before any cache key is computed.

        Triggers lazy WebbPSF instrument construction so that a missing-data
        OSError is caught here and permanently switches _backend to 'analytic'.
        Subsequent calls are no-ops (backend is already resolved).
        """
        if self._backend == "webbpsf":
            self._get_instrument()

    def _get_instrument(self) -> Any:
        """Lazily instantiate and return the WebbPSF Roman WFI instrument.

        Uses double-checked locking so only one thread pays the ~10 s
        instantiation cost.  If the STPSF data files are missing, falls back
        to the analytic backend permanently for this provider instance.
        """
        if self._backend != "webbpsf":
            return None
        if self._instrument is None:
            with self._instrument_lock:
                if self._instrument is None:
                    try:
                        wfi = _webbpsf.roman.WFI()
                        wfi.filter = _resolve_stpsf_filter_name(self._config.filter_name)
                        wfi.options["output_mode"] = "oversampled"
                        self._instrument = wfi
                    except (OSError, EnvironmentError, FileNotFoundError) as exc:
                        import warnings
                        warnings.warn(
                            f"STPSF data files unavailable ({exc}); "
                            "falling back to analytic PSF backend.",
                            stacklevel=2,
                        )
                        self._backend = "analytic"
                        return None
        return self._instrument

    def _compute_webbpsf_mono(
        self,
        sca: str,
        qfp: tuple[float, float],
        wavelength_um: float,
    ) -> np.ndarray:
        """Compute a monochromatic PSF using WebbPSF.

        Maps fractional ``field_position`` to detector pixel coordinates on
        the 4096×4096 SCA grid and calls
        ``webbpsf.roman.WFI.calc_psf(monochromatic=...)`` with the wavelength
        expressed in **metres** (WebbPSF convention).
        """
        inst = self._get_instrument()
        # Map fractional [0,1] → integer detector indices [0, 4095].
        # Multiply by 4095 (not 4096) so that fp=1.0 lands on pixel 4095,
        # not 4096, which is outside the valid range for WebbPSF.
        det_x = qfp[0] * 4095.0
        det_y = qfp[1] * 4095.0
        inst.detector = sca
        inst.detector_position = (det_x, det_y)

        wavelength_m = wavelength_um * 1e-6  # WebbPSF expects metres
        hdulist = inst.calc_psf(
            monochromatic=wavelength_m,
            oversample=self._config.oversample,
            fov_pixels=self._config.fov_native_pixels,
        )

        # Extract oversampled PSF array; try named extension first.
        psf_data = None
        for extname in ("OVERSAMP", "DET_SAMP"):
            try:
                if hdulist[extname].data is not None:
                    psf_data = hdulist[extname].data
                    break
            except (KeyError, AttributeError):
                pass
        if psf_data is None:
            for hdu in hdulist:
                if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                    psf_data = hdu.data
                    break
        hdulist.close()

        if psf_data is None:
            raise RuntimeError(
                f"WebbPSF calc_psf returned no 2-D PSF data for "
                f"sca={sca!r}, wavelength={wavelength_um} μm."
            )
        return np.array(psf_data, dtype=np.float64)

    # ------------------------------------------------------------------
    # Analytic fallback backend
    # ------------------------------------------------------------------

    def _compute_analytic_mono(
        self,
        qfp: tuple[float, float],
        wavelength_um: float,
    ) -> np.ndarray:
        """Compute an analytic Airy+Gaussian PSF (fallback without WebbPSF).

        Wavelength dependence
        ~~~~~~~~~~~~~~~~~~~~~
        GalSim's :class:`~galsim.Airy` profile has FWHM ∝ λ/D, so the PSF is
        physically broader at longer wavelengths.  ``lam`` must be in
        **nanometres** (GalSim convention).

        Field-position dependence
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        A field-position-dependent Gaussian aberration is convolved with the
        Airy disk.  The aberration FWHM grows quadratically with the distance
        ``r`` from the SCA centre ``(0.5, 0.5)``::

            fwhm_aberration = 0.02 × (1 + 2 × r²)  arcsec

        At the centre (r = 0): 0.02 arcsec.
        At the corner (r ≈ 0.707): 0.04 arcsec.
        This ensures the PSF measurably varies with field position even without
        WebbPSF data files.
        """
        wavelength_nm = wavelength_um * 1000.0  # GalSim Airy expects nanometres

        # Airy disk: diffraction-limited profile for a 2.4 m mirror.
        airy = _galsim.Airy(
            lam=wavelength_nm,
            diam=_ROMAN_PRIMARY_DIAMETER_M,
            obscuration=0.0,  # simplified; Roman has ~30% central obscuration
        )

        # Field-position-dependent aberration Gaussian.
        r = float(np.sqrt((qfp[0] - 0.5) ** 2 + (qfp[1] - 0.5) ** 2))
        fwhm_aberration_arcsec = 0.02 * (1.0 + 2.0 * r ** 2)
        gaussian_aberration = _galsim.Gaussian(fwhm=fwhm_aberration_arcsec)

        psf_profile = _galsim.Convolve([airy, gaussian_aberration])

        # PSF stamp size comes from config — GalSim InterpolatedImage returns hard
        # zero outside this boundary, so fov_native_pixels must match the rendering
        # context stamp to avoid rectangular clipping artefacts.
        n_pix = self._config.fov_native_pixels * self._config.oversample
        image = _galsim.Image(n_pix, n_pix, scale=self._pixel_scale_arcsec)
        psf_profile.drawImage(image=image, method="auto")

        return np.array(image.array, dtype=np.float64)

    # ------------------------------------------------------------------
    # Jitter
    # ------------------------------------------------------------------

    def _resolve_jitter_seed(self, jitter_seed: int | None) -> int:
        """Resolve ``jitter_seed`` to a concrete deterministic integer.

        When ``None``, derives a fixed seed from the ``config_hash`` so the
        default realization is reproducible but distinct from every explicit
        integer seed.
        """
        if jitter_seed is not None:
            return int(jitter_seed)
        digest = hashlib.sha256(
            f"smig/v2/psf/jitter:{self._config_hash}".encode("utf-8")
        ).hexdigest()
        return 1 + (int(digest, 16) % (2**31 - 1))

    def _apply_jitter(
        self,
        psf_array: np.ndarray,
        jitter_seed: int | None,
    ) -> np.ndarray:
        """Apply a two-component jitter model to the PSF.

        1. **Gaussian blur** (``sigma = jitter_rms_pixels``) — increases FWHM;
           this is the dominant effect of line-of-sight pointing jitter.
        2. **Seed-dependent sub-pixel shift** — applies a small centroid
           displacement drawn from the seeded RNG, creating realization-specific
           PSF asymmetry.  Same seed → same shift; different seed → different
           shift.

        Parameters
        ----------
        psf_array:
            2-D float64 PSF array to convolve.
        jitter_seed:
            Seed for the random sub-pixel shift.  ``None`` → deterministic
            default derived from the config hash.
        """
        from scipy.ndimage import gaussian_filter
        from scipy.ndimage import shift as ndimage_shift

        jitter_rms_arcsec = self._config.jitter_rms_mas / 1000.0
        jitter_rms_pixels = jitter_rms_arcsec / self._pixel_scale_arcsec

        # 1. Gaussian blur — the primary FWHM-broadening component.
        # mode="reflect" avoids the hard step to zero at the array boundary that
        # mode="constant" introduces; that step is the source of axial (crosshair)
        # ringing visible under aggressive image normalization.
        result = gaussian_filter(
            psf_array,
            sigma=jitter_rms_pixels,
            mode="reflect",
        )

        # 2. Sub-pixel centroid offset — provides seed-specific variation.
        #    Shift magnitude ≈ jitter_rms_pixels / 4 to keep it sub-dominant.
        seed = self._resolve_jitter_seed(jitter_seed)
        rng = np.random.default_rng(seed)
        sigma_shift = max(jitter_rms_pixels / 4.0, 0.01)
        shift_yx = rng.normal(0.0, sigma_shift, size=2)
        result = ndimage_shift(result, shift_yx, mode="reflect")

        return result

    # ------------------------------------------------------------------
    # Edge taper
    # ------------------------------------------------------------------

    def _apply_edge_taper(self, array: np.ndarray) -> np.ndarray:
        """Multiply a 2-D PSF array by a separable cosine edge taper.

        Applied after raw array computation on both backends, before the
        sum-to-1 normalization step — flux conservation is handled by the
        caller's existing normalization.  Returns ``array`` unchanged when
        ``psf_edge_taper_pixels`` is None or <= 0.
        """
        w_cfg = self._config.psf_edge_taper_pixels
        if w_cfg is None or w_cfg <= 0.0:
            return array
        ny, nx = array.shape
        w = int(np.ceil(float(w_cfg)))
        taper = np.outer(
            _cosine_taper_1d(ny, min(w, ny // 2)),
            _cosine_taper_1d(nx, min(w, nx // 2)),
        )
        return array * taper

    # ------------------------------------------------------------------
    # GalSim wrapping
    # ------------------------------------------------------------------

    def _to_interpolated_image(
        self, array: np.ndarray
    ) -> "_galsim_type.InterpolatedImage":
        """Wrap a 2-D numpy array as a :class:`galsim.InterpolatedImage`.

        The pixel scale is ``0.11 / config.oversample`` arcsec per pixel,
        matching the oversampled Roman WFI grid.
        """
        gs_image = _galsim.Image(array, scale=self._pixel_scale_arcsec)
        return _galsim.InterpolatedImage(gs_image)

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _get_cache_path(self) -> Path | None:
        """Return the HDF5 cache file path, or ``None`` if caching is disabled."""
        if self._config.cache_dir is None:
            return None
        cache_dir = Path(self._config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "psf_cache.h5"

    def _load_from_disk(self, cache_key: str) -> np.ndarray | None:
        """Attempt to load a cached PSF array from the HDF5 disk cache.

        Returns ``None`` if caching is disabled, the file does not exist,
        the key is absent, or any I/O error occurs.
        """
        cache_path = self._get_cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        try:
            import h5py
            with h5py.File(cache_path, "r") as f:
                if cache_key in f:
                    return np.array(f[cache_key], dtype=np.float64)
        except Exception:
            # Silently ignore any I/O or locking error.
            pass
        return None

    def _save_to_disk(
        self,
        cache_key: str,
        array: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        """Save a PSF array to the HDF5 disk cache with human-readable attrs.

        Uses ``h5py.File(..., 'a')`` (append mode).  All exceptions are
        swallowed so that concurrent access or a full disk never crashes the
        caller.

        HDF5 layout::

            /{sha256_hex_64chars}             2-D float64 dataset
                .attrs["created_utc"]         ISO-8601 timestamp
                .attrs[key]                   string-encoded metadata
        """
        cache_path = self._get_cache_path()
        if cache_path is None:
            return
        try:
            import h5py
            with h5py.File(cache_path, "a") as f:
                if cache_key not in f:
                    ds = f.create_dataset(cache_key, data=array)
                    ds.attrs["created_utc"] = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    for k, v in metadata.items():
                        ds.attrs[k] = str(v)
        except Exception:
            pass
