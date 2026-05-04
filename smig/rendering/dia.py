"""
smig/rendering/dia.py
=====================
Difference Image Analysis (DIA) pipeline for SMIG v2 Phase 2.

Provides :class:`DIAPipeline`, which:

1. Builds a multi-epoch inverse-variance-weighted reference image from
   pre-rendered context stamps (``build_reference``).
2. Performs MVP Alard-Lupton kernel-basis image subtraction (``subtract``).
3. Extracts a science-size central crop from the difference image
   (``extract_stamp``).

Architecture boundary
---------------------
This module imports **only** from ``smig.config.schemas`` and
``smig.config.optics_schemas`` for config types.  No sensor-physics or
detector-pipeline modules from ``smig.sensor.*`` are imported.

Mixed-fidelity approximations (MVP)
------------------------------------
* Reference construction uses a **scalar variance** per epoch (read noise +
  dark + sky background).  Per-pixel Poisson variance from source photons is
  intentionally omitted for MVP speed.  MULTIACCUM covariance and IPC are
  also omitted.  This is a pragmatic mixed-fidelity approximation.
* Alard-Lupton subtraction uses **spatially constant** 3-Gaussian kernel
  basis; polynomial spatial variation is a future enhancement.
* A single additive background constant is fit; higher-order background
  polynomials are a future enhancement.
* The 4-parameter LSQ fit (3 basis convolutions + constant) can absorb a
  small *global* perturbation between science and reference into a
  coefficient adjustment, suppressing it from the residual.  For LOM
  ablation the perturbation is per-epoch flux modulation, so this matters
  most at small ds_dt where the Δflux is below the constant-term resolution.
  Expand the basis or fit per-epoch (rather than per-pair) if this becomes
  the limiting factor in the residual CDF.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from smig.config.optics_schemas import DIAConfig
from smig.config.schemas import DetectorConfig


class DIAPipeline:
    """MVP Difference Image Analysis pipeline.

    Parameters
    ----------
    config:
        DIA-specific configuration (stamp sizes, reference depth,
        subtraction method).
    detector_config:
        Phase 1 detector configuration.  Exposure time, read noise, and
        dark current are derived from this object — no hardcoded constants.
    rng:
        NumPy random Generator injected by the caller.  All stochastic
        operations (noise injection in ``build_reference``) use *this*
        generator exclusively.  The caller is responsible for seeding.
    """

    # Alard-Lupton basis: 4 Gaussian sigmas (pixels).
    # σ=0.5 acts as a near-identity kernel (G_{0.5}⊛PSF ≈ PSF), allowing the
    # fit to represent "identity" when science and reference PSFs are identical
    # or nearly so.  Without it the fit must approximate the identity as a sum
    # of wider Gaussians, creating a positive core + negative ring structural
    # residual that spans the full science stamp.
    _AL_SIGMAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

    def __init__(
        self,
        config: DIAConfig,
        detector_config: DetectorConfig,
        rng: np.random.Generator,
    ) -> None:
        self._config = config
        self._detector = detector_config
        self._rng = rng

        # Guard: AL kernel must be strictly smaller than the context stamp.
        # kernel_size = 2*ceil(4*sigma_max)+1 = 33 px for _AL_SIGMAS=(1,2,4).
        # When kernel_size >= context_stamp_size every pixel in the stamp falls
        # within the kernel's half-width of a boundary; scipy's boundary="symm"
        # padding then contaminates *every* entry of the AL design matrix and
        # the least-squares fit diverges.  This guard fires only for the
        # alard_lupton path because SFFT does not use these Gaussian bases.
        if config.subtraction_method == "alard_lupton":
            _al_kernel_size = 2 * int(np.ceil(4.0 * max(self._AL_SIGMAS))) + 1
            _ctx = config.context_stamp_size
            if _al_kernel_size >= _ctx:
                _min_ctx = _al_kernel_size + config.science_stamp_size
                raise ValueError(
                    f"AL kernel_size={_al_kernel_size} px (sigma_max="
                    f"{max(self._AL_SIGMAS):.1f} px, support=4σ) must be "
                    f"strictly less than context_stamp_size={_ctx} px.  "
                    f"With the current stamp every pixel is within the "
                    f"kernel's half-width of a boundary; 'symm' padding "
                    f"corrupts the entire AL design matrix.  "
                    f"Set context_stamp_size >= {_al_kernel_size + 1} "
                    f"(recommended: >= {_min_ctx} so the boundary-free "
                    f"interior covers the {config.science_stamp_size}-px "
                    f"science stamp; 64 px is a safe default)."
                )

        # Derived constants — all extracted from DetectorConfig, no hardcoding
        self._t_exp_s: float = (
            (detector_config.readout.n_ramp_reads - 1)
            * detector_config.readout.frame_time_s
        )
        if self._t_exp_s <= 0.0:
            raise ValueError(
                f"Derived exposure time t_exp_s={self._t_exp_s!r} must be > 0. "
                "Check readout.n_ramp_reads (must be >= 2) and readout.frame_time_s."
            )
        self._read_noise_e: float = (
            detector_config.electrical.read_noise_cds_electrons
        )
        self._dark_e_per_s: float = (
            detector_config.electrical.dark_current_e_per_s
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_reference(
        self,
        ideal_electrons_epochs: list[np.ndarray],
        backgrounds_e_per_s: list[float],
    ) -> np.ndarray:
        """Inverse-variance weighted coadd of baseline epochs in rate space.

        Each epoch is independently noise-injected using a scalar per-epoch
        variance (read noise + dark + sky background).  Source Poisson noise
        is intentionally omitted (see module docstring).  The resulting noisy
        rate images are combined via strict inverse-variance weighting.

        Parameters
        ----------
        ideal_electrons_epochs:
            List of 2D arrays, each of shape
            ``(context_stamp_size, context_stamp_size)``, representing
            ideal (noiseless) rendered electrons for one reference epoch.
            An epoch-specific PSF may have been applied upstream before
            passing arrays here.
        backgrounds_e_per_s:
            Sky-background rate in electrons/s for each epoch.  Must have
            the same length as ``ideal_electrons_epochs``.

        Returns
        -------
        np.ndarray
            2D reference image in rate space (e⁻/s), dtype float64, shape
            ``(context_stamp_size, context_stamp_size)``.

        Raises
        ------
        ValueError
            If any input array is not 2D, has wrong shape, or if the list
            lengths do not match.
        """
        ctx = self._config.context_stamp_size
        expected_shape = (ctx, ctx)

        # --- Input validation ---
        if len(ideal_electrons_epochs) != len(backgrounds_e_per_s):
            raise ValueError(
                f"len(ideal_electrons_epochs)={len(ideal_electrons_epochs)} must "
                f"equal len(backgrounds_e_per_s)={len(backgrounds_e_per_s)}."
            )
        if len(ideal_electrons_epochs) == 0:
            raise ValueError("ideal_electrons_epochs must not be empty.")

        for i, arr in enumerate(ideal_electrons_epochs):
            if arr.ndim != 2:
                raise ValueError(
                    f"ideal_electrons_epochs[{i}] is {arr.ndim}D; expected 2D."
                )
            if arr.shape != expected_shape:
                raise ValueError(
                    f"ideal_electrons_epochs[{i}] has shape {arr.shape}; "
                    f"expected {expected_shape} (context_stamp_size={ctx})."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"ideal_electrons_epochs[{i}] contains non-finite values (NaN/Inf)."
                )

        for i, bg in enumerate(backgrounds_e_per_s):
            if not np.isfinite(bg):
                raise ValueError(
                    f"backgrounds_e_per_s[{i}]={bg!r} is non-finite (NaN/Inf)."
                )

        t = self._t_exp_s
        if t <= 0.0:
            raise ValueError(
                f"t_exp_s={t!r} must be > 0 before division. "
                "Check readout.n_ramp_reads (must be >= 2) and readout.frame_time_s."
            )
        rn = self._read_noise_e
        dk = self._dark_e_per_s

        weighted_sum = np.zeros(expected_shape, dtype=np.float64)
        weight_total = 0.0

        for ideal_e, bg_e_per_s in zip(ideal_electrons_epochs, backgrounds_e_per_s):
            # Convert ideal electrons to rate space (e-/s)
            rate_image = ideal_e.astype(np.float64, copy=False) / t + bg_e_per_s

            # Expected noise variance in electrons (scalar per epoch)
            # Omits source Poisson noise for MVP; see module docstring.
            var_e = rn**2 + (dk + bg_e_per_s) * t

            # Convert electron variance to rate-space variance
            variance_rate = var_e / (t**2)

            # Inject noise using only the injected RNG — never np.random global
            noise = self._rng.normal(
                0.0, np.sqrt(variance_rate), size=rate_image.shape
            )
            noisy_rate = rate_image + noise

            # Inverse-variance weight (scalar; same for all pixels in this epoch)
            weight = 1.0 / variance_rate

            weighted_sum += weight * noisy_rate
            weight_total += weight

        return (weighted_sum / weight_total).astype(np.float64)

    def subtract(
        self,
        science_rate_image: np.ndarray,
        reference_rate_image: np.ndarray,
    ) -> np.ndarray:
        """Subtract reference from science using the configured DIA method.

        Behavior is selected by ``config.subtraction_method``:

        * ``'identity'`` (default for matched-reference simulated runs):
          returns ``science - reference`` directly, cast to float64.  Appropriate
          only when reference and science share the same PSF and detector chain.
          Current QA shows this preserves both null statistics and lensed
          signal cleanly, whereas the AL fit introduces repeated source-core
          null bias and suppresses real lensed signal.
        * ``'alard_lupton'``: 4-Gaussian-basis (σ = 0.5, 1, 2, 4 px) Alard &
          Lupton (1998) kernel-based subtraction with an additive constant
          background term (no spatial variation).  Retained for PSF-mismatch
          experiments and future real-DIA paths.
        * ``'sfft'``: not yet implemented; raises ``NotImplementedError``.

        Parameters
        ----------
        science_rate_image:
            2D science image in rate space (e⁻/s).
        reference_rate_image:
            2D reference (template) image in rate space (e⁻/s).  Must have
            the same shape as ``science_rate_image``.

        Returns
        -------
        np.ndarray
            Difference image, same shape as inputs and dtype ``float64``.

        Raises
        ------
        NotImplementedError
            If ``config.subtraction_method == 'sfft'``.
        ValueError
            If either input is not 2D, shapes do not match, or any value is
            non-finite.
        """
        if self._config.subtraction_method == "sfft":
            raise NotImplementedError(
                "SFFT subtraction is not yet implemented in this MVP. "
                "Set config.subtraction_method='identity' or 'alard_lupton'."
            )

        # --- Input validation (shared across identity and alard_lupton) ---
        if science_rate_image.ndim != 2:
            raise ValueError(
                f"science_rate_image must be 2D, got {science_rate_image.ndim}D."
            )
        if reference_rate_image.ndim != 2:
            raise ValueError(
                f"reference_rate_image must be 2D, got {reference_rate_image.ndim}D."
            )
        if science_rate_image.shape != reference_rate_image.shape:
            raise ValueError(
                f"science_rate_image.shape {science_rate_image.shape} must match "
                f"reference_rate_image.shape {reference_rate_image.shape}."
            )

        if not np.all(np.isfinite(science_rate_image)):
            raise ValueError(
                "science_rate_image contains non-finite values (NaN/Inf)."
            )
        if not np.all(np.isfinite(reference_rate_image)):
            raise ValueError(
                "reference_rate_image contains non-finite values (NaN/Inf)."
            )

        sci = science_rate_image.astype(np.float64, copy=False)
        ref = reference_rate_image.astype(np.float64, copy=False)

        if self._config.subtraction_method == "identity":
            # Direct rate-space subtraction.  Returned array is always a fresh
            # float64 allocation so the caller can mutate it without aliasing
            # the input arrays.
            return (sci - ref).astype(np.float64, copy=False)

        # Kernel size: 2 * ceil(4 * sigma_max) + 1 = 33 px for sigma_max=4.0
        sigma_max = max(self._AL_SIGMAS)
        kernel_size = 2 * int(np.ceil(4.0 * sigma_max)) + 1

        # Convolve reference with each normalized Gaussian basis kernel
        convolved_planes: list[np.ndarray] = []
        for sigma in self._AL_SIGMAS:
            kernel = self._make_gaussian_kernel(sigma, kernel_size)
            conv = convolve2d(ref, kernel, mode="same", boundary="symm")
            convolved_planes.append(conv)

        ny, nx = sci.shape
        n_pixels = ny * nx
        n_basis = len(convolved_planes)

        # Design matrix A: [N_pixels, n_basis+1] — basis convolutions + constant term
        A = np.empty((n_pixels, n_basis + 1), dtype=np.float64)
        for k, plane in enumerate(convolved_planes):
            A[:, k] = plane.ravel()
        A[:, n_basis] = 1.0  # additive background constant

        b = sci.ravel()

        # Solve least squares: A @ coeffs ≈ b
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Reconstruct matched reference from basis coefficients
        matched_ref = np.zeros((ny, nx), dtype=np.float64)
        for k, plane in enumerate(convolved_planes):
            matched_ref += coeffs[k] * plane
        matched_ref += coeffs[n_basis]  # additive background constant term

        return sci - matched_ref

    def extract_stamp(self, difference_image: np.ndarray) -> np.ndarray:
        """Dynamic central crop to science_stamp_size.

        Crop boundaries are computed dynamically from config:
        center = context_stamp_size // 2, cropped symmetrically by
        science_stamp_size // 2 in each direction.

        Parameters
        ----------
        difference_image:
            2D difference image.  Both dimensions must be >=
            ``science_stamp_size``.  Typically of shape
            ``(context_stamp_size, context_stamp_size)``.

        Returns
        -------
        np.ndarray
            2D array of shape
            ``(science_stamp_size, science_stamp_size)``, dtype preserved.

        Raises
        ------
        ValueError
            If ``difference_image`` is not 2D or either dimension is smaller
            than ``science_stamp_size``.
        """
        sci_size = self._config.science_stamp_size
        ctx_size = self._config.context_stamp_size

        if difference_image.ndim != 2:
            raise ValueError(
                f"difference_image must be 2D, got {difference_image.ndim}D."
            )
        h, w = difference_image.shape
        if h < sci_size or w < sci_size:
            raise ValueError(
                f"difference_image shape {difference_image.shape} has a dimension "
                f"smaller than science_stamp_size={sci_size}."
            )

        # Derive center from actual input dimensions so oversized arrays are
        # handled correctly regardless of config.context_stamp_size.
        center_r = h // 2
        center_c = w // 2
        half = sci_size // 2
        row_start = center_r - half
        col_start = center_c - half
        # Use row_start + sci_size (not center + half) so odd sci_size is exact.
        row_stop = row_start + sci_size
        col_stop = col_start + sci_size

        # Copy ensures the parent array can be freed by the GC after this call.
        return difference_image[row_start:row_stop, col_start:col_stop].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_gaussian_kernel(sigma: float, size: int) -> np.ndarray:
        """Build a 2D Gaussian kernel, normalized so its sum equals 1.0.

        The kernel is centered on the middle pixel of a ``size × size`` grid.

        Parameters
        ----------
        sigma:
            Gaussian standard deviation in pixels.
        size:
            Side length of the square kernel (should be odd so the peak
            falls exactly on the centre pixel).

        Returns
        -------
        np.ndarray
            2D float64 array of shape ``(size, size)`` summing to 1.0.
        """
        half = size // 2
        # mgrid gives integer offsets from -half to +half inclusive
        y, x = np.mgrid[-half : half + 1, -half : half + 1]
        kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
        return kernel
