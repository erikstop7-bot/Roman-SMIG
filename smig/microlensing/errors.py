"""Exceptions for smig.microlensing."""
from __future__ import annotations


class MicrolensingComputationError(Exception):
    """Raised when VBBinaryLensing returns an unphysical or failed result.

    Never raised for PSPL or FSPL (those use analytic / semi-analytic paths).
    """

    def __init__(self, params: dict, cause: BaseException | None = None) -> None:
        self.params = params
        self.cause = cause
        msg = f"VBBL computation failed: params={params}"
        if cause is not None:
            msg += f" (caused by {type(cause).__name__}: {cause})"
        super().__init__(msg)


class ClaretGridError(Exception):
    """Raised when source parameters are outside the Claret 2000 LD grid.

    Only raised when strict_ld_grid=True.
    """

    def __init__(self, teff_K: float, log_g: float, feh: float, band: str) -> None:
        self.teff_K = teff_K
        self.log_g = log_g
        self.feh = feh
        self.band = band
        super().__init__(
            f"Source (Teff={teff_K} K, log_g={log_g}, [Fe/H]={feh}) is outside the "
            f"Claret 2000 LD grid for band={band!r}. Use strict_ld_grid=False to allow "
            f"nearest-neighbour fallback."
        )
