"""
smig/sensor/memory_profiler.py
================================
Peak memory measurement stub for the SMIG v2 detector simulation.

Returns peak resident memory in megabytes during a simulation run.
A real implementation would use ``tracemalloc``, ``resource``, or
``psutil`` depending on platform availability.  The stub returns None
so that callers can gracefully handle the absence of profiling data
(e.g. by recording None in the provenance record's ``peak_memory_mb``
field) without breaking the simulation pipeline.
"""

from __future__ import annotations

__all__ = ["get_peak_memory_mb"]


def get_peak_memory_mb() -> float | None:
    """Return peak resident memory consumed so far, in megabytes.

    Returns
    -------
    float | None
        Peak memory in MB, or None if memory profiling is not available
        in this environment.

    Notes
    -----
    Stub implementation — always returns None.  To enable real profiling,
    replace this body with a call to ``tracemalloc.get_traced_memory()[1]``
    (divided by 1024**2) or ``psutil.Process().memory_info().rss``
    (divided by 1024**2).
    """
    return None
