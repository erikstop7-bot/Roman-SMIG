"""smig.microlensing — Phase 3.3 microlensing physics engine + event-source binding."""
from __future__ import annotations

from smig.microlensing.backends import get_primary_backend
from smig.microlensing.binding import bind_event_to_source
from smig.microlensing.errors import ClaretGridError, MicrolensingComputationError
from smig.microlensing.event import EventClass, MicrolensingEvent, SourceProperties
from smig.microlensing.priors import sample_event

__all__ = [
    "EventClass",
    "MicrolensingEvent",
    "SourceProperties",
    "bind_event_to_source",
    "get_primary_backend",
    "sample_event",
    "MicrolensingComputationError",
    "ClaretGridError",
]
