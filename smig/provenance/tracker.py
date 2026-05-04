"""
smig/provenance/tracker.py
==========================
Accumulates ProvenanceRecord instances for a single microlensing event and
writes a cluster-safe atomic JSON sidecar to disk when the event is complete.

Design constraints
------------------
* One tracker instance per event.
* Records are stored in insertion order; the sidecar is always sorted by
  epoch_index before writing.
* Each (event_id, epoch_index) pair is a unique primary key; duplicate
  epoch_index values are rejected immediately on append.
* git_commit and container_digest are read once at tracker construction
  from the environment variables GIT_COMMIT_SHA and IMAGE_DIGEST
  respectively, with None as the fallback.
* config_sha256, python_version, and numpy_version are anchored from the
  first record appended.  All subsequent records must match exactly.
* Strict drift enforcement: ANY mismatch between the tracker's anchored
  metadata and an incoming record raises ValueError — including None vs.
  non-None mismatches in either direction (no silent partial drift).
* The JSON sidecar is written atomically: data is flushed and fsync'd to
  a temporary file, then os.replace() renames it to the target.  The temp
  file is always outside the with-block before os.replace() is called,
  which is required for correctness on Windows (os.replace() raises
  PermissionError if the file handle is still open).
* The sidecar filename includes a 6-char SHA-256 hash of the raw event_id
  appended to the sanitized name to guarantee uniqueness when two event IDs
  share the same sanitized representation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .schema import ProvenanceRecord

# Sentinel for uninitialized anchored metadata fields.
_UNSET: Any = object()


class ProvenanceTracker:
    """Accumulates and persists provenance records for one microlensing event.

    Parameters
    ----------
    event_id:
        The unique identifier for the event being tracked.  Must match the
        ``event_id`` field of every appended ProvenanceRecord.

    Attributes
    ----------
    event_id : str
        The event identifier supplied at construction.
    git_commit : str | None
        Value of the ``GIT_COMMIT_SHA`` environment variable at construction
        time, or None if unset.
    container_digest : str | None
        Value of the ``IMAGE_DIGEST`` environment variable at construction
        time, or None if unset.

    Example
    -------
    >>> tracker = ProvenanceTracker(event_id="ob230001")
    >>> tracker.append_record(record)          # repeat for each epoch
    >>> tracker.write_sidecar(Path("output/"))
    """

    def __init__(self, event_id: str) -> None:
        self.event_id: str = event_id
        self.git_commit: str | None = os.environ.get("GIT_COMMIT_SHA")
        self.container_digest: str | None = os.environ.get("IMAGE_DIGEST")
        self._records: list[ProvenanceRecord] = []
        self._seen_epochs: set[int] = set()
        # Anchored from the first appended record; _UNSET until then.
        self._config_sha256: Any = _UNSET
        self._python_version: Any = _UNSET
        self._numpy_version: Any = _UNSET

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_record(self, record: ProvenanceRecord) -> None:
        """Append one epoch's provenance record to the in-memory accumulator.

        Parameters
        ----------
        record:
            A fully populated ProvenanceRecord.

        Raises
        ------
        ValueError
            If ``record.event_id`` does not match ``self.event_id``.
        ValueError
            If a record with the same ``epoch_index`` has already been appended.
        ValueError
            If any of the five consistency-anchored metadata fields
            (``git_commit``, ``container_digest``, ``config_sha256``,
            ``python_version``, ``numpy_version``) drifts from the tracker's
            anchored value — including None vs. non-None mismatches in either
            direction (no silent partial drift).
        """
        if record.event_id != self.event_id:
            raise ValueError(
                f"Record event_id '{record.event_id}' does not match "
                f"tracker event_id '{self.event_id}'."
            )

        if record.epoch_index in self._seen_epochs:
            raise ValueError(
                f"Duplicate epoch_index {record.epoch_index} for event "
                f"'{self.event_id}'. Each epoch must have a unique index."
            )

        # Anchor config_sha256, python_version, numpy_version from first record.
        if self._config_sha256 is _UNSET:
            self._config_sha256 = record.config_sha256
            self._python_version = record.python_version
            self._numpy_version = record.numpy_version

        # Strict equality checks — no silent partial drift allowed.
        for field_name, tracker_val, record_val in (
            ("git_commit", self.git_commit, record.git_commit),
            ("container_digest", self.container_digest, record.container_digest),
            ("config_sha256", self._config_sha256, record.config_sha256),
            ("python_version", self._python_version, record.python_version),
            ("numpy_version", self._numpy_version, record.numpy_version),
        ):
            if tracker_val != record_val:
                raise ValueError(
                    f"Metadata drift detected for field '{field_name}': "
                    f"tracker has {tracker_val!r} but record has {record_val!r}.  "
                    "All epochs in an event must share identical environment metadata."
                )

        self._seen_epochs.add(record.epoch_index)
        self._records.append(record)

    def write_sidecar(self, output_dir: Path) -> Path:
        """Atomically serialise all accumulated records to a JSON sidecar file.

        The output filename is
        ``{sanitized_event_id}_{6char_sha256}_provenance.json`` inside
        ``output_dir``.  The 6-character suffix is the first 6 hex chars of
        ``SHA-256(event_id.encode())`` and guarantees filename uniqueness even
        when two event IDs share the same sanitized representation.

        Top-level metadata (``git_commit``, ``container_digest``,
        ``config_sha256``, ``python_version``, ``numpy_version``) is
        sourced from the validated records, not from the tracker's potentially
        unset environment state.

        The write sequence is:
        1. Serialise payload to a NamedTemporaryFile in the same directory.
        2. ``flush()`` and ``fsync()`` while the file handle is still open.
        3. Close the file handle (exits the ``with`` block).
        4. ``os.replace()`` the closed temp file to the target path.

        This order is critical on Windows: ``os.replace()`` raises
        ``PermissionError`` if the file handle is still open.

        Parameters
        ----------
        output_dir:
            Directory in which to write the sidecar.  Must already exist
            and must be a directory (not a file).

        Returns
        -------
        Path
            Absolute path to the written sidecar file.

        Raises
        ------
        ValueError
            If no records have been appended (``_records`` is empty).
        NotADirectoryError
            If ``output_dir`` does not exist or is not a directory.
        """
        if not self._records:
            raise ValueError(
                "No records to write; append at least one ProvenanceRecord "
                "before calling write_sidecar."
            )

        output_dir = Path(output_dir).resolve()
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"output_dir does not exist or is not a directory: {output_dir}"
            )

        # Sanitize event_id to prevent path-traversal in the filename, then
        # append a 6-char SHA-256 hash of the raw event_id to guarantee
        # uniqueness when two IDs share the same sanitized representation.
        safe_event_id = re.sub(r"[^a-zA-Z0-9_\-.]", "_", self.event_id)
        short_hash = hashlib.sha256(self.event_id.encode()).hexdigest()[:6]
        target_path = output_dir / f"{safe_event_id}_{short_hash}_provenance.json"

        sorted_records = sorted(self._records, key=lambda r: r.epoch_index)
        first = sorted_records[0]

        payload = {
            "event_id": self.event_id,
            "git_commit": first.git_commit,
            "container_digest": first.container_digest,
            "config_sha256": first.config_sha256,
            "python_version": first.python_version,
            "numpy_version": first.numpy_version,
            "epoch_count": len(sorted_records),
            "records": [r.model_dump(mode="json") for r in sorted_records],
        }

        # Cluster-safe atomic write.
        # The file must be CLOSED before os.replace() on Windows.
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_dir,
                prefix=f".{safe_event_id}_provenance_",
                suffix=".tmp.json",
                mode="w",
                encoding="utf-8",
                delete=False,
            ) as fh:
                tmp_path = fh.name
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            # File handle is now closed; safe to rename on all platforms.
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

        return target_path

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of records accumulated so far."""
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"ProvenanceTracker("
            f"event_id={self.event_id!r}, "
            f"records={len(self._records)})"
        )
