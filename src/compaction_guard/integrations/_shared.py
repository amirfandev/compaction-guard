"""Report plumbing shared by the adapters. Not a public surface.

The one rule here comes from the guard: every emitted report sets
``last_report``, fires ``on_report``, and drains the removal ledger, no
matter which door the compaction came through. An adapter that built
reports without that bookkeeping would give hosts two telemetry streams
with different guarantees, so this module goes through the guard's own
private emission path instead of imitating it.

UNOBSERVED findings are synthesised here rather than by any detector.
When a compaction is opaque there is no summary to examine, no detector
runs, and one UNVERIFIABLE per invariant is the strongest claim a missing
rendering supports. ``decided_by`` says ``integration.unobserved`` so the
decided_by distribution never attributes these rows to a layer that never
executed.
"""

from __future__ import annotations

import time
from typing import Any

from ..guard import Guard
from ..report import CompactionReport, Finding
from ..taxonomy import Kind, Mode

__all__ = ["UNOBSERVED_DECIDED_BY", "emit_unobserved"]

UNOBSERVED_DECIDED_BY = "integration.unobserved"
"""Attribution for findings synthesised because no summary was inspectable."""

_EMPTY_REGISTRY_NOTE = "no invariants registered"


def emit_unobserved(
    guard: Guard[Any],
    *,
    started: float,
    chars_after: int,
    repaired: bool,
    block_checksum: str | None,
    note: str | None,
    evidence: str,
) -> CompactionReport:
    """Build and emit one UNOBSERVED report through the guard's bookkeeping.

    ``evidence`` states why nothing could be examined and lands on every
    finding, so each row explains itself in a log. With an empty registry
    the note is exactly "no invariants registered", matching the wording
    ``compact()`` and ``check()`` use, and there are no findings because
    nothing registered could have been lost.
    """
    registered = guard.invariants()
    findings = tuple(
        Finding(
            invariant_id=invariant.id,
            kind=Kind.UNVERIFIABLE,
            severity=invariant.severity,
            decided_by=UNOBSERVED_DECIDED_BY,
            evidence=evidence,
            score=None,
            survived_in=None,
            at_risk=False,
        )
        for invariant in registered
    )
    report = CompactionReport(
        mode=Mode.UNOBSERVED,
        findings=findings,
        chars_before=None,
        chars_after=chars_after,
        repaired=repaired,
        block_checksum=block_checksum,
        removed=guard._drain_removed(),
        duration_ms=(time.perf_counter() - started) * 1000.0,
        note=note if registered else _EMPTY_REGISTRY_NOTE,
    )
    guard._emit(report)
    return report
