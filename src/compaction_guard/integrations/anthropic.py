"""Anthropic: pause-and-verify helpers (REASSERTED) and a SessionStart hook helper (UNOBSERVED).

Two surfaces with very different visibility, and the helpers say which is
which instead of blurring them.

Server-side compaction on the Messages API (the ``compact_20260112``
context edit) with ``pause_after_compaction: true`` pauses the run with
the compaction summary inspectable. That is the one real hook server-side
compaction offers. ``verify_paused_summary`` runs full detection over the
summary text (``Guard.check``, Mode REASSERTED), and ``reassert_summary``
returns that summary with stale sentinel blocks stripped and the current
block appended, for the host to place in the continuation request.
Guarantee: the summary was actually examined, and the returned text
carries the current checksummed block. It cannot supply a before side
(``chars_before`` is None; eviction and survival-site attribution need
the wrapper), and it cannot verify the host sends the reasserted text.
Injection is the host's act; that is what REASSERTED means.

The Anthropic agent SDK harness fires a ``SessionStart`` hook with
``source="compact"`` after a compaction the hook cannot inspect.
``session_start_context`` returns the block for the hook to inject and
emits an UNOBSERVED report; ``session_start_hook_output`` wraps it in the
hook-output shape the harness reads. Guarantee: the text handed over is
the current registry, rendered and checksummed. It cannot see the
summary, verify the injection landed, or check presence later, and the
report says so: ``repaired`` stays False and every finding is
UNVERIFIABLE. This is the weakest posture in the library, labeled as
exactly that.

Neither surface needs the ``anthropic`` SDK: both are strings in, strings
out, so this module imports no framework and a hook process needs nothing
but this library installed.
"""

from __future__ import annotations

import time
from typing import Any

from ..guard import Guard
from ..render import expected_checksum, strip_blocks
from ..report import CompactionReport
from ._shared import emit_unobserved

__all__ = [
    "COMPACT_SOURCE",
    "HOOK_EVENT_SESSION_START",
    "reassert_summary",
    "session_start_context",
    "session_start_hook_output",
    "verify_and_reassert",
    "verify_paused_summary",
]

COMPACT_SOURCE = "compact"
"""The SessionStart source value that means a compaction just happened."""

HOOK_EVENT_SESSION_START = "SessionStart"
"""The hook event name the harness expects in hook output."""


def verify_paused_summary(guard: Guard[Any], summary_text: str) -> CompactionReport:
    """Run full detection over a paused compaction's summary text.

    A named delegate to ``Guard.check`` so the pause flow reads as two
    steps with matching names: verify, then re-assert. The host extracts
    the summary text from the paused response; this library ships no API
    client and does not parse response envelopes. Report bookkeeping
    (``last_report``, ``on_report``, the removal ledger) rides along
    exactly as it does at every other boundary.
    """
    return guard.check(summary_text)


def reassert_summary(guard: Guard[Any], summary_text: str) -> str:
    """The summary with stale blocks stripped and the current block appended.

    Mirrors the codec's string injection rules, so repeated pauses
    converge on exactly one current block instead of accumulating them.
    With an empty registry the text is returned unchanged: injecting an
    empty block would spend tokens to protect nothing declared.
    """
    if not guard.invariants():
        return summary_text
    stripped = strip_blocks(summary_text)
    block = guard.reassertion_block()
    if stripped.strip():
        return stripped.rstrip("\n") + "\n\n" + block
    return block


def verify_and_reassert(
    guard: Guard[Any], summary_text: str
) -> tuple[CompactionReport, str]:
    """The whole pause flow in order: examine, then rebuild the summary.

    Nothing here raises on bad findings, whatever the guard's policy:
    ``check`` is pure verification, and gating belongs to the ``compact()``
    boundary. A host that wants to stop the continuation on a bad summary
    reads ``report.gating`` and decides; it is holding the pause either
    way.
    """
    report = verify_paused_summary(guard, summary_text)
    return report, reassert_summary(guard, summary_text)


def session_start_context(guard: Guard[Any], source: str) -> str | None:
    """The reassertion block for a SessionStart hook, on compaction only.

    Returns None for every other source (startup, resume, clear): no
    compaction happened, so injecting would just duplicate the block. For
    ``source="compact"`` the UNOBSERVED report is emitted before the block
    is returned, because the compaction is real even though nothing about
    it is inspectable from a hook. ``repaired`` stays False: handing text
    to the harness is an offer of repair, not proof of one, and the block
    checksum rides in the note so a later transcript audit can look for
    it.
    """
    if source != COMPACT_SOURCE:
        return None
    started = time.perf_counter()
    registered = guard.invariants()
    evidence = (
        "the compacted transcript is not visible to a SessionStart hook; "
        "no detector ran"
    )
    if not registered:
        emit_unobserved(
            guard,
            started=started,
            chars_after=0,
            repaired=False,
            block_checksum=None,
            note=None,
            evidence=evidence,
        )
        return None
    block = guard.reassertion_block()
    emit_unobserved(
        guard,
        started=started,
        chars_after=len(block),
        repaired=False,
        block_checksum=None,
        note=(
            "session_start(source=compact): reassertion block emitted for "
            f"hook injection (sha256={expected_checksum(registered)}); "
            "presence in the compacted transcript is not verifiable from a "
            "hook"
        ),
        evidence=evidence,
    )
    return block


def session_start_hook_output(guard: Guard[Any], source: str) -> dict[str, Any]:
    """``session_start_context`` wrapped in the hook-output JSON shape.

    A hook script can ``json.dumps`` this straight to stdout; the harness
    reads ``hookSpecificOutput.additionalContext`` from SessionStart
    hooks. An empty dict means nothing to inject, which hooks treat as a
    no-op.
    """
    context = session_start_context(guard, source)
    if context is None:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT_SESSION_START,
            "additionalContext": context,
        }
    }
