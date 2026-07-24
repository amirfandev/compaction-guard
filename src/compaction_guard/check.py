"""The pure functional core: invariants plus summary text in, report out.

``check()`` is the library without the wrapper. No Guard, no registry state,
no codec, no compactor: hand it invariants and the text a compaction
produced, get back a full ``CompactionReport``. This is what makes the
library testable (every verdict fixture is one ``check()`` call) and
embeddable (a host that owns its own compaction loop can verify summaries
without adopting the Guard at all). ``Guard.check`` delegates here, and
``Guard.compact`` runs the same detector loop through ``examine_view`` with
a diff-attributed view instead of a bare-text one.

Reports from this function carry ``mode=REASSERTED``: the summary text was
inspectable, so full drift detection applies, but the guard did not run the
compactor and injection is the host's responsibility. ``chars_before`` is
None because there is no before side to measure, and ``repaired`` is False
because a pure function repairs nothing.

The seam with the detector layer is deliberately thin. All chain semantics
(cheap-to-expensive ordering, ``can_issue`` enforcement, the escalation
matrix, exhaustion to UNVERIFIABLE) live in ``detectors.base.DetectorChain``;
this module only decides which detectors run when the caller names none: the
stdlib ``LexicalDetector``, alone, because that is the zero-dependency
install's honest capability.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from .detectors.base import Detector, DetectorChain, SummaryView
from .detectors.lexical import LexicalDetector
from .diff import view_from_text
from .invariant import Invariant
from .report import CompactionReport, Finding
from .taxonomy import Mode

__all__ = ["check", "examine_view"]


def examine_view(
    invariants: Sequence[Invariant],
    view: SummaryView,
    *,
    detectors: Sequence[Detector] | None = None,
) -> tuple[Finding, ...]:
    """Run the detector chain over every invariant against one shared view.

    One view, many invariants: the view is the expensive part (sentence
    splitting, normalisation, site attribution) and it is identical for
    every constraint, so it is computed by the caller exactly once.
    ``detectors=None`` means the default chain; an explicitly empty sequence
    is respected and yields UNVERIFIABLE for everything, which is the honest
    result of asking no one.

    Findings come back in registry order, one per invariant, never fewer:
    an invariant the chain cannot judge still gets a finding, because a
    report with silently missing rows would read as a report with nothing
    wrong.
    """
    chain = DetectorChain(
        tuple(detectors) if detectors is not None else (LexicalDetector(),)
    )
    return tuple(chain.examine(invariant, view) for invariant in invariants)


def check(
    invariants: Sequence[Invariant | str] | Invariant | str,
    summary_text: str,
    *,
    detectors: Sequence[Detector] | None = None,
) -> CompactionReport:
    """Verify ``summary_text`` against ``invariants``. No side effects.

    Accepts what ``Guard`` accepts: ``Invariant`` objects or bare constraint
    strings, mixed freely, and a single one of either for convenience. The
    asymmetry where the constructor took strings but this function crashed
    on them (deep in a generator, naming neither the argument nor the fix)
    was a confirmed adoption trap; parsing here costs microseconds and
    removes it. A bare string is treated as one constraint, not iterated
    character by character.

    The empty-registry note matches ``Guard.compact``'s wording exactly:
    a report with zero findings must say why there are zero, or a clean
    report over an empty registry becomes indistinguishable from a clean
    report over a verified one.
    """
    started = time.perf_counter()
    if isinstance(invariants, (Invariant, str)):
        invariants = (invariants,)
    parsed = tuple(
        item if isinstance(item, Invariant) else Invariant.parse(item)
        for item in invariants
    )
    view = view_from_text(summary_text)
    findings = examine_view(parsed, view, detectors=detectors)
    duration_ms = (time.perf_counter() - started) * 1000.0
    return CompactionReport(
        mode=Mode.REASSERTED,
        findings=findings,
        chars_before=None,
        chars_after=len(summary_text),
        repaired=False,
        block_checksum=None,
        removed=(),
        duration_ms=duration_ms,
        note=None if parsed else "no invariants registered",
    )
