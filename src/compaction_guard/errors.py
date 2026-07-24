"""Every exception this library raises, in one flat module.

The split between exceptions and findings is load-bearing. A finding describes
summariser behaviour: text drifted, a value changed, a constraint vanished.
Those are data, reported and acted on by policy. An exception means the
machinery itself cannot be trusted or refuses to proceed: the sentinel block
the guard wrote was edited, the codec cannot write into the context, the
token budget would be exceeded. Classifying an integrity failure as a finding
would let a broken harness masquerade as a chatty summariser, so the boundary
is hard: findings for behaviour, exceptions for bugs and refusals.

Host code that wants a single net can catch ``CompactionGuardError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .report import CompactionReport

__all__ = [
    "BlockIntegrityError",
    "BudgetExceeded",
    "CodecError",
    "CompactionGuardError",
    "DuplicateInvariantId",
    "InvariantViolation",
]


class CompactionGuardError(Exception):
    """Base class for everything this library raises."""


class InvariantViolation(CompactionGuardError):
    """Raised under ``Policy.RAISE`` when a gating finding lands on a BLOCK invariant.

    Carries the full report so the caller can inspect exactly what the
    summariser did without re-running detection. When this raises, the
    compacted context is not returned; the caller still holds the original,
    which is the point of gating: do not proceed on this summary.
    """

    report: CompactionReport

    def __init__(self, message: str, report: CompactionReport) -> None:
        super().__init__(message)
        self.report = report


class BlockIntegrityError(CompactionGuardError):
    """The sentinel block is missing or was edited after the guard wrote it.

    Always an exception, never a finding. A summariser cannot cause this on
    the repair path: the block is regenerated from the registry after
    summarisation, so a missing or mangled block at verification time means
    something between the guard and the model rewrote guard-owned bytes.
    That is a harness bug, and classifying it as summariser behaviour would
    hide it inside a report nobody treats as fatal.
    """


class CodecError(CompactionGuardError):
    """The codec could not render or inject for this context shape.

    A ``render`` failure downgrades every invariant to UNVERIFIABLE, because
    no verdict may be stronger than the rendering supports. An ``inject``
    failure under REPAIR is a hard error: a guard that cannot write cannot
    pin, and pretending otherwise would be fake protection.
    """


class BudgetExceeded(CompactionGuardError):
    """Registering this invariant would push the sentinel block past its budget.

    Raised at ``add()`` time, loudly, and nothing is ever truncated. A
    truncated constraint is a mutated constraint, injected by the tool whose
    whole job is to detect mutation. The outs are explicit: remove an
    invariant with a recorded reason, or raise ``max_block_tokens``.
    """


class DuplicateInvariantId(CompactionGuardError):
    """Two invariants resolved to the same id.

    With derived ids this usually means the same constraint text (up to
    normalisation) was registered twice, which is worth surfacing rather
    than silently double-pinning.
    """
