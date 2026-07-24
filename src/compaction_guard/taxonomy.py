"""The names, in one place: verdict kinds, severities, policies, modes, and orderings.

This module holds every enumeration the rest of the library agrees on, and
nothing that computes. ``Policy`` and ``Mode`` live here rather than in
``guard.py`` for a structural reason as much as a stylistic one: reports carry
a ``Mode``, the guard applies a ``Policy``, and if either enum lived in the
orchestration module the data layer would have to import the behaviour layer
to name its own fields.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "GATING_KINDS",
    "SEVERITY_ORDER",
    "Kind",
    "Mode",
    "Policy",
    "Severity",
]


class Kind(StrEnum):
    """What the compaction did to one invariant, as one of seven verdicts.

    The taxonomy maps onto the published failure structure of this problem:
    omission versus commission (Slipstream, arXiv 2605.08580) and the decay of
    soft constraints under summarisation (Governance Decay, arXiv 2606.22528).
    ``MUTATED`` is named separately from ``CONTRADICTED`` because value
    confusion is its own failure class: semantic layers score ``$500`` and
    ``$5000`` as near-identical, so only deterministic anchor comparison
    catches it.

    ``UNVERIFIABLE`` is a first-class verdict, not an error. It means the
    installed detector layers were exhausted without a sound answer, and it is
    consumed by ``fail_closed`` at the guard level. Integrity failures (a
    missing or edited sentinel block) are never expressed through this enum;
    those raise ``BlockIntegrityError`` because they indicate a harness bug,
    not summariser behaviour to classify.
    """

    PRESERVED = "preserved"
    """Text survives verbatim or near-verbatim after normalisation."""

    PARAPHRASED = "paraphrased"
    """Content survives in different words; values and obligation force intact."""

    WEAKENED = "weakened"
    """Topic survives; obligation force or scope reduced. "Must not" gone,
    "production database" became "the database"."""

    MUTATED = "mutated"
    """Structure survives but a bound value or identifier changed or vanished.
    $500 became $5000, or the wrong table name."""

    CONTRADICTED = "contradicted"
    """Post-compaction text asserts the negation or an incompatible permission,
    such as "writes are fine here"."""

    DROPPED = "dropped"
    """No lexical or semantic trace remains."""

    UNVERIFIABLE = "unverifiable"
    """Installed layers exhausted without a verdict. Consumed by fail_closed."""


class Severity(StrEnum):
    """How much weight an invariant carries when findings land on it."""

    BLOCK = "block"
    """Findings on this invariant trigger the guard's policy."""

    WARN = "warn"
    """Reported, never gates, regardless of the finding's kind."""


class Policy(StrEnum):
    """What the guard does with findings at the compaction boundary."""

    REPAIR = "repair"
    """Default: re-inject the sentinel block, verify it, return the context.
    This is constraint pinning, the intervention with published evidence
    behind it (Governance Decay, arXiv 2606.22528), and it never breaks a
    healthy run."""

    RAISE = "raise"
    """Gate: raise InvariantViolation on gating findings; repair on pass."""

    WARN = "warn"
    """Return the compactor's output unchanged; report only. The observer
    posture, obtained as a policy value instead of a second architecture."""


class Mode(StrEnum):
    """How much of the compaction the guard could actually see.

    Degradation against opaque compaction is a visible field on every report,
    never fine print. The ecosystem is splitting into stacks you can wrap and
    stacks you can only re-assert around, and a report that hid which side it
    ran on would overstate its own claims.
    """

    OWNED = "owned"
    """The guard ran the compactor; full verify and repair applied."""

    REASSERTED = "reasserted"
    """Summary text was inspectable; the block is injected by host code."""

    UNOBSERVED = "unobserved"
    """Opaque compaction; presence asserted, summary uninspectable."""


SEVERITY_ORDER: tuple[Kind, ...] = (
    Kind.CONTRADICTED,
    Kind.MUTATED,
    Kind.DROPPED,
    Kind.WEAKENED,
    Kind.UNVERIFIABLE,
    Kind.PARAPHRASED,
    Kind.PRESERVED,
)
"""Verdict kinds from worst to best. ``CompactionReport.worst`` uses this order.

``MUTATED`` outranks ``DROPPED`` deliberately. A wrong live value drives
confident wrong action: an agent holding "$5000 cap" spends against it without
hesitating. Absence at least sometimes triggers a clarifying question, because
the agent knows it does not know. Commission beats omission in the damage
ranking even though omission is far more common, and the ordering encodes that
judgement so no caller has to re-derive it.

``UNVERIFIABLE`` sits between ``WEAKENED`` and ``PARAPHRASED``: worse than any
verified survival, better than any verified loss, because it asserts nothing.
"""


GATING_KINDS: frozenset[Kind] = frozenset(
    {Kind.CONTRADICTED, Kind.MUTATED, Kind.DROPPED, Kind.WEAKENED}
)
"""The kinds that trigger ``Policy.RAISE`` when they land on a BLOCK invariant.

``UNVERIFIABLE`` joins this set only when the guard is constructed with
``fail_closed=True``; that extension is applied by the guard, not encoded
here, because the report data model does not carry the fail_closed flag.
``PARAPHRASED`` never gates: content that survived in different words with
values and force intact is a summariser doing its job.
"""
