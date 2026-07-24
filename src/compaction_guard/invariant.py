"""The Invariant record and its parsing constructor.

An ``Invariant`` is a registered constraint plus everything derived from it at
registration time: anchors, topic set, token cost, id. All derivation happens
once, in ``Invariant.parse``, so the record is immutable and every later
comparison is against fixed data. Detection must never depend on when you look
at the registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .anchors import Anchor, extract_anchors, extract_topic
from .normalize import normalize
from .taxonomy import Severity

__all__ = ["Invariant", "derive_id"]


def derive_id(text: str) -> str:
    """The default invariant id: 12 hex chars of sha256 over the normalised text.

    Hashing the normalised form, not the raw form, is deliberate: two
    registrations that differ only in case, punctuation or invisible
    characters are the same constraint, and they should collide into a
    ``DuplicateInvariantId`` at ``add()`` time rather than pin the same rule
    twice under two ids. Twelve hex chars keep block lines readable while
    leaving collisions between genuinely different constraints implausible
    at registry scale (tens of invariants, not millions).
    """
    return sha256(normalize(text).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Invariant:
    """A registered constraint, frozen with everything derived from it.

    ``text`` is canonical: this exact string is what must survive compaction
    verbatim, what the sentinel block re-injects, and what detectors compare
    the summary against. Nothing downstream ever edits it.
    """

    id: str
    text: str
    severity: Severity
    anchors: tuple[Anchor, ...]
    topic: frozenset[str]
    source: str | None
    token_cost: int

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        id: str | None = None,
        severity: Severity = Severity.BLOCK,
        source: str | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ) -> Invariant:
        """Build an Invariant from raw text, deriving everything else.

        Leading and trailing whitespace is stripped because it is framing
        from the call site, not constraint content; interior whitespace and
        newlines are kept untouched, since ``text`` must remain verbatim.
        Empty text and ids containing whitespace are rejected here rather
        than at render time, so a bad registration fails at the line that
        caused it. ``token_cost`` is fixed now, with the estimator the guard
        will use for budget accounting, so the number on the record and the
        number in the budget can never drift apart.
        """
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("invariant text is empty after stripping whitespace")
        if id is not None:
            if not id or any(ch.isspace() for ch in id):
                raise ValueError(
                    f"invariant id {id!r} is empty or contains whitespace; "
                    "ids appear as single tokens in the sentinel wire format"
                )
        estimator = token_estimator
        if estimator is None:
            # Imported here, not at module top: budget.py sits above this
            # module in the layering (it renders whole blocks), and the
            # default estimator is the only thing needed from it.
            from .budget import estimate_tokens

            estimator = estimate_tokens
        anchors = extract_anchors(cleaned)
        return cls(
            id=id if id is not None else derive_id(cleaned),
            text=cleaned,
            severity=severity,
            anchors=anchors,
            topic=extract_topic(cleaned, anchors),
            source=source,
            token_cost=estimator(cleaned),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form. ``topic`` is sorted so serialisation is stable

        across processes; frozenset iteration order depends on hash seeding
        and would otherwise make byte-identical output runs impossible.
        """
        return {
            "id": self.id,
            "text": self.text,
            "severity": self.severity.value,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "topic": sorted(self.topic),
            "source": self.source,
            "token_cost": self.token_cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Invariant:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            severity=Severity(data["severity"]),
            anchors=tuple(Anchor.from_dict(a) for a in data["anchors"]),
            topic=frozenset(data["topic"]),
            source=data["source"],
            token_cost=int(data["token_cost"]),
        )
