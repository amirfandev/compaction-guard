"""Findings, the compaction report, and the block budget record.

These are the data the library emits; behaviour lives elsewhere. All three
classes are frozen, slotted, keyword-only, and serialise to plain dicts with
stable key order, because the report's whole job is to be logged: one
``to_json()`` line per compaction is the full telemetry surface of the
package, and two identical runs must produce byte-identical lines (durations
aside) or the determinism test in the suite has nothing to hold on to.

``SurvivalSite`` is imported only for type checking and lazily in
``from_dict``: the data layer must import and run without the detector layer,
both for layering hygiene and so serialisation round-trips do not drag
detection code into processes that only read logs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .taxonomy import GATING_KINDS, SEVERITY_ORDER, Kind, Mode, Severity

if TYPE_CHECKING:
    from .detectors.base import SurvivalSite

__all__ = ["BlockBudget", "CompactionReport", "Finding"]

_RANK: dict[Kind, int] = {kind: rank for rank, kind in enumerate(SEVERITY_ORDER)}


def _survival_site(value: Any) -> SurvivalSite | None:
    if value is None:
        return None
    from .detectors.base import SurvivalSite

    return SurvivalSite(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class Finding:
    """One verdict about one invariant after one compaction.

    ``decided_by`` names the exact rule that produced the verdict
    ("lexical.exact", "lexical.anchor_diff", "embedding.floor",
    "nli.bidirectional", "judge") so every aggregate number decomposes by
    layer and no verdict is ever unattributable. ``evidence`` must be
    recomputable: a matched span, the missing anchors, or a score, never
    free prose.

    ``at_risk`` exists because PRESERVED can lie by omission: a constraint
    that survived only in the kept-verbatim tail is one compaction from
    death, and a report that called that safe would be teaching users the
    wrong thing. It is True when survival depends on RETAINED_TAIL alone.
    """

    invariant_id: str
    kind: Kind
    severity: Severity
    decided_by: str
    evidence: str
    score: float | None = None
    survived_in: SurvivalSite | None = None
    at_risk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "decided_by": self.decided_by,
            "evidence": self.evidence,
            "score": self.score,
            "survived_in": None if self.survived_in is None else self.survived_in.value,
            "at_risk": self.at_risk,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Finding:
        return cls(
            invariant_id=str(data["invariant_id"]),
            kind=Kind(data["kind"]),
            severity=Severity(data["severity"]),
            decided_by=str(data["decided_by"]),
            evidence=str(data["evidence"]),
            score=None if data["score"] is None else float(data["score"]),
            survived_in=_survival_site(data["survived_in"]),
            at_risk=bool(data["at_risk"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockBudget:
    """Token accounting for the sentinel block at a point in time.

    ``used_tokens`` measures the whole rendered block, wire overhead
    included, because that is what actually lands in the context window.
    ``per_invariant`` carries each invariant's registration-time text cost,
    so the two numbers deliberately do not sum: the difference is the price
    of the sentinel format itself, and hiding it inside the per-invariant
    figures would misattribute overhead to whichever constraint happened to
    be counted last.
    """

    used_tokens: int
    max_tokens: int
    per_invariant: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_tokens": self.used_tokens,
            "max_tokens": self.max_tokens,
            "per_invariant": [[inv_id, tokens] for inv_id, tokens in self.per_invariant],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BlockBudget:
        return cls(
            used_tokens=int(data["used_tokens"]),
            max_tokens=int(data["max_tokens"]),
            per_invariant=tuple(
                (str(inv_id), int(tokens)) for inv_id, tokens in data["per_invariant"]
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionReport:
    """Everything one compaction did to the registry, as one loggable record.

    ``mode`` states how much the guard could see (OWNED, REASSERTED,
    UNOBSERVED) so a clean report from an opaque compaction can never be
    mistaken for a verified one. ``chars_before`` is None exactly when there
    was no before-side to measure. ``removed`` carries (id, reason) pairs
    for every eviction since the last report, because removal is the only
    way out of the registry and it should leave a visible trace.
    """

    schema_version: int = 1
    mode: Mode
    findings: tuple[Finding, ...]
    chars_before: int | None
    chars_after: int
    repaired: bool
    block_checksum: str | None
    removed: tuple[tuple[str, str], ...]
    duration_ms: float
    note: str | None = None

    @property
    def worst(self) -> Kind:
        """The most severe kind present, by taxonomy order; PRESERVED if empty.

        PRESERVED for an empty report is honest because an empty findings
        tuple only occurs when there was nothing registered to lose, and the
        note field says so.
        """
        return min(
            (finding.kind for finding in self.findings),
            key=_RANK.__getitem__,
            default=Kind.PRESERVED,
        )

    @property
    def gating(self) -> tuple[Finding, ...]:
        """Findings that gate under ``Policy.RAISE``: gating kind on a BLOCK invariant.

        This property encodes the policy-independent part of the gate. It
        does NOT include UNVERIFIABLE findings; the guard adds those when it
        was constructed with ``fail_closed=True``, because the report does
        not carry that flag and pretending to know it here would be wrong in
        one direction or the other. WARN-severity invariants never appear
        regardless of kind.
        """
        return tuple(
            finding
            for finding in self.findings
            if finding.severity is Severity.BLOCK and finding.kind in GATING_KINDS
        )

    def losses(self) -> tuple[Finding, ...]:
        """Findings strictly worse than PARAPHRASED in the taxonomy order.

        UNVERIFIABLE is included: unverified survival is not survival, and a
        losses list that omitted it would read as "all clear" precisely when
        the detectors could not tell. PARAPHRASED and PRESERVED are the two
        outcomes where the content demonstrably made it through.
        """
        threshold = _RANK[Kind.PARAPHRASED]
        return tuple(
            finding for finding in self.findings if _RANK[finding.kind] < threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "repaired": self.repaired,
            "block_checksum": self.block_checksum,
            "removed": [[inv_id, reason] for inv_id, reason in self.removed],
            "duration_ms": self.duration_ms,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompactionReport:
        return cls(
            schema_version=int(data["schema_version"]),
            mode=Mode(data["mode"]),
            findings=tuple(Finding.from_dict(f) for f in data["findings"]),
            chars_before=(
                None if data["chars_before"] is None else int(data["chars_before"])
            ),
            chars_after=int(data["chars_after"]),
            repaired=bool(data["repaired"]),
            block_checksum=data["block_checksum"],
            removed=tuple((str(i), str(r)) for i, r in data["removed"]),
            duration_ms=float(data["duration_ms"]),
            note=data["note"],
        )

    def to_json(self) -> str:
        """One JSONL-ready line, stable key order, no trailing newline.

        Key order is declaration order, fixed above; separators are compact
        so a run log stays greppable and small. Non-ASCII text is emitted
        raw (UTF-8) rather than escaped, because invariant text and evidence
        spans should read in a log the way they read in context.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)
