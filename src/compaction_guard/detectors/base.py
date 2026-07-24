"""The detector contract, the escalation matrix, and the chain that enforces both.

Every layer past lexical has a known blind spot: embeddings score negations as
near-identical, NLI entails $5000 from $500, judges acquiesce and drift between
runs. A chain that let any layer say anything would launder those blind spots
into false PRESERVED verdicts, the one failure this library must not have. So
the soundness argument is data, not discipline: ``ESCALATION_MATRIX`` records
what each named layer is allowed to claim, ``DetectorChain`` refuses anything
outside that whitelist, and every verdict short-circuits the chain, which is
what makes "NLI may not override a lexical MUTATED" structural rather than a
convention someone has to remember.

The chain's escalation logic, per invariant:

1. Run detectors in the order given, cheap to expensive, against the
   inspectable view: the chain strips reassertion-block sentences before any
   detector sees the text, so the guard's own carried block can neither
   certify survival nor mask damage. A detector returns a ``LayerVerdict``
   to stop the chain, or ``None`` to escalate.
2. A returned verdict must be inside the detector's ``can_issue`` whitelist;
   anything else raises ``CompactionGuardError``, because a layer exceeding its
   declared competence is a harness bug, not a summariser behaviour.
3. The final detector, and only the final one, is offered a second call:
   ``conclude()``, if it defines one. This is how the spec's terminal rule
   ("a lexical miss with no further layers installed is DROPPED") is expressed
   without the detector knowing what chain it sits in. With later layers
   present, the same miss escalates instead, because those layers exist
   precisely to tell paraphrase from absence.
4. Exhaustion is UNVERIFIABLE, decided by ``chain.exhausted``. It is a
   first-class verdict: the layers ran out of soundness, and saying so beats
   guessing. Detectors that could not load their optional extra surface the
   reason here through their ``unavailable`` attribute, so a bare-install user
   reading the report learns which extra would have answered.
5. Block echo, the chain's own final rule: when the outcome would otherwise
   be DROPPED or UNVERIFIABLE and the invariant's text sits verbatim in a
   carried sentinel block, the finding is PRESERVED with
   ``survived_in=reassertion_block`` and ``decided_by=chain.block_echo``.
   The text really is present in context, so DROPPED would be false; but a
   positive damage verdict from the summary is never overridden by the
   guard's own echo.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..errors import CompactionGuardError
from ..normalize import normalize
from ..report import Finding
from ..taxonomy import Kind

if TYPE_CHECKING:
    from ..invariant import Invariant

__all__ = [
    "BLOCK_ECHO_DECIDED_BY",
    "ESCALATION_MATRIX",
    "SITE_PREFERENCE",
    "SURVIVAL_KINDS",
    "Detector",
    "DetectorChain",
    "DetectorRule",
    "JudgeFn",
    "LayerVerdict",
    "Sentence",
    "SummaryView",
    "SurvivalSite",
    "clip",
    "contains_tokens",
    "inspectable_view",
    "site_texts",
    "split_sentences",
]


JudgeFn = Callable[[str], str]
"""Prompt in, raw model text out. The only model surface in the package.

The judge detector is constructed with one of these; the package ships no
HTTP client and imports no provider SDK, so the caller's closure is the entire
integration. This is also what makes judge tests trivial: inject a lambda.
"""


class SurvivalSite(StrEnum):
    """Where in the post-compaction context a constraint's text was found.

    PRESERVED without a site would lie by omission: text that survived only in
    the kept-verbatim tail is one compaction from death, and text found only
    in a stale sentinel block was carried by the guard, not the summariser.
    """

    SUMMARY = "summary"
    """Inside the inserted summary region."""

    RETAINED_TAIL = "retained_tail"
    """In messages the compactor kept verbatim."""

    REASSERTION_BLOCK = "reassertion_block"
    """Only in a prior sentinel block."""


SITE_PREFERENCE: tuple[SurvivalSite, ...] = (
    SurvivalSite.SUMMARY,
    SurvivalSite.RETAINED_TAIL,
    SurvivalSite.REASSERTION_BLOCK,
)
"""Order in which detectors credit survival when text appears at several sites.

Summary survival is the durable kind: the compactor chose to carry the words
forward. The retained tail survives only until the next compaction evicts it.
A stale reassertion block is the guard's own earlier write and detectors never
credit it at all: the chain hands every detector a view with block sentences
removed, and block survival is reported only by the chain's own echo rule,
after the summary and tail have been examined on their own evidence.
Crediting the block "last" was not enough; a carried block still stopped the
chain before damage in the summary was seen, which blinded every compaction
after the first repair.
"""


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence of the post-compaction context, tagged with its region."""

    text: str
    normalized: str
    site: SurvivalSite

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "normalized": self.normalized, "site": self.site.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Sentence:
        return cls(
            text=str(data["text"]),
            normalized=str(data["normalized"]),
            site=SurvivalSite(data["site"]),
        )


@dataclass(frozen=True, slots=True)
class SummaryView:
    """The post-compaction text, computed once and shared across invariants and layers.

    ``diff.py`` builds this in OWNED mode from its region attribution;
    ``check()`` builds it from bare summary text. Detectors never re-derive
    these fields, because two layers normalising or splitting differently
    would make their verdicts incomparable.
    """

    full_text: str
    normalized: str
    sentences: tuple[Sentence, ...]
    token_set: frozenset[str]

    @classmethod
    def from_regions(cls, regions: Iterable[tuple[str, SurvivalSite]]) -> SummaryView:
        """Build a view from (text, site) regions, in the order given.

        Lives here rather than in the builders so the field semantics have
        exactly one definition next to their consumers: ``normalized`` is
        ``normalize(full_text)``, ``token_set`` is its whitespace split, and
        ``sentences`` cover every region with each sentence carrying its
        region's site. Empty regions are skipped; they carry no evidence and
        would only produce empty sentences.
        """
        kept = [(text, site) for text, site in regions if text.strip()]
        sentences: list[Sentence] = []
        for text, site in kept:
            for sent in split_sentences(text):
                sentences.append(Sentence(text=sent, normalized=normalize(sent), site=site))
        full_text = "\n".join(text for text, _site in kept)
        normalized = normalize(full_text)
        return cls(
            full_text=full_text,
            normalized=normalized,
            sentences=tuple(sentences),
            token_set=frozenset(normalized.split()),
        )

    @classmethod
    def from_summary(cls, summary_text: str) -> SummaryView:
        """Build a view for bare summary text, everything attributed to SUMMARY.

        The REASSERTED path: ``check()`` receives only the summary side, so
        there is no tail or block region to attribute and no before-side to
        diff against.
        """
        return cls.from_regions([(summary_text, SurvivalSite.SUMMARY)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_text": self.full_text,
            "normalized": self.normalized,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "token_set": sorted(self.token_set),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SummaryView:
        return cls(
            full_text=str(data["full_text"]),
            normalized=str(data["normalized"]),
            sentences=tuple(Sentence.from_dict(s) for s in data["sentences"]),
            token_set=frozenset(data["token_set"]),
        )


@dataclass(frozen=True, slots=True)
class LayerVerdict:
    """One layer's answer for one invariant, before the chain turns it into a Finding.

    ``evidence`` must be recomputable from the inputs: a matched span, the
    missing anchors, or a score, never free prose. ``site`` is set when the
    verdict rests on text found at a particular region; the chain copies it
    into ``Finding.survived_in`` only for survival kinds.
    """

    kind: Kind
    evidence: str
    score: float | None = None
    site: SurvivalSite | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "evidence": self.evidence,
            "score": self.score,
            "site": None if self.site is None else self.site.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LayerVerdict:
        return cls(
            kind=Kind(data["kind"]),
            evidence=str(data["evidence"]),
            score=None if data["score"] is None else float(data["score"]),
            site=None if data["site"] is None else SurvivalSite(data["site"]),
        )


class Detector(Protocol):
    """One detection layer. Return a verdict to stop the chain, None to escalate.

    Two optional extensions, both duck-typed so the protocol stays minimal:

    - ``conclude(invariant, view)``: called by the chain only when this
      detector is the last layer and ``examine`` returned None. It may issue a
      terminal verdict that would be unsound with more layers installed (the
      lexical detector's complete-miss DROPPED is the one shipped example).
    - ``unavailable``: a string attribute naming why the layer cannot run
      (typically a missing extra). The chain quotes it in the exhaustion
      verdict's evidence so degradation is visible in the report, not silent.
    """

    name: str
    can_issue: frozenset[Kind]

    def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None: ...


@dataclass(frozen=True, slots=True)
class DetectorRule:
    """One row of the escalation matrix: what a named layer may claim, and how
    its verdicts are labeled.

    ``decided_by`` maps each permitted kind to the ``Finding.decided_by``
    string, so verdict attribution is table data with one fixture test per
    cell rather than string formatting scattered through detector code.
    """

    can_issue: frozenset[Kind]
    decided_by: Mapping[Kind, str]


ESCALATION_MATRIX: Mapping[str, DetectorRule] = {
    "lexical": DetectorRule(
        can_issue=frozenset(
            {Kind.PRESERVED, Kind.MUTATED, Kind.WEAKENED, Kind.DROPPED}
        ),
        decided_by={
            Kind.PRESERVED: "lexical.exact",
            Kind.MUTATED: "lexical.anchor_diff",
            Kind.WEAKENED: "lexical.modality",
            Kind.DROPPED: "lexical.miss",
        },
    ),
    "embedding": DetectorRule(
        can_issue=frozenset({Kind.DROPPED}),
        decided_by={Kind.DROPPED: "embedding.floor"},
    ),
    "nli": DetectorRule(
        can_issue=frozenset({Kind.PARAPHRASED, Kind.WEAKENED, Kind.CONTRADICTED}),
        decided_by={
            Kind.PARAPHRASED: "nli.bidirectional",
            Kind.WEAKENED: "nli.bidirectional",
            Kind.CONTRADICTED: "nli.bidirectional",
        },
    ),
    "judge": DetectorRule(
        can_issue=frozenset(
            {
                Kind.PARAPHRASED,
                Kind.WEAKENED,
                Kind.CONTRADICTED,
                Kind.DROPPED,
                Kind.UNVERIFIABLE,
            }
        ),
        decided_by={
            Kind.PARAPHRASED: "judge",
            Kind.WEAKENED: "judge",
            Kind.CONTRADICTED: "judge",
            Kind.DROPPED: "judge",
            Kind.UNVERIFIABLE: "judge",
        },
    ),
}
"""The blind spots of each layer, as executable restrictions.

Lexical cannot see paraphrase or contradiction, so it may not name them.
Cosine similarity is negation-blind, so the embedding layer may only confirm
absence, never certify presence. NLI is numerically insensitive, so MUTATED
is out of its reach (and out of PRESERVED's, which only deterministic
comparison earns). The judge may not claim PRESERVED or MUTATED either:
verbatim survival and value changes are decided by layers that recompute, not
by a model's say-so. A detector under one of these names must declare a
``can_issue`` no wider than its row; ``DetectorChain`` refuses it otherwise.
"""


SURVIVAL_KINDS: frozenset[Kind] = frozenset(
    {Kind.PRESERVED, Kind.PARAPHRASED, Kind.WEAKENED}
)
"""Kinds for which ``Finding.survived_in`` is meaningful: some text survived
somewhere. MUTATED, CONTRADICTED, DROPPED and UNVERIFIABLE assert damage or
ignorance, not location, so the chain nulls the site for them."""

_EXHAUSTED_DECIDED_BY = "chain.exhausted"

BLOCK_ECHO_DECIDED_BY = "chain.block_echo"
"""Attribution for the chain's own rule: text alive only in a prior sentinel
block, credited only after every installed layer failed to find the
constraint or its damage in the summariser's actual output."""


@lru_cache(maxsize=8)
def inspectable_view(view: SummaryView) -> SummaryView:
    """The view with reassertion-block sentences removed.

    This is what every detector examines. Block text is the guard's own
    earlier write: letting it satisfy containment, entailment, or anchor
    survival is how a compactor that keeps the tail (the commonest wrapping
    pattern) makes every later compaction read as PRESERVED, killing drift
    detection and the RAISE gate from the second compaction onward. Views
    with no block sentences come back identical, so the common path pays
    one membership scan.
    """
    if not any(s.site is SurvivalSite.REASSERTION_BLOCK for s in view.sentences):
        return view
    return SummaryView.from_regions(
        (s.text, s.site)
        for s in view.sentences
        if s.site is not SurvivalSite.REASSERTION_BLOCK
    )


@lru_cache(maxsize=8)
def _block_normalized(view: SummaryView) -> str:
    """Normalised text of the view's reassertion-block sentences, joined."""
    return " ".join(
        s.normalized for s in view.sentences if s.site is SurvivalSite.REASSERTION_BLOCK
    )


class DetectorChain:
    """Runs detectors cheap to expensive, enforcing the escalation matrix.

    Stateless after construction and safe to share: all per-compaction state
    lives in the ``SummaryView`` argument. ``examine`` returns a full
    ``Finding`` rather than a bare verdict because the chain is the only
    place that knows both the deciding detector and the invariant, and
    splitting the attribution across callers is how ``decided_by`` fields
    drift.
    """

    def __init__(self, detectors: Sequence[Detector]) -> None:
        self._detectors: tuple[Detector, ...] = tuple(detectors)
        for detector in self._detectors:
            rule = ESCALATION_MATRIX.get(detector.name)
            if rule is not None and not detector.can_issue <= rule.can_issue:
                excess = ", ".join(
                    sorted(kind.value for kind in detector.can_issue - rule.can_issue)
                )
                raise CompactionGuardError(
                    f"detector {detector.name!r} declares kinds outside the escalation "
                    f"matrix row for that name: {excess}. The matrix encodes the layer's "
                    "known blind spots; widening it is an edit to detectors/base.py with "
                    "a fixture, not a per-detector declaration."
                )

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return self._detectors

    def examine(self, invariant: Invariant, view: SummaryView) -> Finding:
        """Decide one invariant against one view. Always returns a Finding.

        Detectors see only the inspectable view (block sentences removed).
        The chain's block-echo rule runs last and only where it is sound: a
        verdict of DROPPED or UNVERIFIABLE asserts absence or ignorance, and
        verbatim presence of the invariant in a carried block refutes
        absence, so those two downgrade to PRESERVED-in-block. Positive
        damage verdicts (MUTATED, CONTRADICTED, WEAKENED, PARAPHRASED) stand
        untouched: they describe what the summariser wrote, and the guard's
        own echo is not evidence against that.
        """
        inspectable = inspectable_view(view)
        last = len(self._detectors) - 1
        for index, detector in enumerate(self._detectors):
            verdict = detector.examine(invariant, inspectable)
            if verdict is None and index == last:
                conclude = getattr(detector, "conclude", None)
                if callable(conclude):
                    verdict = cast(
                        "LayerVerdict | None", conclude(invariant, inspectable)
                    )
            if verdict is None:
                continue
            if verdict.kind not in detector.can_issue:
                raise CompactionGuardError(
                    f"detector {detector.name!r} issued {verdict.kind.value!r}, which is "
                    "outside its can_issue whitelist. This is a defect in the detector, "
                    "not summariser behaviour; refusing to launder it into a finding."
                )
            if verdict.kind in (Kind.DROPPED, Kind.UNVERIFIABLE):
                echo = self._block_echo(invariant, view)
                if echo is not None:
                    return echo
            return self._finding(invariant, detector, verdict)
        echo = self._block_echo(invariant, view)
        if echo is not None:
            return echo
        return self._exhausted(invariant)

    def examine_all(
        self, invariants: Sequence[Invariant], view: SummaryView
    ) -> tuple[Finding, ...]:
        """One finding per invariant, in registry order."""
        return tuple(self.examine(invariant, view) for invariant in invariants)

    def _finding(
        self, invariant: Invariant, detector: Detector, verdict: LayerVerdict
    ) -> Finding:
        rule = ESCALATION_MATRIX.get(detector.name)
        decided_by = detector.name
        if rule is not None:
            decided_by = rule.decided_by.get(verdict.kind, detector.name)
        survived_in = verdict.site if verdict.kind in SURVIVAL_KINDS else None
        return Finding(
            invariant_id=invariant.id,
            kind=verdict.kind,
            severity=invariant.severity,
            decided_by=decided_by,
            evidence=verdict.evidence,
            score=verdict.score,
            survived_in=survived_in,
            at_risk=survived_in is SurvivalSite.RETAINED_TAIL,
        )

    def _block_echo(self, invariant: Invariant, view: SummaryView) -> Finding | None:
        """PRESERVED-in-block, when the carried block is the only survival.

        Requires verbatim whole-token containment of the invariant's
        normalised text in the block-site sentences. The evidence string
        says what the verdict means: the guard carried the words, the
        summariser did not, and nothing stronger is being claimed.
        """
        block_norm = _block_normalized(view)
        if not block_norm:
            return None
        inv_norm = normalize(invariant.text)
        if not inv_norm or not contains_tokens(block_norm, inv_norm):
            return None
        return Finding(
            invariant_id=invariant.id,
            kind=Kind.PRESERVED,
            severity=invariant.severity,
            decided_by=BLOCK_ECHO_DECIDED_BY,
            evidence=(
                f'verbatim only in a prior reassertion block: "{clip(inv_norm, 120)}"; '
                "carried by the guard, not preserved by the summariser"
            ),
            score=1.0,
            survived_in=SurvivalSite.REASSERTION_BLOCK,
            at_risk=False,
        )

    def _exhausted(self, invariant: Invariant) -> Finding:
        names = ", ".join(detector.name for detector in self._detectors) or "none"
        notes: list[str] = []
        for detector in self._detectors:
            reason = getattr(detector, "unavailable", None)
            if reason:
                notes.append(f"{detector.name} unavailable: {reason}")
        evidence = f"layers exhausted: {names}"
        if notes:
            evidence = evidence + "; " + "; ".join(notes)
        return Finding(
            invariant_id=invariant.id,
            kind=Kind.UNVERIFIABLE,
            severity=invariant.severity,
            decided_by=_EXHAUSTED_DECIDED_BY,
            evidence=evidence,
            score=None,
            survived_in=None,
            at_risk=False,
        )


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> tuple[str, ...]:
    """Split text into sentences on terminal punctuation and newlines.

    Deliberately crude and deterministic: whitespace after ``.``, ``!`` or
    ``?``, or any newline run, ends a sentence. Abbreviations ("e.g. foo")
    over-split, which costs a little window quality in the NLI layer and
    nothing in correctness; a smarter splitter would buy accuracy at the
    price of a dependency or a pile of locale rules, and every layer above
    this depends on the split being identical everywhere, forever.
    """
    return tuple(part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part and part.strip())


def site_texts(view: SummaryView) -> tuple[tuple[SurvivalSite, str], ...]:
    """The view's text grouped per site, in SITE_PREFERENCE order.

    Sentences of a site are rejoined with single spaces. That loses original
    whitespace, which is safe for every consumer here: containment checks run
    in normalize space where whitespace has already collapsed, and site
    attribution needs the words, not the layout.
    """
    grouped: dict[SurvivalSite, list[str]] = {}
    for sentence in view.sentences:
        grouped.setdefault(sentence.site, []).append(sentence.text)
    return tuple(
        (site, " ".join(grouped[site])) for site in SITE_PREFERENCE if site in grouped
    )


def clip(text: str, limit: int) -> str:
    """Truncate evidence text deterministically, marking the cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def contains_tokens(haystack: str, needle: str) -> bool:
    """Whole-token containment for strings already in normalize space.

    Bare substring search is a false-certify bug wearing a convenience: "the
    cap is $500" is a character-level substring of "the cap is $5000", and a
    containment check that accepted it would certify PRESERVED on exactly the
    value mutation the anchor layer exists to catch. Padding both strings
    with spaces makes every match start and end on token boundaries, since
    normalize space separates tokens with single spaces and nothing else.
    An empty needle matches nothing: no text is evidence of no survival.
    """
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "
