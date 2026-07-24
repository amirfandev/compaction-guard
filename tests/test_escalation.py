"""The escalation matrix: one test per cell, and the chain that enforces it.

The matrix is the whole soundness argument for layered detection, so it gets
cell-by-cell pinning: which kinds each named layer may issue, how each
permitted verdict is attributed, construction-time rejection of a too-wide
detector, and runtime rejection of a verdict outside a detector's own
declaration. Custom (unnamed) detectors are covered too: they escape the
matrix but never their own whitelist.
"""

from __future__ import annotations

import pytest

from compaction_guard.detectors.base import (
    ESCALATION_MATRIX,
    DetectorChain,
    LayerVerdict,
    SummaryView,
    SurvivalSite,
)
from compaction_guard.errors import CompactionGuardError
from compaction_guard.guard import Guard
from compaction_guard.invariant import Invariant
from compaction_guard.taxonomy import Kind

INV = Invariant.parse("The budget cap for this run is $500.")
VIEW = SummaryView.from_summary("Some unrelated summary text.")

# The spec's table, restated independently so a drift in base.py fails here.
SPEC_CAN_ISSUE: dict[str, frozenset[Kind]] = {
    "lexical": frozenset({Kind.PRESERVED, Kind.MUTATED, Kind.WEAKENED, Kind.DROPPED}),
    "embedding": frozenset({Kind.DROPPED}),
    "nli": frozenset({Kind.PARAPHRASED, Kind.WEAKENED, Kind.CONTRADICTED}),
    "judge": frozenset(
        {Kind.PARAPHRASED, Kind.WEAKENED, Kind.CONTRADICTED, Kind.DROPPED, Kind.UNVERIFIABLE}
    ),
}

SPEC_DECIDED_BY: dict[tuple[str, Kind], str] = {
    ("lexical", Kind.PRESERVED): "lexical.exact",
    ("lexical", Kind.MUTATED): "lexical.anchor_diff",
    ("lexical", Kind.WEAKENED): "lexical.modality",
    ("lexical", Kind.DROPPED): "lexical.miss",
    ("embedding", Kind.DROPPED): "embedding.floor",
    ("nli", Kind.PARAPHRASED): "nli.bidirectional",
    ("nli", Kind.WEAKENED): "nli.bidirectional",
    ("nli", Kind.CONTRADICTED): "nli.bidirectional",
    ("judge", Kind.PARAPHRASED): "judge",
    ("judge", Kind.WEAKENED): "judge",
    ("judge", Kind.CONTRADICTED): "judge",
    ("judge", Kind.DROPPED): "judge",
    ("judge", Kind.UNVERIFIABLE): "judge",
}


class ScriptedDetector:
    """A detector that says whatever the test tells it to."""

    def __init__(self, name: str, can_issue: frozenset[Kind], verdict: LayerVerdict | None) -> None:
        self.name = name
        self.can_issue = can_issue
        self._verdict = verdict

    def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
        return self._verdict


@pytest.mark.parametrize("name", sorted(SPEC_CAN_ISSUE))
@pytest.mark.parametrize("kind", list(Kind), ids=[k.value for k in Kind])
def test_matrix_cell(name: str, kind: Kind) -> None:
    """Every (layer, kind) cell of the matrix matches the spec table exactly.

    Cells outside a row are the executable form of that layer's blind spot;
    a cell silently flipping would let a layer certify what it cannot see.
    """
    assert (kind in ESCALATION_MATRIX[name].can_issue) == (kind in SPEC_CAN_ISSUE[name])


@pytest.mark.parametrize(
    ("name", "kind", "expected"),
    [(name, kind, by) for (name, kind), by in SPEC_DECIDED_BY.items()],
    ids=[f"{name}-{kind.value}" for (name, kind) in SPEC_DECIDED_BY],
)
def test_decided_by_label_per_cell(name: str, kind: Kind, expected: str) -> None:
    assert ESCALATION_MATRIX[name].decided_by[kind] == expected


@pytest.mark.parametrize("name", sorted(SPEC_CAN_ISSUE))
def test_chain_rejects_widened_named_detector(name: str) -> None:
    """A detector reusing a matrix name may not widen that row by one kind."""
    forbidden = sorted(set(Kind) - SPEC_CAN_ISSUE[name], key=lambda k: k.value)[0]
    widened = ScriptedDetector(name, SPEC_CAN_ISSUE[name] | {forbidden}, None)
    with pytest.raises(CompactionGuardError, match="outside the escalation matrix"):
        DetectorChain((widened,))


def test_chain_accepts_exact_named_rows() -> None:
    detectors = tuple(
        ScriptedDetector(name, SPEC_CAN_ISSUE[name], None) for name in sorted(SPEC_CAN_ISSUE)
    )
    chain = DetectorChain(detectors)
    assert chain.detectors == detectors


def test_chain_rejects_runtime_verdict_outside_declaration() -> None:
    """A verdict outside the detector's own can_issue is a harness bug and raises,

    never a finding: laundering it into a report would hide a broken layer.
    """
    rogue = ScriptedDetector(
        "rogue",
        frozenset({Kind.DROPPED}),
        LayerVerdict(kind=Kind.PRESERVED, evidence="trust me"),
    )
    chain = DetectorChain((rogue,))
    with pytest.raises(CompactionGuardError, match="outside its can_issue"):
        chain.examine(INV, VIEW)


def test_named_detector_cannot_issue_forbidden_kind_at_runtime() -> None:
    """An embedding-named detector declaring DROPPED but answering PRESERVED

    is stopped by the runtime whitelist even though construction passed.
    """

    class LyingEmbedding(ScriptedDetector):
        def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
            return LayerVerdict(kind=Kind.PRESERVED, evidence="cosine says so", score=0.99)

    chain = DetectorChain((LyingEmbedding("embedding", frozenset({Kind.DROPPED}), None),))
    with pytest.raises(CompactionGuardError, match="outside its can_issue"):
        chain.examine(INV, VIEW)


def test_guard_validates_detectors_at_construction() -> None:
    """A bad detector list fails at Guard(), not at the first compaction."""
    widened = ScriptedDetector("embedding", frozenset({Kind.DROPPED, Kind.PRESERVED}), None)
    with pytest.raises(CompactionGuardError, match="escalation matrix"):
        Guard(["some constraint"], detectors=[widened])


def test_custom_detector_within_declaration_is_honoured() -> None:
    """Unnamed detectors escape the matrix but keep their own whitelist,

    and their findings are attributed to the detector's name.
    """
    custom = ScriptedDetector(
        "custom",
        frozenset({Kind.CONTRADICTED}),
        LayerVerdict(kind=Kind.CONTRADICTED, evidence="scripted", site=SurvivalSite.SUMMARY),
    )
    finding = DetectorChain((custom,)).examine(INV, VIEW)
    assert finding.kind is Kind.CONTRADICTED
    assert finding.decided_by == "custom"
    # CONTRADICTED is not a survival kind: the site is nulled, at_risk stays False.
    assert finding.survived_in is None
    assert finding.at_risk is False


def test_survival_site_copied_and_at_risk_computed() -> None:
    weak = ScriptedDetector(
        "custom",
        frozenset({Kind.WEAKENED}),
        LayerVerdict(kind=Kind.WEAKENED, evidence="scripted", site=SurvivalSite.RETAINED_TAIL),
    )
    finding = DetectorChain((weak,)).examine(INV, VIEW)
    assert finding.survived_in is SurvivalSite.RETAINED_TAIL
    assert finding.at_risk is True


def test_short_circuit_makes_first_verdict_final() -> None:
    """NLI may not override a lexical MUTATED because MUTATED already ended

    the chain; the second detector must never be consulted.
    """
    consulted: list[str] = []

    class Recording(ScriptedDetector):
        def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
            consulted.append(self.name)
            return self._verdict

    first = Recording("first", frozenset({Kind.MUTATED}), LayerVerdict(kind=Kind.MUTATED, evidence="anchor diff"))
    second = Recording("second", frozenset({Kind.PARAPHRASED}), LayerVerdict(kind=Kind.PARAPHRASED, evidence="never reached"))
    finding = DetectorChain((first, second)).examine(INV, VIEW)
    assert finding.kind is Kind.MUTATED
    assert consulted == ["first"]


def test_conclude_offered_only_to_last_detector() -> None:
    calls: list[str] = []

    class Concluder(ScriptedDetector):
        def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
            return None

        def conclude(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
            calls.append(self.name)
            return LayerVerdict(kind=Kind.DROPPED, evidence="terminal")

    inner = Concluder("inner", frozenset({Kind.DROPPED}), None)
    outer = Concluder("outer", frozenset({Kind.DROPPED}), None)
    finding = DetectorChain((inner, outer)).examine(INV, VIEW)
    assert finding.kind is Kind.DROPPED
    assert calls == ["outer"], "only the terminal layer gets the second call"


def test_empty_chain_exhausts_to_unverifiable() -> None:
    finding = DetectorChain(()).examine(INV, VIEW)
    assert finding.kind is Kind.UNVERIFIABLE
    assert finding.decided_by == "chain.exhausted"
    assert "layers exhausted: none" in finding.evidence
