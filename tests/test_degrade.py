"""Optional detectors degrading when their extra is absent.

Both model-backed layers must return None from examine, record why in
``unavailable``, and let the chain exhaust to UNVERIFIABLE with the missing
extra named in the evidence. The imports are force-blocked via sys.modules,
so these tests pin the degradation path even in an environment that happens
to have an extra installed.
"""

from __future__ import annotations

import sys

import pytest

from compaction_guard.detectors.base import DetectorChain, SummaryView
from compaction_guard.detectors.embedding import EmbeddingDetector
from compaction_guard.detectors.lexical import LexicalDetector
from compaction_guard.detectors.nli import NLIDetector
from compaction_guard.guard import Guard
from compaction_guard.invariant import Invariant
from compaction_guard.taxonomy import Kind
from stubs import INV_DB, DropAll, Message, base_messages

# A pair the lexical layer must escalate on (anchor present, topic gone), so
# the chain actually reaches the degraded layer.
ESCALATING_INVARIANT = Invariant.parse(INV_DB)
ESCALATING_VIEW = SummaryView.from_summary("orders_prod exists.")

_EXTRA_MODULES = {
    "embedding": ("huggingface_hub", "model2vec", "numpy"),
    "nli": ("onnxruntime", "tokenizers", "huggingface_hub"),
}


def _block_imports(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def test_embedding_degrades_to_unverifiable(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_imports(monkeypatch, _EXTRA_MODULES["embedding"])
    detector = EmbeddingDetector()
    chain = DetectorChain((LexicalDetector(), detector))
    finding = chain.examine(ESCALATING_INVARIANT, ESCALATING_VIEW)
    assert finding.kind is Kind.UNVERIFIABLE
    assert finding.decided_by == "chain.exhausted"
    assert detector.unavailable is not None
    assert "compaction-guard[embeddings]" in finding.evidence
    assert "embedding unavailable" in finding.evidence


def test_nli_degrades_to_unverifiable(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_imports(monkeypatch, _EXTRA_MODULES["nli"])
    detector = NLIDetector()
    chain = DetectorChain((LexicalDetector(), detector))
    finding = chain.examine(ESCALATING_INVARIANT, ESCALATING_VIEW)
    assert finding.kind is Kind.UNVERIFIABLE
    assert detector.unavailable is not None
    assert "compaction-guard[nli]" in finding.evidence
    assert "nli unavailable" in finding.evidence


def test_full_degraded_chain_names_every_missing_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_imports(monkeypatch, _EXTRA_MODULES["embedding"] + _EXTRA_MODULES["nli"])
    chain = DetectorChain((LexicalDetector(), EmbeddingDetector(), NLIDetector()))
    finding = chain.examine(ESCALATING_INVARIANT, ESCALATING_VIEW)
    assert finding.kind is Kind.UNVERIFIABLE
    assert "layers exhausted: lexical, embedding, nli" in finding.evidence
    assert "compaction-guard[embeddings]" in finding.evidence
    assert "compaction-guard[nli]" in finding.evidence


def test_degraded_layers_never_crash_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    """The degradation is invisible to the compaction path: repair proceeds."""
    _block_imports(monkeypatch, _EXTRA_MODULES["embedding"] + _EXTRA_MODULES["nli"])
    guard: Guard[list[Message]] = Guard(
        [INV_DB],
        detectors=[LexicalDetector(), EmbeddingDetector(), NLIDetector()],
    )
    result = guard.compact(base_messages(), DropAll())
    report = guard.last_report
    assert report is not None
    # With semantic layers nominally present but unavailable, the lexical
    # complete-miss stays escalation material and the chain exhausts.
    assert report.findings[0].kind is Kind.UNVERIFIABLE
    guard.assert_present(result)


def test_lexical_complete_miss_is_dropped_only_when_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same miss is DROPPED alone but UNVERIFIABLE with layers behind it,

    because those layers exist precisely to tell paraphrase from absence.
    """
    _block_imports(monkeypatch, _EXTRA_MODULES["embedding"])
    invariant = Invariant.parse("The budget cap for this run is $500.")
    view = SummaryView.from_summary("The team fixed the parser and shipped notes.")
    alone = DetectorChain((LexicalDetector(),)).examine(invariant, view)
    assert alone.kind is Kind.DROPPED
    assert alone.evidence == "lexical_only"
    followed = DetectorChain((LexicalDetector(), EmbeddingDetector())).examine(invariant, view)
    assert followed.kind is Kind.UNVERIFIABLE
