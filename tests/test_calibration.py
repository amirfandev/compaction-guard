"""Per-extra calibration: committed expected verdicts and scores against the
pinned model revisions, per spec section 8.

Every expectation in ``tests/fixtures/calibration/`` was produced by running
the shipped detector against the pinned weights; nothing here is a guessed
number. The suite runs offline (the socket-refusing fixture applies), so the
models must already sit in the local Hugging Face cache: CI warms the cache
in a separate step before pytest, and ``HF_HUB_OFFLINE`` keeps the hub
client from touching the network during the tests. When the extra is not
installed or the weights are absent, every case skips with the detector's
own reason string, cleanly, never silently.

Scores are asserted with a small absolute tolerance because ONNX and BLAS
builds differ across platforms in late decimals; a verdict flip is a real
failure, a fourth-decimal wobble is not.
"""

from __future__ import annotations

from typing import Any

import pytest

from compaction_guard.detectors.base import Detector, SummaryView
from compaction_guard.invariant import Invariant
from conftest import load_fixture_cases

SCORE_TOLERANCE = 0.05

EMBEDDING_CASES = load_fixture_cases("calibration/embedding")
NLI_CASES = load_fixture_cases("calibration/nli")


@pytest.fixture(autouse=True)
def hub_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the hub client to the local cache before it is first imported."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def embedding_detector() -> Detector:
    from compaction_guard.detectors.embedding import EmbeddingDetector

    detector = EmbeddingDetector()
    _probe_or_skip(detector)
    return detector


@pytest.fixture(scope="module")
def nli_detector() -> Detector:
    from compaction_guard.detectors.nli import NLIDetector

    detector = NLIDetector()
    _probe_or_skip(detector)
    return detector


def _probe_or_skip(detector: Any) -> None:
    detector.examine(
        Invariant.parse("The probe budget cap is $1."),
        SummaryView.from_summary("The probe budget cap is $1."),
    )
    reason = getattr(detector, "unavailable", None)
    if reason:
        pytest.skip(reason)


def _assert_case(detector: Detector, case: dict[str, Any]) -> None:
    invariant = Invariant.parse(case["invariant"])
    view = SummaryView.from_summary(case["summary"])
    verdict = detector.examine(invariant, view)
    if case["expected"] == "escalate":
        assert verdict is None, f"expected escalation, got {verdict}"
        return
    assert verdict is not None, "expected a verdict, layer escalated"
    assert verdict.kind.value == case["expected"], verdict.evidence
    expected_score = case["expected_score"]
    if expected_score is not None:
        assert verdict.score is not None
        assert abs(verdict.score - expected_score) <= SCORE_TOLERANCE, (
            f"score {verdict.score} drifted from committed {expected_score} "
            f"beyond {SCORE_TOLERANCE}; the pinned revision should be "
            "deterministic, so investigate before updating the fixture"
        )


@pytest.mark.parametrize(
    ("rel_path", "case"), EMBEDDING_CASES, ids=[c[0] for c in EMBEDDING_CASES]
)
def test_embedding_calibration(
    embedding_detector: Detector, rel_path: str, case: dict[str, Any]
) -> None:
    _assert_case(embedding_detector, case)


@pytest.mark.parametrize(("rel_path", "case"), NLI_CASES, ids=[c[0] for c in NLI_CASES])
def test_nli_calibration(
    nli_detector: Detector, rel_path: str, case: dict[str, Any]
) -> None:
    _assert_case(nli_detector, case)


def test_negation_case_is_present() -> None:
    """Spec section 8: the negation pair must exist and must show the
    embedding layer refusing to certify. Fixture presence is asserted even
    where the extras are absent, so deleting the case fails core CI."""
    names = [path for path, _ in EMBEDDING_CASES]
    assert any("negation" in name for name in names)
    negation = [case for path, case in EMBEDDING_CASES if "negation" in path]
    assert all(case["expected"] == "escalate" for case in negation)
