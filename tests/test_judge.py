"""The judge detector: an untrusted callable behind a checkable contract.

A FixtureJudge replays taught prompt-digest responses and fails loudly on
any prompt it was never taught, so no test can silently depend on a live
model. The cases pin the whole failure ladder: fabricated spans, forbidden
verdicts, drop claims that cite text, unparseable replies, and raising
callables all degrade to UNVERIFIABLE; only verified spans produce verdicts.
"""

from __future__ import annotations

import hashlib
import json

from compaction_guard.detectors.base import DetectorChain, SummaryView, SurvivalSite
from compaction_guard.detectors.judge import JudgeDetector, build_prompt
from compaction_guard.invariant import Invariant
from compaction_guard.taxonomy import Kind

INV_BUDGET = Invariant.parse("The budget cap for this run is $500.")
SUMMARY = "Try to keep spending somewhere near $500 for this session."
VIEW = SummaryView.from_summary(SUMMARY)


class FixtureJudge:
    """Replays committed prompt-digest responses; unknown prompts fail the test."""

    def __init__(self) -> None:
        self._responses: dict[str, str] = {}

    @staticmethod
    def _digest(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def teach(self, prompt: str, response: str) -> None:
        self._responses[self._digest(prompt)] = response

    def __call__(self, prompt: str) -> str:
        digest = self._digest(prompt)
        if digest not in self._responses:
            raise AssertionError(
                "FixtureJudge got a prompt it was never taught; a test would "
                "otherwise need a live model here"
            )
        return self._responses[digest]


def _examine(invariant: Invariant, view: SummaryView, response: str) -> object:
    judge = FixtureJudge()
    judge.teach(build_prompt(invariant.text, view.full_text), response)
    return DetectorChain((JudgeDetector(judge),)).examine(invariant, view)


def _reply(span: str | None, verdict: str, reason: str = "because") -> str:
    return json.dumps({"span": span, "verdict": verdict, "reason": reason})


def test_weakened_with_verified_span() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply(SUMMARY, "weakened"))
    assert finding.kind is Kind.WEAKENED
    assert finding.decided_by == "judge"
    assert finding.survived_in is SurvivalSite.SUMMARY


def test_paraphrased_requires_anchors_inside_the_span() -> None:
    view = SummaryView.from_summary("Spending is capped at $500 for this run. Other notes.")
    span = "Spending is capped at $500 for this run."
    finding = _examine(INV_BUDGET, view, _reply(span, "paraphrased"))
    assert finding.kind is Kind.PARAPHRASED


def test_paraphrased_span_missing_the_value_degrades() -> None:
    """A paraphrase that lost the number is not a paraphrase, whatever the judge thinks."""
    view = SummaryView.from_summary("Spending is capped sensibly for this run. The cap is generous.")
    span = "Spending is capped sensibly for this run."
    finding = _examine(INV_BUDGET, view, _reply(span, "paraphrased"))
    assert finding.kind is Kind.UNVERIFIABLE
    assert "lacks bound anchors" in finding.evidence


def test_fabricated_span_degrades_to_unverifiable() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply("The cap was removed entirely.", "weakened"))
    assert finding.kind is Kind.UNVERIFIABLE
    assert "failed re-verification" in finding.evidence


def test_near_miss_span_fails_token_boundary_reverification() -> None:
    """$500 inside $5000 must not re-verify: whole tokens only, like every layer."""
    view = SummaryView.from_summary("The budget cap for this run is $5000.")
    span = "The budget cap for this run is $500"
    finding = _examine(INV_BUDGET, view, _reply(span, "weakened"))
    assert finding.kind is Kind.UNVERIFIABLE
    assert "failed re-verification" in finding.evidence


def test_forbidden_verdict_preserved_degrades() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply(SUMMARY, "preserved"))
    assert finding.kind is Kind.UNVERIFIABLE
    assert "not one of the forced choices" in finding.evidence


def test_forbidden_verdict_mutated_degrades() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply(SUMMARY, "mutated"))
    assert finding.kind is Kind.UNVERIFIABLE


def test_dropped_with_cited_span_degrades() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply(SUMMARY, "dropped"))
    assert finding.kind is Kind.UNVERIFIABLE
    assert "must cite nothing" in finding.evidence


def test_dropped_with_null_span_is_honoured() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply(None, "dropped", "no trace anywhere"))
    assert finding.kind is Kind.DROPPED
    assert finding.decided_by == "judge"


def test_contradicted_with_verified_span() -> None:
    view = SummaryView.from_summary("There is no spending limit at all for this run.")
    span = "There is no spending limit at all for this run."
    finding = _examine(INV_BUDGET, view, _reply(span, "contradicted"))
    assert finding.kind is Kind.CONTRADICTED


def test_unparseable_reply_degrades() -> None:
    finding = _examine(INV_BUDGET, VIEW, "I think the constraint is fine, mostly.")
    assert finding.kind is Kind.UNVERIFIABLE
    assert "no parseable JSON" in finding.evidence


def test_fenced_json_is_tolerated() -> None:
    fenced = "Here is my analysis:\n```json\n" + _reply(SUMMARY, "weakened") + "\n```"
    finding = _examine(INV_BUDGET, VIEW, fenced)
    assert finding.kind is Kind.WEAKENED


def test_judge_abstention_degrades_with_reason() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply(None, "unverifiable", "cannot tell"))
    assert finding.kind is Kind.UNVERIFIABLE
    assert "cannot tell" in finding.evidence


def test_raising_judge_degrades_instead_of_crashing() -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("connection reset")

    finding = DetectorChain((JudgeDetector(broken),)).examine(INV_BUDGET, VIEW)
    assert finding.kind is Kind.UNVERIFIABLE
    assert "judge callable raised RuntimeError" in finding.evidence


def test_survival_verdict_without_span_degrades() -> None:
    finding = _examine(INV_BUDGET, VIEW, _reply(None, "weakened"))
    assert finding.kind is Kind.UNVERIFIABLE
    assert "requires a cited span" in finding.evidence
