"""The labeled verdict fixtures: one directory per Kind, four or more cases each.

Every case pins the lexical tier's exact verdict and decided_by, including
the tier's documented honest failures: paraphrase and contradiction cases
carry a ground-truth label from their directory and an expected lexical
verdict that is deliberately weaker (UNVERIFIABLE, DROPPED, or a damage
verdict), never a certification. The false-certify assertion at the bottom is
the suite's copy of the release gate: no case whose ground truth is damage
may ever be reported PRESERVED or PARAPHRASED.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from compaction_guard.check import check
from compaction_guard.invariant import Invariant
from compaction_guard.taxonomy import Kind
from conftest import FIXTURES, load_fixture_cases

CASES = load_fixture_cases("verdicts")

_DAMAGE_LABELS = {"mutated", "contradicted", "dropped"}
_CERTIFYING = {Kind.PRESERVED, Kind.PARAPHRASED}


def test_fixture_coverage() -> None:
    """Every Kind directory exists and carries at least four cases."""
    root = FIXTURES / "verdicts"
    for kind in Kind:
        cases = list((root / kind.value).glob("*.json"))
        assert len(cases) >= 4, f"kind {kind.value} has {len(cases)} fixture cases, need 4"


@pytest.mark.parametrize(("rel_path", "case"), CASES, ids=[c[0] for c in CASES])
def test_lexical_tier_verdict(rel_path: str, case: dict[str, Any]) -> None:
    assert case["label"] == Path(rel_path).parent.name, "label must match its directory"
    invariant = Invariant.parse(case["invariant"])
    report = check([invariant], case["summary"])
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.kind is Kind(case["expected_kind"]), finding.evidence
    assert finding.decided_by == case["expected_decided_by"], finding.evidence


_DAMAGE_CASES = [c for c in CASES if c[1]["label"] in _DAMAGE_LABELS]


@pytest.mark.parametrize(("rel_path", "case"), _DAMAGE_CASES, ids=[c[0] for c in _DAMAGE_CASES])
def test_no_false_certify(rel_path: str, case: dict[str, Any]) -> None:
    """The release gate in miniature: damage is never certified as survival."""
    invariant = Invariant.parse(case["invariant"])
    report = check([invariant], case["summary"])
    assert report.findings[0].kind not in _CERTIFYING, (
        f"false certify: ground truth {case['label']} reported as "
        f"{report.findings[0].kind.value}"
    )


@pytest.mark.parametrize(("rel_path", "case"), CASES, ids=[c[0] for c in CASES])
def test_verdicts_deterministic(rel_path: str, case: dict[str, Any]) -> None:
    """Same inputs, same finding, byte for byte; durations aside, the report

    is the library's only output and it must not wobble between runs.
    """
    invariant = Invariant.parse(case["invariant"])
    first = check([invariant], case["summary"]).findings
    second = check([invariant], case["summary"]).findings
    assert first == second
