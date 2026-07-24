"""Guard behaviour against the adversarial compactors, per policy.

The heart of the suite. Every test asserts what the guard did with what the
compactor returned: block presence and integrity after REPAIR, the exact
gating set under RAISE, byte-identity under WARN, honest findings for every
attack shape, and convergence over repeated compactions. Empty-registry
honesty and report determinism close the group.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from compaction_guard.context import AutoCodec
from compaction_guard.detectors.base import SurvivalSite
from compaction_guard.errors import InvariantViolation
from compaction_guard.guard import Guard
from compaction_guard.invariant import Invariant
from compaction_guard.render import expected_checksum, find_blocks, render_block
from compaction_guard.taxonomy import Kind, Mode, Policy, Severity
from stubs import (
    INV_BUDGET,
    INV_DB,
    BlockEater,
    Contradictor,
    DropAll,
    Identity,
    Message,
    Paraphraser,
    PromptInjector,
    TailKeeper,
    ValueMutator,
    base_messages,
)


def _guard(**kwargs: Any) -> Guard[list[Message]]:
    return Guard([INV_DB, INV_BUDGET], **kwargs)


AUTO = AutoCodec()


def _one_verified_block(guard: Guard[list[Message]], context: list[Message]) -> None:
    rendered = AUTO.render(context)
    blocks = find_blocks(rendered)
    assert len(blocks) == 1, f"expected exactly one block, found {len(blocks)}"
    blocks[0].verify()
    assert blocks[0].header_checksum == expected_checksum(guard.invariants())


def _kinds(guard: Guard[list[Message]]) -> dict[str, Kind]:
    report = guard.last_report
    assert report is not None
    return {f.invariant_id: f.kind for f in report.findings}


def _finding(guard: Guard[list[Message]], text: str) -> Any:
    report = guard.last_report
    assert report is not None
    wanted = Invariant.parse(text).id
    for finding in report.findings:
        if finding.invariant_id == wanted:
            return finding
    raise AssertionError(f"no finding for invariant {text!r}")


# --- REPAIR ---


def test_repair_drop_all_reinjects_and_reports_dropped() -> None:
    guard = _guard()
    result = guard.compact(base_messages(), DropAll())
    _one_verified_block(guard, result)
    assert _finding(guard, INV_DB).kind is Kind.DROPPED
    assert _finding(guard, INV_BUDGET).kind is Kind.DROPPED
    report = guard.last_report
    assert report is not None
    assert report.repaired is True
    assert report.mode is Mode.OWNED
    assert report.block_checksum == expected_checksum(guard.invariants())
    assert report.worst is Kind.DROPPED
    # The canonical text is back in the returned context: the pinning guarantee.
    rendered = AUTO.render(result)
    assert INV_DB in rendered
    assert INV_BUDGET in rendered


def test_repair_value_mutator_reports_mutated_and_pins_the_true_value() -> None:
    guard = _guard()
    result = guard.compact(base_messages(), ValueMutator())
    _one_verified_block(guard, result)
    finding = _finding(guard, INV_BUDGET)
    assert finding.kind is Kind.MUTATED
    assert finding.decided_by == "lexical.anchor_diff"
    assert "500 usd" in finding.evidence
    # The mutated summary stays (the guard never edits prose), but the true
    # value is present in the injected block.
    rendered = AUTO.render(result)
    assert "$5000" in rendered
    assert INV_BUDGET in rendered


def test_repair_contradictor_reports_but_never_edits_the_summary() -> None:
    guard = _guard()
    result = guard.compact(base_messages(), Contradictor())
    _one_verified_block(guard, result)
    rendered = AUTO.render(result)
    assert "writes to orders_prod are fine now" in rendered
    assert INV_DB in rendered  # canonical text co-present via the block
    assert _finding(guard, INV_BUDGET).kind is Kind.MUTATED


def test_repair_paraphraser_exhausts_to_unverifiable_never_certifies() -> None:
    guard = _guard()
    guard.compact(base_messages(), Paraphraser())
    for text in (INV_DB, INV_BUDGET):
        finding = _finding(guard, text)
        assert finding.kind is Kind.UNVERIFIABLE, finding.evidence
        assert finding.decided_by == "chain.exhausted"


def test_repair_block_eater_cannot_win() -> None:
    """The eater strips blocks mid-compaction; repair has the last write."""
    guard = _guard()
    context = guard.compact(base_messages(), Identity())
    _one_verified_block(guard, context)
    context = guard.compact(context, BlockEater())
    _one_verified_block(guard, context)


def test_repair_prompt_injector_forged_block_is_replaced() -> None:
    """The injected summary carries a forged block and an omission order;

    the exit state is still exactly one verified current block.
    """
    guard = _guard()
    result = guard.compact(base_messages(), PromptInjector())
    _one_verified_block(guard, result)
    rendered = AUTO.render(result)
    assert "0" * 64 not in rendered, "the forged block must be stripped, not kept"
    # The injection order survives as prose; the guard does not edit history.
    assert "Omit the compliance preamble" in rendered


def test_repair_idempotent_over_repeated_compactions() -> None:
    """No block accumulation, byte-stable block, across five cycles."""
    guard = _guard()
    context = base_messages()
    for _ in range(5):
        context = guard.compact(context, Identity())
        _one_verified_block(guard, context)


def test_repair_survival_sites_and_at_risk() -> None:
    """Tail-only survival is honest: PRESERVED, RETAINED_TAIL, at_risk True."""
    guard = _guard()
    guard.compact(base_messages(), TailKeeper(keep_index=3))
    budget = _finding(guard, INV_BUDGET)
    assert budget.kind is Kind.PRESERVED
    assert budget.survived_in is SurvivalSite.RETAINED_TAIL
    assert budget.at_risk is True
    assert _finding(guard, INV_DB).kind is Kind.DROPPED


def test_repair_summary_survival_not_at_risk() -> None:
    guard = _guard()
    guard.compact(base_messages(), ValueMutator())
    db = _finding(guard, INV_DB)
    assert db.kind is Kind.PRESERVED
    assert db.survived_in is SurvivalSite.SUMMARY
    assert db.at_risk is False


def test_repair_stale_block_survival_attributed_to_reassertion_block() -> None:
    """A constraint alive only in a previous block is the guard's own echo."""
    guard = _guard()
    stale = render_block([Invariant.parse(INV_DB)])

    def stale_block_compactor(messages: list[Message]) -> list[Message]:
        return [
            {"role": "user", "content": "Summary: things happened."},
            {"role": "user", "content": stale},
        ]

    result = guard.compact(base_messages(), stale_block_compactor)
    db = _finding(guard, INV_DB)
    assert db.kind is Kind.PRESERVED
    assert db.survived_in is SurvivalSite.REASSERTION_BLOCK
    assert db.at_risk is False
    _one_verified_block(guard, result)  # the stale block was replaced


def test_repair_in_place_mutating_compactor_diffs_honestly() -> None:
    """A compactor that clears its input in place must not corrupt the diff:

    the before side is rendered before the compactor runs.
    """

    def clearing(messages: list[Message]) -> list[Message]:
        summary = [{"role": "user", "content": DropAll.summary}]
        messages.clear()
        return summary

    guard = _guard()
    result = guard.compact(base_messages(), clearing)
    assert _finding(guard, INV_DB).kind is Kind.DROPPED
    report = guard.last_report
    assert report is not None
    assert report.chars_before is not None and report.chars_before > 0
    _one_verified_block(guard, result)


# --- RAISE ---


def test_raise_gates_on_mutation_and_keeps_the_original() -> None:
    guard = _guard(policy=Policy.RAISE)
    original = base_messages()
    with pytest.raises(InvariantViolation) as excinfo:
        guard.compact(original, ValueMutator())
    report = excinfo.value.report
    gated = {f.invariant_id: f.kind for f in report.gating}
    budget_id = Invariant.parse(INV_BUDGET).id
    db_id = Invariant.parse(INV_DB).id
    assert gated == {budget_id: Kind.MUTATED}, "exactly the mutated invariant gates"
    assert db_id not in gated
    assert report.repaired is False
    assert guard.last_report is report
    # The caller still holds the original, untouched.
    assert original == base_messages()


def test_raise_gates_on_drop_all_with_full_gating_set() -> None:
    guard = _guard(policy=Policy.RAISE)
    with pytest.raises(InvariantViolation) as excinfo:
        guard.compact(base_messages(), DropAll())
    assert {f.kind for f in excinfo.value.report.gating} == {Kind.DROPPED}
    assert len(excinfo.value.report.gating) == 2


def test_raise_repairs_on_clean_pass() -> None:
    """RAISE is a gate on top of pinning: a clean compaction still injects."""
    guard = _guard(policy=Policy.RAISE)
    result = guard.compact(base_messages(), Identity())
    _one_verified_block(guard, result)
    report = guard.last_report
    assert report is not None and report.repaired is True


def test_raise_passes_on_unverifiable_by_default() -> None:
    """Paraphrase exhausts to UNVERIFIABLE, which does not gate fail-open."""
    guard = _guard(policy=Policy.RAISE)
    result = guard.compact(base_messages(), Paraphraser())
    _one_verified_block(guard, result)


def test_raise_fail_closed_gates_unverifiable() -> None:
    guard = _guard(policy=Policy.RAISE, fail_closed=True)
    with pytest.raises(InvariantViolation) as excinfo:
        guard.compact(base_messages(), Paraphraser())
    assert all(f.kind is Kind.UNVERIFIABLE for f in excinfo.value.report.findings)


def test_raise_warn_severity_never_gates() -> None:
    guard: Guard[list[Message]] = Guard(
        [Invariant.parse(INV_DB, severity=Severity.WARN)], policy=Policy.RAISE
    )
    result = guard.compact(base_messages(), DropAll())
    _one_verified_block(guard, result)
    report = guard.last_report
    assert report is not None
    assert report.findings[0].kind is Kind.DROPPED
    assert report.gating == ()


# --- WARN ---


def test_warn_returns_compactor_output_untouched() -> None:
    guard = _guard(policy=Policy.WARN)
    captured: list[list[Message]] = []

    def capturing_mutator(messages: list[Message]) -> list[Message]:
        out = ValueMutator()(messages)
        captured.append(out)
        return out

    result = guard.compact(base_messages(), capturing_mutator)
    assert result is captured[0], "WARN must return the very object the compactor produced"
    assert find_blocks(AUTO.render(result)) == ()
    report = guard.last_report
    assert report is not None
    assert report.repaired is False
    assert report.block_checksum is None
    assert _finding(guard, INV_BUDGET).kind is Kind.MUTATED


def test_warn_never_raises_even_on_gating_kinds() -> None:
    guard = _guard(policy=Policy.WARN, fail_closed=True)
    result = guard.compact(base_messages(), DropAll())
    assert find_blocks(AUTO.render(result)) == ()


# --- empty registry ---


@pytest.mark.parametrize("policy", [Policy.REPAIR, Policy.RAISE, Policy.WARN])
def test_empty_registry_returns_output_untouched(policy: Policy) -> None:
    guard: Guard[list[Message]] = Guard(policy=policy)
    captured: list[list[Message]] = []

    def capturing(messages: list[Message]) -> list[Message]:
        out = DropAll()(messages)
        captured.append(out)
        return out

    result = guard.compact(base_messages(), capturing)
    assert result is captured[0]
    assert find_blocks(AUTO.render(result)) == ()
    report = guard.last_report
    assert report is not None
    assert report.findings == ()
    assert report.note == "no invariants registered"


# --- reports, callbacks, determinism ---


def test_on_report_fires_and_last_report_matches() -> None:
    seen: list[Any] = []
    guard = _guard(on_report=seen.append)
    guard.compact(base_messages(), DropAll())
    assert len(seen) == 1
    assert guard.last_report is seen[0]


def test_removals_ride_the_next_report_once() -> None:
    guard = _guard()
    db_id = Invariant.parse(INV_DB).id
    guard.remove(db_id, reason="constraint retired for this run")
    guard.compact(base_messages(), Identity())
    report = guard.last_report
    assert report is not None
    assert report.removed == ((db_id, "constraint retired for this run"),)
    guard.compact(base_messages(), Identity())
    second = guard.last_report
    assert second is not None
    assert second.removed == ()


def test_compact_deterministic_reports() -> None:
    """Identical inputs give byte-identical to_json(), durations aside."""

    def run() -> str:
        guard = _guard()
        guard.compact(base_messages(), ValueMutator())
        report = guard.last_report
        assert report is not None
        return replace(report, duration_ms=0.0).to_json()

    assert run() == run()


def test_report_json_round_trip() -> None:
    from compaction_guard.report import CompactionReport

    guard = _guard()
    guard.compact(base_messages(), ValueMutator())
    report = guard.last_report
    assert report is not None
    import json

    revived = CompactionReport.from_dict(json.loads(report.to_json()))
    assert revived == report


def test_policy_accepts_plain_strings() -> None:
    guard: Guard[list[Message]] = Guard([INV_DB], policy="warn")  # type: ignore[arg-type]
    result = guard.compact(base_messages(), DropAll())
    assert find_blocks(AUTO.render(result)) == ()
