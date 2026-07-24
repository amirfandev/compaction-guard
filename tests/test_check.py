"""check(): the pure functional core, and Guard.check's REASSERTED bookkeeping."""

from __future__ import annotations

import pytest

from compaction_guard.check import check
from compaction_guard.errors import CompactionGuardError
from compaction_guard.guard import Guard
from compaction_guard.invariant import Invariant
from compaction_guard.taxonomy import Kind, Mode
from stubs import INV_BUDGET, INV_DB, Message


def test_check_reports_reasserted_mode() -> None:
    report = check([Invariant.parse(INV_DB)], "unrelated summary text")
    assert report.mode is Mode.REASSERTED
    assert report.chars_before is None
    assert report.repaired is False
    assert report.block_checksum is None
    assert report.chars_after == len("unrelated summary text")


def test_check_empty_registry_note() -> None:
    report = check([], "any text")
    assert report.findings == ()
    assert report.note == "no invariants registered"
    assert report.worst is Kind.PRESERVED  # honest: nothing registered to lose


def test_check_findings_in_registry_order() -> None:
    invariants = [Invariant.parse(INV_DB), Invariant.parse(INV_BUDGET)]
    report = check(invariants, INV_BUDGET)
    assert [f.invariant_id for f in report.findings] == [inv.id for inv in invariants]


def test_check_empty_detector_sequence_is_respected() -> None:
    """detectors=() means ask no one, which honestly answers UNVERIFIABLE;

    it is distinct from detectors=None, the default lexical chain.
    """
    report = check([Invariant.parse(INV_DB)], INV_DB, detectors=())
    assert report.findings[0].kind is Kind.UNVERIFIABLE
    assert report.findings[0].decided_by == "chain.exhausted"

    default = check([Invariant.parse(INV_DB)], INV_DB, detectors=None)
    assert default.findings[0].kind is Kind.PRESERVED


def test_guard_check_does_not_touch_registry_or_inject() -> None:
    guard: Guard[list[Message]] = Guard([INV_DB, INV_BUDGET])
    before = guard.invariants()
    report = guard.check("The budget cap for this run is $5000.")
    assert guard.invariants() == before
    assert report.mode is Mode.REASSERTED
    budget_id = Invariant.parse(INV_BUDGET).id
    kinds = {f.invariant_id: f.kind for f in report.findings}
    assert kinds[budget_id] is Kind.MUTATED


def test_guard_check_drains_removal_ledger() -> None:
    """A host that only ever re-asserts still gets its eviction trace."""
    guard: Guard[list[Message]] = Guard([INV_DB, INV_BUDGET])
    db_id = Invariant.parse(INV_DB).id
    guard.remove(db_id, reason="scoped out at turn 12")
    report = guard.check("whatever text")
    assert report.removed == ((db_id, "scoped out at turn 12"),)
    assert guard.check("again").removed == ()
    assert guard.last_report is not None


def test_remove_requires_registered_id_and_reason() -> None:
    guard: Guard[list[Message]] = Guard([INV_DB])
    with pytest.raises(CompactionGuardError, match="nothing was removed"):
        guard.remove("missing000000", reason="why not")
    with pytest.raises(ValueError, match="reason"):
        guard.remove(Invariant.parse(INV_DB).id, reason="   ")


def test_reassertion_block_is_current_and_verifiable() -> None:
    from compaction_guard.render import expected_checksum, find_blocks

    guard: Guard[list[Message]] = Guard([INV_DB])
    block = guard.reassertion_block()
    found = find_blocks(block)
    assert len(found) == 1
    found[0].verify()
    assert found[0].header_checksum == expected_checksum(guard.invariants())
    guard.add(INV_BUDGET)
    grown = find_blocks(guard.reassertion_block())[0]
    assert grown.header_checksum == expected_checksum(guard.invariants())
    assert grown.header_checksum != found[0].header_checksum


def test_check_report_losses_and_gating_semantics() -> None:
    invariants = [Invariant.parse(INV_DB), Invariant.parse(INV_BUDGET)]
    report = check(invariants, "The budget cap for this run is $5000.")
    losses = report.losses()
    assert all(f.kind is not Kind.PRESERVED for f in losses)
    # UNVERIFIABLE counts as a loss (unverified survival is not survival)
    # but never gates without fail_closed, which the report cannot know.
    assert {f.kind for f in report.gating} <= {Kind.MUTATED, Kind.DROPPED, Kind.WEAKENED, Kind.CONTRADICTED}
