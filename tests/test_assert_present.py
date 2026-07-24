"""assert_present: the microsecond per-turn integrity check.

Passes after repair; raises after a downstream BlockEater; raises on a
checksum-breaking edit; raises when the block is stale relative to a
registry that grew mid-run; returns vacuously on an empty registry.
"""

from __future__ import annotations

import pytest

from compaction_guard.errors import BlockIntegrityError
from compaction_guard.guard import Guard
from compaction_guard.taxonomy import Kind
from stubs import INV_BUDGET, INV_DB, BlockEater, Identity, Message, base_messages


def _repaired_guard_and_context() -> tuple[Guard[list[Message]], list[Message]]:
    guard: Guard[list[Message]] = Guard([INV_DB, INV_BUDGET])
    context = guard.compact(base_messages(), Identity())
    return guard, context


def test_passes_after_repair() -> None:
    guard, context = _repaired_guard_and_context()
    guard.assert_present(context)


def test_raises_after_block_eater() -> None:
    """A downstream trim of guard-owned text is a harness bug, not a finding."""
    guard, context = _repaired_guard_and_context()
    eaten = BlockEater()(context)
    assert isinstance(eaten, list)
    with pytest.raises(BlockIntegrityError, match="no sentinel block found"):
        guard.assert_present(eaten)
    # Never a quiet finding: last_report still describes the compaction, and
    # carries no integrity verdicts of any kind.
    report = guard.last_report
    assert report is not None
    assert all(f.kind in set(Kind) for f in report.findings)


def test_raises_on_checksum_breaking_edit() -> None:
    guard, context = _repaired_guard_and_context()
    edited = [
        (
            {**message, "content": message["content"].replace("$500", "$5000")}
            if isinstance(message.get("content"), str) and "COMPACTION-GUARD" in message["content"]
            else message
        )
        for message in context
    ]
    assert edited != context, "the edit must actually land inside the block"
    with pytest.raises(BlockIntegrityError, match="edited or is stale"):
        guard.assert_present(edited)


def test_raises_when_block_is_stale_after_add() -> None:
    """An add() since the last injection means the new constraint is unpinned;

    passing silently would be fake protection for exactly that constraint.
    """
    guard, context = _repaired_guard_and_context()
    guard.assert_present(context)
    guard.add("Actually, cap this run at $200.", source="turn:30")
    with pytest.raises(BlockIntegrityError, match="stale"):
        guard.assert_present(context)
    # The next compaction repairs with the grown registry and the check passes.
    repaired = guard.compact(context, Identity())
    guard.assert_present(repaired)


def test_vacuous_on_empty_registry() -> None:
    """No block is owed, so nothing is rendered and nothing raises,

    whatever shape the context is (even one the codec would refuse).
    """
    guard: Guard[object] = Guard()
    guard.assert_present({"unrenderable": True})
