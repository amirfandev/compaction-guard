"""The Anthropic helpers: pause-and-verify (REASSERTED) and the SessionStart
hook path (UNOBSERVED). Strings in, strings out, so no fake beyond the guard
itself is needed; what is under test is mode honesty and block hygiene.
"""

from __future__ import annotations

import json
from typing import Any

from compaction_guard.guard import Guard
from compaction_guard.integrations.anthropic import (
    COMPACT_SOURCE,
    session_start_context,
    session_start_hook_output,
    verify_and_reassert,
)
from compaction_guard.render import expected_checksum, find_blocks
from compaction_guard.taxonomy import Kind, Mode
from stubs import INV_BUDGET, INV_DB


def test_pause_flow_verifies_and_appends_current_block() -> None:
    guard: Guard[str] = Guard([INV_DB, INV_BUDGET])
    summary = "Summary: the budget cap for this run is $5000."
    report, reasserted = verify_and_reassert(guard, summary)
    assert report.mode is Mode.REASSERTED
    assert report.chars_before is None
    assert report.repaired is False
    assert Kind.MUTATED in {f.kind for f in report.findings}
    assert guard.last_report is report
    blocks = find_blocks(reasserted)
    assert len(blocks) == 1
    assert blocks[0].header_checksum == expected_checksum(guard.invariants())
    assert summary.split("\n")[0] in reasserted, "the summary prose is kept, never edited"


def test_repeated_pauses_converge_to_one_block() -> None:
    guard: Guard[str] = Guard([INV_DB])
    _report, text = verify_and_reassert(guard, "Summary: work continued.")
    for _ in range(3):
        _report, text = verify_and_reassert(guard, text)
    assert len(find_blocks(text)) == 1


def test_pause_flow_empty_registry_returns_summary_unchanged() -> None:
    guard: Guard[str] = Guard()
    report, reasserted = verify_and_reassert(guard, "Summary: things happened.")
    assert reasserted == "Summary: things happened."
    assert report.note == "no invariants registered"


def test_session_start_only_fires_on_compact_source() -> None:
    guard: Guard[str] = Guard([INV_DB])
    for source in ("startup", "resume", "clear"):
        assert session_start_context(guard, source) is None
    assert guard.last_report is None, "non-compact sources emit nothing"


def test_session_start_compact_emits_unobserved_and_returns_block() -> None:
    reports: list[Any] = []
    guard: Guard[str] = Guard([INV_DB], on_report=reports.append)
    block = session_start_context(guard, COMPACT_SOURCE)
    assert block is not None
    found = find_blocks(block)
    assert len(found) == 1
    assert found[0].header_checksum == expected_checksum(guard.invariants())
    assert len(reports) == 1
    report = reports[0]
    assert report.mode is Mode.UNOBSERVED
    assert report.repaired is False, "handing text to a hook is an offer, not proof"
    assert all(f.kind is Kind.UNVERIFIABLE for f in report.findings)


def test_session_start_hook_output_shape() -> None:
    guard: Guard[str] = Guard([INV_DB])
    payload = session_start_hook_output(guard, COMPACT_SOURCE)
    encoded = json.loads(json.dumps(payload))
    inner = encoded["hookSpecificOutput"]
    assert inner["hookEventName"] == "SessionStart"
    assert find_blocks(inner["additionalContext"])
    assert session_start_hook_output(guard, "startup") == {}
