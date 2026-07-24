"""Permanent regressions from adversarial review, per the spec's rule that any
false-certify counterexample becomes a fixture before the fix ships.

Four confirmed defect families live here: the carried-block blinding of every
compaction after the first repair, anchor pollution across sentence and
invariant boundaries, the escape scheme collapsing under normalize and making
distinct texts share a checksum, and the assert_present crashes on the spec's
own canonical usage. Each test failed against the pre-fix code.
"""

from __future__ import annotations

import pytest

import compaction_guard as cg
from compaction_guard.detectors.base import BLOCK_ECHO_DECIDED_BY, SurvivalSite
from compaction_guard.errors import BlockIntegrityError, InvariantViolation
from compaction_guard.guard import Guard
from compaction_guard.invariant import Invariant
from compaction_guard.render import (
    assert_block_present,
    expected_checksum,
    find_blocks,
    render_block,
)
from compaction_guard.taxonomy import Kind, Policy
from stubs import Message

CAP = "The budget cap for this run is $500."


def _base() -> list[Message]:
    return [
        {"role": "system", "content": "You are a task agent."},
        {"role": "user", "content": f"Standing instruction: {CAP}"},
        {"role": "user", "content": "lots of chatter"},
    ]


def _keep_tail_mutator(messages: list[Message]) -> list[Message]:
    """A realistic compactor: fresh summary with a mutated value, last message
    (the guard's own injected block) kept verbatim."""
    return [
        messages[0],
        {"role": "user", "content": "Summary: the budget cap for this run is $5000."},
        *messages[-1:],
    ]


# --- the carried block must not blind later compactions ---


def test_raise_still_gates_when_compactor_carries_the_guards_block() -> None:
    """Second compaction keeps the injected block and mutates the value.

    Pre-fix, the carried block satisfied rule 1a and every later compaction
    reported PRESERVED, so RAISE never gated again after the first repair.
    """
    guard: Guard[list[Message]] = Guard([CAP], policy=Policy.RAISE)
    out1 = guard.compact(
        _base(),
        compactor=lambda m: [m[0], {"role": "user", "content": "Summary: a $500 cap applies."}],
    )
    with pytest.raises(InvariantViolation) as excinfo:
        guard.compact(out1, compactor=_keep_tail_mutator)
    gated = {f.kind for f in excinfo.value.report.gating}
    assert gated == {Kind.MUTATED}


def test_block_only_survival_is_echo_not_summariser_credit() -> None:
    """Constraint alive only in a carried block: PRESERVED, but attributed to
    the chain's echo rule, never to lexical evidence the summariser produced."""
    guard: Guard[list[Message]] = Guard([CAP])
    out1 = guard.compact(_base(), compactor=lambda m: [m[0]])
    guard.compact(
        out1,
        compactor=lambda m: [
            {"role": "user", "content": "Summary: routine work continued."},
            *m[-1:],
        ],
    )
    report = guard.last_report
    assert report is not None
    finding = report.findings[0]
    assert finding.kind is Kind.PRESERVED
    assert finding.decided_by == BLOCK_ECHO_DECIDED_BY
    assert finding.survived_in is SurvivalSite.REASSERTION_BLOCK


def test_reasserted_pause_flow_gates_despite_carried_block() -> None:
    """The Anthropic pause shape: the second pause's summary carries the block
    the host injected earlier plus a mutated restatement. The report must
    still gate on the mutation."""
    guard: Guard[str] = Guard([CAP])
    carried = guard.reassertion_block()
    report = guard.check(
        f"Summary: the budget cap for this run is $5000.\n\n{carried}"
    )
    assert {f.kind for f in report.gating} == {Kind.MUTATED}


# --- anchor pollution across invariant boundaries ---


def test_shared_value_anchor_between_invariants_does_not_mask_mutation() -> None:
    """Two invariants share the $500 value. The summary mutates the cap and
    preserves the refund rule; pre-fix both certified PRESERVED and RAISE
    stayed silent while a wrong live value drove action."""
    refund = "Refunds above $500 require human approval."
    guard: Guard[list[Message]] = Guard([CAP, refund], policy=Policy.RAISE)

    def compactor(messages: list[Message]) -> list[Message]:
        return [
            messages[0],
            {
                "role": "user",
                "content": (
                    "Summary: the budget cap for this run is $5000. "
                    "Refunds above $500 require human approval."
                ),
            },
        ]

    with pytest.raises(InvariantViolation) as excinfo:
        guard.compact(_base(), compactor)
    report = excinfo.value.report
    kinds = {f.invariant_id: f.kind for f in report.findings}
    assert kinds[Invariant.parse(CAP).id] is Kind.MUTATED
    assert kinds[Invariant.parse(refund).id] is Kind.PRESERVED


# --- checksum integrity under the escape scheme ---


def test_distinct_multiline_texts_have_distinct_checksums() -> None:
    """normalize deletes backslashes, so the old backslash escapes made
    "a\\nb" and "a nb" hash identically: a forgeable equivalence in the
    integrity primitive. The escape mark now survives normalisation."""
    with_newline = Invariant.parse(
        "Deploy only to us-east-1.\nNever touch eu-west-1.", id="x1"
    )
    lookalike = Invariant.parse(
        "Deploy only to us-east-1. nNever touch eu-west-1.", id="x1"
    )
    assert expected_checksum([with_newline]) != expected_checksum([lookalike])
    rendered = render_block([with_newline])
    forged = render_block([lookalike])
    with pytest.raises(BlockIntegrityError):
        assert_block_present(forged, expected_checksum([with_newline]))
    assert_block_present(rendered, expected_checksum([with_newline]))


def test_escape_mark_literal_round_trips() -> None:
    """A constraint containing the escape mark itself must round-trip and
    must not collide with its escaped spelling."""
    text = "Rate limit is 5 req¦s; never exceed it."
    inv = Invariant.parse(text, id="mark")
    block = find_blocks(render_block([inv]))[0]
    block.verify()
    assert block.entries == (("mark", text),)


# --- assert_present on the canonical loop ---


def test_assert_present_is_quiet_before_any_block_was_issued() -> None:
    """The spec's canonical loop calls assert_present every turn, starting
    before the first compaction. No block has been issued, none is owed."""
    guard: Guard[list[Message]] = Guard([CAP])
    guard.assert_present(_base())  # must not raise


def test_assert_present_demands_presence_after_reassertion_block() -> None:
    """Asking for the block is intent to inject it; from then on absence is
    a failure, which keeps the UNOBSERVED promise checkable."""
    guard: Guard[list[Message]] = Guard([CAP])
    block = guard.reassertion_block()
    context = [*_base(), {"role": "user", "content": block}]
    guard.assert_present(context)
    with pytest.raises(BlockIntegrityError, match="no sentinel block"):
        guard.assert_present(_base())


def test_assert_present_stale_error_names_the_repin_path() -> None:
    guard: Guard[list[Message]] = Guard([CAP])
    context = guard.compact(_base(), lambda m: list(m))
    guard.add("Actually, cap this run at $200.", source="turn:30")
    with pytest.raises(BlockIntegrityError, match="stale") as excinfo:
        guard.assert_present(context)
    assert "reassertion_block" in str(excinfo.value)
    repinned = guard.compact(context, lambda m: list(m))
    guard.assert_present(repinned)


# --- functional core ergonomics ---


def test_module_check_accepts_strings_and_single_string() -> None:
    """cg.check takes what Guard takes; the str/Invariant asymmetry was a
    confirmed first-call crash deep inside a generator."""
    report = cg.check([CAP], "The budget cap for this run is $5000.")
    assert report.findings[0].kind is Kind.MUTATED
    single = cg.check(CAP, "The budget cap for this run is $500.")
    assert single.findings[0].kind is Kind.PRESERVED
    mixed = cg.check([Invariant.parse(CAP), "Never touch eu-west-1."], CAP)
    assert len(mixed.findings) == 2


def test_detector_classes_are_public() -> None:
    """The extension surface must be reachable through the package: passing
    detectors= replaces the default chain, so the default and the judge
    wrapper both need public names."""
    for name in ("LexicalDetector", "JudgeDetector", "EmbeddingDetector", "NLIDetector"):
        assert name in cg.__all__
        assert hasattr(cg, name)
    judged = cg.Guard(
        [CAP],
        detectors=[cg.LexicalDetector(), cg.JudgeDetector(judge=lambda prompt: "not json")],
    )
    judged.check("unrelated text entirely")
    report = judged.last_report
    assert report is not None
    assert report.findings[0].kind is Kind.UNVERIFIABLE
