"""Budget refusal at add() time: loud, early, and nothing ever truncates."""

from __future__ import annotations

import pytest

from compaction_guard.budget import estimate_tokens
from compaction_guard.errors import BudgetExceeded, DuplicateInvariantId
from compaction_guard.guard import Guard
from compaction_guard.render import render_block
from stubs import INV_BUDGET, INV_DB, Message


def _fits_exactly_one() -> int:
    """A budget that admits INV_DB alone, wire overhead included."""
    guard: Guard[list[Message]] = Guard([INV_DB])
    return estimate_tokens(render_block(guard.invariants()))


def test_default_estimator_is_utf8_bytes_over_three() -> None:
    assert estimate_tokens("abcdef") == 2
    assert estimate_tokens("") == 0
    assert estimate_tokens("caña") == len("caña".encode()) // 3


def test_add_past_budget_raises_and_commits_nothing() -> None:
    guard: Guard[list[Message]] = Guard([INV_DB], max_block_tokens=_fits_exactly_one())
    before = guard.invariants()
    with pytest.raises(BudgetExceeded) as excinfo:
        guard.add(INV_BUDGET)
    assert guard.invariants() == before, "a refused add must leave the registry untouched"
    message = str(excinfo.value)
    assert "Nothing is truncated" in message
    assert "max_block_tokens" in message


def test_constructor_enforces_budget_too() -> None:
    with pytest.raises(BudgetExceeded):
        Guard([INV_DB, INV_BUDGET], max_block_tokens=_fits_exactly_one())


def test_refusal_names_the_largest_contributors() -> None:
    guard: Guard[list[Message]] = Guard([INV_DB], max_block_tokens=_fits_exactly_one())
    with pytest.raises(BudgetExceeded) as excinfo:
        guard.add(INV_BUDGET)
    db_id = guard.invariants()[0].id
    assert db_id in str(excinfo.value)


def test_budget_accounting_shape() -> None:
    guard: Guard[list[Message]] = Guard([INV_DB, INV_BUDGET], max_block_tokens=1024)
    budget = guard.budget()
    assert budget.max_tokens == 1024
    assert budget.used_tokens == estimate_tokens(render_block(guard.invariants()))
    ids = [inv_id for inv_id, _tokens in budget.per_invariant]
    assert ids == [inv.id for inv in guard.invariants()]
    # Wire overhead belongs to the format, not to any invariant: the block
    # costs more than the sum of its entries.
    assert budget.used_tokens > sum(tokens for _inv_id, tokens in budget.per_invariant)


def test_empty_registry_uses_zero_tokens() -> None:
    guard: Guard[list[Message]] = Guard()
    assert guard.budget().used_tokens == 0
    assert guard.budget().per_invariant == ()


def test_custom_estimator_flows_through() -> None:
    words: Guard[list[Message]] = Guard(token_estimator=lambda text: len(text.split()))
    added = words.add(INV_BUDGET)
    assert added.token_cost == len(INV_BUDGET.split())
    assert words.budget().used_tokens == len(render_block(words.invariants()).split())


def test_duplicate_id_refused_including_cosmetic_variants() -> None:
    """Derived ids hash the normalised text, so cosmetic re-registration collides."""
    guard: Guard[list[Message]] = Guard([INV_DB])
    with pytest.raises(DuplicateInvariantId):
        guard.add(INV_DB)
    with pytest.raises(DuplicateInvariantId):
        guard.add("the database ORDERS_PROD is production!!  Read-only queries only")


def test_invariant_parse_rejects_bad_input() -> None:
    from compaction_guard.invariant import Invariant

    with pytest.raises(ValueError, match="empty"):
        Invariant.parse("   \n  ")
    with pytest.raises(ValueError, match="whitespace"):
        Invariant.parse("fine text", id="bad id")
