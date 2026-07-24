"""Token estimation default, block budget accounting, refusal semantics.

The budget exists because the sentinel block competes with the transcript for
context window, and a pinning tool that quietly grew without bound would
recreate the pressure that causes compaction in the first place. Enforcement
happens at ``add()`` time, through ``ensure_fits``: the one moment a human is
at the call site and can decide what to drop. Nothing is ever truncated. A
truncated constraint is a mutated constraint, injected by the tool whose job
is to detect mutation, so the only honest options on overflow are refusal or
an explicit ``remove()`` with a recorded reason.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from .errors import BudgetExceeded
from .render import render_block
from .report import BlockBudget

if TYPE_CHECKING:
    from .invariant import Invariant

__all__ = ["ensure_fits", "estimate_tokens", "measure"]


def estimate_tokens(text: str) -> int:
    """The default estimator: UTF-8 byte count divided by 3, rounded down.

    Three bytes per token is a deliberately pessimistic divisor for the
    English prose that constraints are written in, so this over-counts
    relative to production tokenisers. That direction is chosen on purpose:
    the estimate feeds a refusal, and a refusal that fires slightly early
    costs the user a reconsidered ``add()``, while one that fires late costs
    context window at exactly the moment the window is already under
    pressure. Callers with a real tokeniser pass their own
    ``token_estimator`` and get exact accounting.
    """
    return len(text.encode("utf-8")) // 3


def measure(
    invariants: Sequence[Invariant],
    *,
    max_tokens: int,
    estimator: Callable[[str], int] | None = None,
) -> BlockBudget:
    """Account for the block these invariants would render.

    ``used_tokens`` measures the full rendered block, wire overhead
    included, because the context window pays for markers and checksums too.
    ``per_invariant`` reports each invariant's registration-time
    ``token_cost`` unchanged; the guard uses one estimator for both
    registration and measurement, which keeps the two views consistent by
    construction rather than by re-computation. An empty registry measures
    zero: no block would be injected, so no tokens are spent.
    """
    est = estimator if estimator is not None else estimate_tokens
    invs = tuple(invariants)
    used = est(render_block(invs)) if invs else 0
    return BlockBudget(
        used_tokens=used,
        max_tokens=max_tokens,
        per_invariant=tuple((inv.id, inv.token_cost) for inv in invs),
    )


def ensure_fits(
    invariants: Sequence[Invariant],
    *,
    max_tokens: int,
    estimator: Callable[[str], int] | None = None,
) -> BlockBudget:
    """Measure, and refuse loudly if the block exceeds its budget.

    The message names the overrun and the largest contributors, because the
    caller's next move is to pick something to remove or to raise the cap,
    and making them re-derive the numbers would just delay that decision.
    Returns the ``BlockBudget`` on success so ``Guard.budget()`` and
    ``Guard.add()`` share one measurement.
    """
    budget = measure(invariants, max_tokens=max_tokens, estimator=estimator)
    if budget.used_tokens > max_tokens:
        largest = sorted(budget.per_invariant, key=lambda item: item[1], reverse=True)
        top = ", ".join(f"{inv_id} ({tokens} tokens)" for inv_id, tokens in largest[:3])
        raise BudgetExceeded(
            f"sentinel block would use {budget.used_tokens} tokens, budget is "
            f"{max_tokens}. Nothing is truncated. Remove an invariant "
            f"(Guard.remove, with a reason) or raise max_block_tokens. "
            f"Largest entries: {top}"
        )
    return budget
