"""OpenAI Agents SDK: a guarded ``Session``, OWNED or UNOBSERVED by who compacts.

The Sessions surface is a small protocol (``get_items``, ``add_items``,
``pop_item``, ``clear_session``), so this module needs no SDK import at
all: ``GuardedSession`` wraps anything session-shaped, real or fake, and
session items are plain dicts the default codec already reads and writes.
Which mode you get depends on who runs the compactor.

With a caller-supplied ``compactor``: OWNED. When the rendered session
passes ``trigger_tokens`` after an ``add_items``, or on an explicit
``compact_now()``, the full item list goes through ``Guard.compact``:
diff, detection, policy, verified injection, then the store is rewritten
with the repaired items. Guarantee: every compaction is classified and
exits with exactly one verified block, and full fetches after the first
repair run ``assert_present``. It cannot prevent host code from editing
the store behind the wrapper between calls; the next full fetch catches
that, loudly, rather than preventing it.

Without a compactor: UNOBSERVED, for sessions whose compaction happens
server-side (the ``/responses/compact`` wrapper), where the compacted
summary item is documented as opaque and not human-interpretable. There
is nothing to examine, so nothing is examined. When the sentinel block
stops verifying against a full ``get_items``, the wrapper appends a
fresh ``reassertion_block()`` as a new input item, verifies its presence
in what it returns, and emits a report whose findings are all
UNVERIFIABLE. Guarantee: the current block is present in every full
fetch this wrapper returns. It cannot inspect the summary, classify what
survived, or remove stale blocks from server-held history; superseded
blocks accumulate there as benign duplication, the residue this library
accepts everywhere rather than rewriting what was sent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from ..budget import estimate_tokens
from ..errors import BlockIntegrityError, CodecError, CompactionGuardError
from ..guard import Guard
from ..render import assert_block_present, expected_checksum
from ..report import CompactionReport
from ._shared import emit_unobserved

__all__ = ["GuardedSession", "SessionLike"]


class SessionLike(Protocol):
    """The slice of the Agents SDK ``Session`` protocol this wrapper needs.

    Declared locally so the module imports without the SDK and tests run
    against fakes; any real SDK session satisfies it structurally.
    """

    async def get_items(self, limit: int | None = None) -> list[Any]: ...

    async def add_items(self, items: list[Any]) -> None: ...

    async def pop_item(self) -> Any: ...

    async def clear_session(self) -> None: ...


class GuardedSession:
    """Wraps a session so compaction cannot silently unbind the agent.

    ``compactor`` takes and returns a full item list, synchronously; the
    guard must hold both sides of the compaction on the critical path, so
    an async summariser call belongs inside the callable the host writes.
    ``trigger_tokens`` is measured with the guard's own token estimator
    over the rendered item list, so the trigger and the block budget speak
    the same units.

    ``pop_item`` forwards untouched. Popping the block carrier itself is
    possible right after a compaction; the next full fetch raises in OWNED
    mode (the block the guard verified is gone) and re-asserts in
    UNOBSERVED mode. Neither path hides it.
    """

    def __init__(
        self,
        inner: SessionLike,
        guard: Guard[Any],
        compactor: Callable[[list[Any]], list[Any]] | None = None,
        trigger_tokens: int | None = None,
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._compactor = compactor
        self._trigger_tokens = trigger_tokens
        self._pinned = False
        self._seeded = False

    @property
    def session_id(self) -> str:
        """The wrapped session's id, or an empty string if it has none."""
        return str(getattr(self._inner, "session_id", ""))

    async def get_items(self, limit: int | None = None) -> list[Any]:
        """Fetch items; on full fetches, enforce what the mode can promise.

        A limited fetch is returned untouched: a window that may exclude
        the block by construction proves nothing about its absence, and
        appending a block per partial view would spam the transcript. The
        model-facing path fetches the full history.
        """
        items = list(await self._inner.get_items(limit))
        if limit is not None:
            return items
        if self._compactor is not None:
            if self._pinned:
                self._guard.assert_present(items)
            return items
        return await self._reassert(items)

    async def add_items(self, items: list[Any]) -> None:
        """Store items; in OWNED mode, compact when the trigger is passed.

        A render failure during the trigger estimate propagates as
        ``CodecError``: skipping the trigger on unrenderable items would
        mean never compacting and never saying so, which is silent
        no-protection twice over.
        """
        await self._inner.add_items(list(items))
        if self._compactor is None or self._trigger_tokens is None:
            return
        full = list(await self._inner.get_items(None))
        if self._estimate(full) <= self._trigger_tokens:
            return
        await self._compact(full)

    async def pop_item(self) -> Any:
        """Remove and return the most recent item, untouched."""
        return await self._inner.pop_item()

    async def clear_session(self) -> None:
        """Clear the store. A cleared session owes no block until reseeded."""
        await self._inner.clear_session()
        self._pinned = False
        self._seeded = False

    async def compact_now(self) -> CompactionReport | None:
        """Compact explicitly, regardless of the trigger. OWNED mode only.

        Returns the report the guard emitted. Under RAISE a gating finding
        raises ``InvariantViolation`` before the store is touched, so the
        session still holds the uncompacted history.
        """
        if self._compactor is None:
            raise CompactionGuardError(
                "compact_now() requires a compactor. This session wraps "
                "opaque compaction (UNOBSERVED) and can only re-assert the "
                "block; it has nothing to run."
            )
        full = list(await self._inner.get_items(None))
        return await self._compact(full)

    # --- internals ---

    async def _compact(self, full: list[Any]) -> CompactionReport | None:
        if self._compactor is None:
            raise CompactionGuardError("no compactor configured")
        repaired = self._guard.compact(full, self._compactor)
        await self._inner.clear_session()
        await self._inner.add_items(list(repaired))
        report = self._guard.last_report
        if report is not None and report.repaired:
            self._pinned = True
        return report

    async def _reassert(self, items: list[Any]) -> list[Any]:
        """UNOBSERVED presence enforcement over one full fetch.

        The first fetch that finds no block seeds one without a report:
        nothing was compacted yet, so there is nothing to record. Every
        later disappearance is evidence an opaque compaction (or a stale
        registry block after ``add()``) consumed it, and that is exactly
        what the UNOBSERVED report exists to state.
        """
        started = time.perf_counter()
        if not self._guard.invariants():
            return items
        try:
            # A direct presence check, not Guard.assert_present: that method
            # is deliberately quiet before the guard has issued a block, and
            # this wrapper's whole promise is that a block is present in
            # every full fetch, issued or not. Absence here must always
            # trigger seeding or re-assertion.
            assert_block_present(
                self._guard._codec.render(items),
                expected_checksum(self._guard.invariants()),
            )
        except BlockIntegrityError as exc:
            detail = str(exc)
        else:
            self._seeded = True
            return items
        carrier = {"role": "user", "content": self._guard.reassertion_block()}
        await self._inner.add_items([carrier])
        items = [*items, carrier]
        self._guard.assert_present(items)
        if self._seeded:
            emit_unobserved(
                self._guard,
                started=started,
                chars_after=self._measure(items),
                repaired=True,
                block_checksum=expected_checksum(self._guard.invariants()),
                note=(
                    "opaque compaction: sentinel block no longer verified in "
                    f"session items; re-asserted as a new input item. {detail}"
                ),
                evidence=(
                    "server-side compaction items are opaque; no summary text "
                    "was inspectable, so no detector ran"
                ),
            )
        else:
            self._seeded = True
        return items

    def _measure(self, items: list[Any]) -> int:
        """chars_after for the report; 0 when the codec cannot render.

        Uses the guard's codec so the number describes the same text
        verification read. Zero on failure rather than a guess, because an
        invented length would be a fabricated number.
        """
        try:
            return len(self._guard._codec.render(items))
        except CodecError:
            return 0

    def _estimate(self, items: list[Any]) -> int:
        estimator = self._guard._estimator
        if estimator is None:
            estimator = estimate_tokens
        return estimator(self._guard._codec.render(items))
