"""GuardedSession against a minimal fake Session, per the spec's test plan.

The wrapper needs only the four-method session protocol, so a dict-backed
fake exercises the real adapter code with no SDK installed. OWNED mode is
tested through trigger and explicit compaction; UNOBSERVED mode through the
seed-then-reassert lifecycle, including the opaque compaction that eats the
block. Async methods run on a bare coroutine trampoline: no event loop, no
plugin, and by construction no hidden I/O.
"""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any

import pytest

from compaction_guard.errors import BlockIntegrityError, CompactionGuardError
from compaction_guard.guard import Guard
from compaction_guard.integrations.openai_agents import GuardedSession
from compaction_guard.render import find_blocks
from compaction_guard.taxonomy import Kind, Mode
from stubs import INV_BUDGET, Message


def drive(coro: Coroutine[Any, Any, None]) -> None:
    """Run a coroutine that never actually suspends.

    The suite's no-network fixture blocks the socketpair asyncio.run builds
    for its event loop, and the fakes complete every await synchronously, so
    a bare trampoline is both sufficient and proof that the adapter did no
    hidden I/O.
    """
    try:
        coro.send(None)
    except StopIteration:
        return
    raise AssertionError("fake session coroutines must complete synchronously")


class FakeSession:
    """The slice of the Agents SDK Session protocol the wrapper touches."""

    def __init__(self) -> None:
        self.items: list[Message] = []
        self.session_id = "fake-session"

    async def get_items(self, limit: int | None = None) -> list[Message]:
        if limit is None:
            return list(self.items)
        return list(self.items[-limit:])

    async def add_items(self, items: list[Message]) -> None:
        self.items.extend(items)

    async def pop_item(self) -> Message:
        return self.items.pop()

    async def clear_session(self) -> None:
        self.items.clear()


def _summariser(items: list[Any]) -> list[Any]:
    return [{"role": "user", "content": "Summary: the run continues under its cap."}]


def _blocks_in(items: list[Message]) -> int:
    return sum(
        len(find_blocks(item["content"]))
        for item in items
        if isinstance(item.get("content"), str)
    )


def test_owned_trigger_compacts_and_injects_verified_block() -> None:
    async def run() -> None:
        inner = FakeSession()
        guard: Guard[list[Any]] = Guard([INV_BUDGET])
        session = GuardedSession(inner, guard, compactor=_summariser, trigger_tokens=10)
        await session.add_items(
            [{"role": "user", "content": "enough text to pass ten estimated tokens easily"}]
        )
        report = guard.last_report
        assert report is not None
        assert report.mode is Mode.OWNED
        assert report.repaired is True
        items = await session.get_items()
        assert _blocks_in(items) == 1
        guard.assert_present(items)

    drive(run())


def test_owned_compact_now_and_below_trigger_no_compaction() -> None:
    async def run() -> None:
        inner = FakeSession()
        guard: Guard[list[Any]] = Guard([INV_BUDGET])
        session = GuardedSession(
            inner, guard, compactor=_summariser, trigger_tokens=10_000
        )
        await session.add_items([{"role": "user", "content": "small"}])
        assert guard.last_report is None, "below the trigger nothing compacts"
        report = await session.compact_now()
        assert report is not None and report.repaired
        assert _blocks_in(await session.get_items()) == 1

    drive(run())


def test_owned_full_fetch_catches_downstream_trim() -> None:
    async def run() -> None:
        inner = FakeSession()
        guard: Guard[list[Any]] = Guard([INV_BUDGET])
        session = GuardedSession(inner, guard, compactor=_summariser, trigger_tokens=5)
        await session.add_items([{"role": "user", "content": "text long enough to trigger"}])
        # Host code edits the store behind the wrapper.
        inner.items = [
            item
            for item in inner.items
            if not (
                isinstance(item.get("content"), str)
                and find_blocks(item["content"])
            )
        ]
        with pytest.raises(BlockIntegrityError):
            await session.get_items()

    drive(run())


def test_unobserved_seeds_quietly_then_reasserts_loudly() -> None:
    async def run() -> None:
        inner = FakeSession()
        reports: list[Any] = []
        guard: Guard[list[Any]] = Guard([INV_BUDGET], on_report=reports.append)
        session = GuardedSession(inner, guard)  # no compactor: UNOBSERVED
        await session.add_items([{"role": "user", "content": "turn one"}])

        # First full fetch: no block anywhere, so one is seeded, silently
        # (nothing was compacted yet, there is nothing to report).
        items = await session.get_items()
        assert _blocks_in(items) == 1
        assert reports == []

        # An opaque server-side compaction replaces history, block gone.
        inner.items = [{"role": "user", "content": "[opaque compacted summary]"}]
        items = await session.get_items()
        assert _blocks_in(items) == 1
        assert len(reports) == 1
        report = reports[0]
        assert report.mode is Mode.UNOBSERVED
        assert report.repaired is True
        assert all(f.kind is Kind.UNVERIFIABLE for f in report.findings)
        assert all(f.decided_by == "integration.unobserved" for f in report.findings)

    drive(run())


def test_unobserved_limited_fetch_is_untouched() -> None:
    async def run() -> None:
        inner = FakeSession()
        guard: Guard[list[Any]] = Guard([INV_BUDGET])
        session = GuardedSession(inner, guard)
        await session.add_items([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
        window = await session.get_items(limit=1)
        assert window == [{"role": "user", "content": "b"}]
        assert _blocks_in(inner.items) == 0, "a partial view must not trigger injection"

    drive(run())


def test_compact_now_refuses_without_compactor() -> None:
    async def run() -> None:
        session = GuardedSession(FakeSession(), Guard([INV_BUDGET]))
        with pytest.raises(CompactionGuardError, match="requires a compactor"):
            await session.compact_now()

    drive(run())


def test_clear_session_resets_the_owed_block() -> None:
    async def run() -> None:
        inner = FakeSession()
        guard: Guard[list[Any]] = Guard([INV_BUDGET])
        session = GuardedSession(inner, guard, compactor=_summariser, trigger_tokens=5)
        await session.add_items([{"role": "user", "content": "long enough to trigger now"}])
        await session.clear_session()
        assert await session.get_items() == []
        assert inner.items == []

    drive(run())


def test_pop_item_forwards_untouched() -> None:
    async def run() -> None:
        inner = FakeSession()
        guard: Guard[list[Any]] = Guard([INV_BUDGET])
        session = GuardedSession(inner, guard)
        await session.add_items([{"role": "user", "content": "last"}])
        popped = await session.pop_item()
        assert popped == {"role": "user", "content": "last"}
        assert inner.items == []

    drive(run())
