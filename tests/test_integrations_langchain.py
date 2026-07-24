"""The LangChain adapter. Core CI covers the import gate; the wrapped
middleware itself runs only where the ``langchain`` extra is installed (its
own CI job), against a minimal middleware-shaped fake, because everything in
the module needs framework message types.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import pytest

from compaction_guard.guard import Guard
from compaction_guard.render import expected_checksum, find_blocks
from compaction_guard.taxonomy import Kind, Mode
from stubs import INV_BUDGET

_HAS_LANGCHAIN = find_spec("langchain") is not None


@pytest.mark.skipif(_HAS_LANGCHAIN, reason="langchain installed; gate cannot fire")
def test_import_gate_names_the_extra() -> None:
    from compaction_guard.integrations import MissingIntegrationError

    with pytest.raises(MissingIntegrationError, match=r"compaction-guard\[langchain\]"):
        import compaction_guard.integrations.langchain  # noqa: F401
    # The gate must also satisfy except ImportError probing by host code.
    assert issubclass(MissingIntegrationError, ImportError)


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="needs the langchain extra")
def test_guard_middleware_owns_the_compaction() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    from compaction_guard.integrations.langchain import guard_middleware

    try:
        from langgraph.graph.message import REMOVE_ALL_MESSAGES

        marker = str(REMOVE_ALL_MESSAGES)
    except ImportError:
        marker = "__remove_all__"

    summary = AIMessage(content="Summary: the budget cap for this run is $5000.")

    class FakeSummarization:
        """Middleware-shaped: compacts by emitting a full replacement."""

        def before_model(self, state: Any, runtime: Any = None) -> Any:
            return {"messages": [RemoveMessage(id=marker), summary]}

    guard: Guard[Any] = Guard([INV_BUDGET])
    wrapped = guard_middleware(FakeSummarization(), guard)
    state = {
        "messages": [
            HumanMessage(content=INV_BUDGET),
            HumanMessage(content="chatter about the run"),
        ]
    }
    update = wrapped.before_model(state, None)
    report = guard.last_report
    assert report is not None
    assert report.mode is Mode.OWNED
    assert report.repaired is True
    assert Kind.MUTATED in {f.kind for f in report.findings}
    messages = update["messages"]
    assert isinstance(messages[0], RemoveMessage)
    rendered = "\n".join(
        m.content for m in messages[1:] if isinstance(m.content, str)
    )
    blocks = find_blocks(rendered)
    assert len(blocks) == 1
    assert blocks[0].header_checksum == expected_checksum(guard.invariants())


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="needs the langchain extra")
def test_non_compacting_pass_returns_update_untouched() -> None:
    from langchain_core.messages import HumanMessage

    from compaction_guard.integrations.langchain import guard_middleware

    class Passive:
        def before_model(self, state: Any, runtime: Any = None) -> Any:
            return None

    guard: Guard[Any] = Guard([INV_BUDGET])
    wrapped = guard_middleware(Passive(), guard)
    state = {"messages": [HumanMessage(content=INV_BUDGET)]}
    assert wrapped.before_model(state, None) is None
    assert guard.last_report is None
