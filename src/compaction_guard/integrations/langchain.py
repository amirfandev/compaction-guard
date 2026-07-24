"""LangChain 1.0: OWNED wrapping of the summarization middleware.

``guard_middleware(inner, guard)`` returns an ``AgentMiddleware`` that owns
the compaction boundary. It snapshots the messages entering
``before_model``, lets the inner ``SummarizationMiddleware`` run,
reconstructs the message list its update produces, and pushes both sides
through ``Guard.compact``. That earns Mode OWNED, the strongest claim in
the library: diff-attributed detection, policy application, and a
checksum-verified sentinel block inside the update the agent actually
applies. On every non-compacting pass after the first injection,
``assert_present`` runs against the incoming messages, so anything
downstream that trimmed the block is caught at the next model call rather
than never.

What this wrapper cannot do, stated up front. It sees only the sync
``before_model`` hook: a summariser invoked outside this middleware, or an
inner middleware whose async ``abefore_model`` bypasses the sync hook, is
out of reach. It refuses loudly (``CodecError``) on message updates it
cannot reconstruct rather than guessing at reducer semantics. And
constraint text living only in typed tool-call attributes is invisible to
the renderer, which errs toward false alarms, never false certification.

This module is gated at import: without the ``langchain`` package it
raises ``MissingIntegrationError`` naming the extra, because nothing here
works without framework types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import Parameter, signature
from typing import Any

from ..context import AutoCodec, RenderedContext, _strip_message
from ..errors import CodecError
from ..guard import Guard
from ..render import strip_blocks
from . import MissingIntegrationError, _require

_require("langchain", extra="langchain")

__all__ = ["LangChainMessageCodec", "guard_middleware"]


def _message_types() -> tuple[Any, Any]:
    """(HumanMessage, RemoveMessage), imported on first use."""
    try:
        from langchain_core.messages import HumanMessage, RemoveMessage
    except ImportError as exc:
        raise MissingIntegrationError(
            "this integration needs langchain-core>=1.0 for message types "
            f"({exc}). Install the extra: pip install 'compaction-guard[langchain]'"
        ) from exc
    return HumanMessage, RemoveMessage


def _remove_all_marker() -> str:
    """The id value marking a full-replacement messages update.

    Imported from langgraph when present; the literal fallback is the
    constant's stable wire value, kept so this module never needs langgraph
    installed just to compare a string.
    """
    try:
        from langgraph.graph.message import (  # type: ignore[import-not-found]
            REMOVE_ALL_MESSAGES,
        )
    except ImportError:
        return "__remove_all__"
    return str(REMOVE_ALL_MESSAGES)


def _strip_content_blocks(content: Sequence[Any]) -> tuple[list[Any], bool]:
    """Strip sentinel text from a content-block list; (blocks, changed)."""
    blocks: list[Any] = []
    changed = False
    for item in content:
        if isinstance(item, str):
            stripped = strip_blocks(item)
            if stripped != item:
                changed = True
                if stripped.strip():
                    blocks.append(stripped)
                continue
        elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
            stripped = strip_blocks(item["text"])
            if stripped != item["text"]:
                changed = True
                if stripped.strip():
                    replaced = dict(item)
                    replaced["text"] = stripped
                    blocks.append(replaced)
                continue
        blocks.append(item)
    return blocks, changed


class LangChainMessageCodec:
    """A ``ContextCodec`` for LangChain message lists.

    Rendering delegates to ``AutoCodec``: LangChain messages are objects
    with ``.content``, and it already renders those per message with
    skipped-block accounting. Injection is the part ``AutoCodec`` refuses
    for typed objects and this codec exists to provide: stale sentinel
    blocks are stripped from message content (through ``model_copy``,
    never mutation), messages that carried nothing but a stale block are
    dropped, and one fresh ``HumanMessage`` carrier is appended. Human
    role, because the user slot is the one placement chat APIs accept
    mid-conversation.
    """

    def __init__(self) -> None:
        self._auto = AutoCodec()

    def render(self, context: Any) -> str:
        """Flat text for the detectors, via ``AutoCodec``'s object rules."""
        return self._auto.render(context)

    def render_details(self, context: Any) -> RenderedContext:
        """Per-message segments for region attribution, via ``AutoCodec``."""
        return self._auto.render_details(context)

    def inject(self, context: Any, block: str) -> Any:
        """A new message list carrying exactly one current sentinel block."""
        if isinstance(context, str):
            return self._auto.inject(context, block)
        if not isinstance(context, list):
            raise CodecError(
                f"cannot inject into context shape {type(context).__name__}: "
                "this codec writes LangChain message lists and str"
            )
        result: list[Any] = []
        for item in context:
            kept = self._strip_item(item)
            if kept is not None:
                result.append(kept)
        human_message, _ = _message_types()
        result.append(human_message(content=block))
        return result

    def _strip_item(self, item: Any) -> Any | None:
        if isinstance(item, str):
            stripped = strip_blocks(item)
            if stripped == item:
                return item
            return stripped if stripped.strip() else None
        if isinstance(item, Mapping):
            return _strip_message(item)
        return self._strip_typed(item)

    def _strip_typed(self, message: Any) -> Any | None:
        """Strip sentinel blocks from one typed message; None drops it.

        Mirrors the dict rules in ``context._strip_message``: drop only a
        message that stripping emptied, because that combination identifies
        the guard's own former carrier; keep a message with tool calls even
        when emptied so its results stay anchored.
        """
        content = getattr(message, "content", None)
        if isinstance(content, str):
            stripped = strip_blocks(content)
            if stripped == content:
                return message
            if stripped.strip() or getattr(message, "tool_calls", None):
                return self._copy_with_content(message, stripped)
            return None
        if isinstance(content, (list, tuple)):
            blocks, changed = _strip_content_blocks(content)
            if not changed:
                return message
            if blocks or getattr(message, "tool_calls", None):
                return self._copy_with_content(message, blocks)
            return None
        return message

    def _copy_with_content(self, message: Any, content: Any) -> Any:
        copy_fn = getattr(message, "model_copy", None)
        if not callable(copy_fn):
            raise CodecError(
                f"cannot rewrite a {type(message).__name__}: it offers no "
                "model_copy, and mutating a host message in place is banned"
            )
        return copy_fn(update={"content": content})


def _ensure_message_codec(guard: Guard[Any]) -> None:
    """Install the message-list codec when the guard still has the default.

    ``AutoCodec`` renders LangChain messages but refuses to inject into
    them, so a guard left on the default would hard-fail at its first
    REPAIR. Swapping the codec once at wrap time keeps adoption at one
    line; a caller who supplied a custom codec knows their context shape
    better than this module does and keeps it.
    """
    if isinstance(guard._codec, AutoCodec):
        guard._codec = LangChainMessageCodec()


def _accepts_runtime(hook: Any) -> bool:
    """Whether the inner hook takes (state, runtime) or state alone.

    LangChain 1.0 passes both. The inspection happens once at wrap time so
    the per-call path never pays for it, and anything uninspectable is
    assumed to be the current two-argument shape.
    """
    try:
        sig = signature(hook)
    except (TypeError, ValueError):
        return True
    params = list(sig.parameters.values())
    if any(p.kind is Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _state_messages(state: Any) -> list[Any]:
    """Snapshot the message list before the inner middleware runs.

    A copy, because ``Guard.compact`` renders its before side when called,
    which is after the inner middleware has already produced its update;
    holding our own list keeps the diff honest against list mutation.
    """
    if isinstance(state, Mapping):
        return list(state["messages"])
    return list(state.messages)


def _split_replacement(update_messages: Any) -> tuple[Any, tuple[Any, ...]]:
    """Interpret a messages update as (remove-all marker, replacement list).

    ``SummarizationMiddleware`` compacts by emitting a full replacement:
    ``RemoveMessage(REMOVE_ALL_MESSAGES)`` followed by the new history.
    Anything else cannot be reconstructed without re-implementing the
    reducer, and verifying a context the agent never holds would certify
    against fiction, so the wrapper refuses instead of guessing.
    """
    _, remove_message = _message_types()
    items = list(update_messages) if isinstance(update_messages, (list, tuple)) else []
    if items:
        first = items[0]
        marker = _remove_all_marker()
        if isinstance(first, remove_message) and getattr(first, "id", None) == marker:
            return first, tuple(items[1:])
    raise CodecError(
        "cannot interpret this messages update: expected a full-replacement "
        "update whose first element is RemoveMessage(REMOVE_ALL_MESSAGES), "
        "the shape SummarizationMiddleware emits when it compacts. The "
        "wrapper refuses to guess at reducer semantics."
    )


def guard_middleware(inner: Any, guard: Guard[Any]) -> Any:
    """Wrap a summarization middleware so its compactions run through the guard.

    Usage, next to the middleware the host already has::

        middleware = guard_middleware(
            SummarizationMiddleware(model=..., max_tokens_before_summary=...),
            guard,
        )
        agent = create_agent(model, tools, middleware=[middleware])

    Policy applies exactly as at any other guard boundary. Under RAISE, a
    gating finding raises ``InvariantViolation`` out of ``before_model``
    before the update is applied, so the agent state still holds the
    uncompacted history. Under WARN the inner update passes through
    untouched and only the report records what the summariser did. If the
    guard was built with the default codec, the message-list codec is
    installed on it here; the guard is then dedicated to this agent's
    context shape.
    """
    hook = getattr(inner, "before_model", None)
    if not callable(hook):
        raise TypeError(
            "guard_middleware wraps a middleware exposing before_model; "
            f"{type(inner).__name__} has none"
        )
    # Subclassing the real base rather than duck-typing matters: the agent
    # wires hooks by looking at what a middleware class overrides, and a
    # lookalike could register hooks this wrapper never implements.
    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError as exc:
        raise MissingIntegrationError(
            "guard_middleware needs langchain>=1.0 "
            f"(langchain.agents.middleware is unavailable: {exc}). "
            "Install the extra: pip install 'compaction-guard[langchain]'"
        ) from exc
    _ensure_message_codec(guard)
    pass_runtime = _accepts_runtime(hook)

    class GuardedSummarizationMiddleware(AgentMiddleware):
        """The wrapper. Holds no policy of its own; the guard decides."""

        def __init__(self) -> None:
            super().__init__()
            self._pinned = False
            # The agent reads these from the middleware object; forwarding
            # them keeps the inner middleware's state schema and tools
            # visible through the wrapper.
            for attr in ("state_schema", "tools"):
                if hasattr(inner, attr):
                    setattr(self, attr, getattr(inner, attr))

        def before_model(self, state: Any, runtime: Any = None) -> Any:
            messages = _state_messages(state)
            if pass_runtime:
                update = inner.before_model(state, runtime)
            else:
                update = inner.before_model(state)
            if not isinstance(update, Mapping) or "messages" not in update:
                # No compaction this pass. Once a block has been injected,
                # its absence or edit here means something downstream
                # trimmed guard-owned text: exactly what the per-turn
                # check exists to catch.
                if self._pinned:
                    guard.assert_present(messages)
                return update
            marker, compacted = _split_replacement(update["messages"])
            repaired = guard.compact(messages, lambda _before: list(compacted))
            report = guard.last_report
            if report is not None and report.repaired:
                self._pinned = True
            rebuilt = dict(update)
            rebuilt["messages"] = [marker, *repaired]
            return rebuilt

    return GuardedSummarizationMiddleware()
