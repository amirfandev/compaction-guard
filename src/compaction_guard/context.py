"""The ContextCodec protocol, AutoCodec, and the per-shape render/inject rules.

Context is never a guard-owned type. The guard is generic over the user's
context and touches it only through a two-method codec: ``render`` produces
the text detectors examine, ``inject`` writes the sentinel block back. Two
methods is the whole protocol on purpose: anything richer (segmentation,
metadata, roles) would force every custom codec to answer questions most
context shapes cannot answer.

``AutoCodec`` recognises the common shapes without importing any framework:

- ``str``: the text is the context.
- ``list[str]``: each element is one message.
- ``list[dict]``: chat-style messages. String ``content`` renders as is;
  content-block lists render their text blocks and count everything
  unrenderable (images, audio, unknown block types) as skipped, so the guard
  can say in the report what it could not read. Tool-call structs render
  their string arguments and string results; the constraint you registered
  may well survive only inside a retained tool message.
- objects with ``.content``: duck-typed messages (LangChain-shaped, or any
  host type). Rendered, never constructed: ``inject`` on a list of typed
  objects refuses, because building a message of a foreign type means either
  importing its framework or guessing its constructor, and both are banned.

Everything else is a refusal, not a guess. ``render`` raises ``CodecError``
for a shape it does not recognise, the guard turns that into UNVERIFIABLE
findings, and the run continues with the failure named in the report. The
design rule: no verdict stronger than the rendering supports. The one
mis-render this codec must never commit is rendering garbage while claiming
success, because every verdict downstream would then certify against text
the model never saw.

``inject`` replaces rather than appends: stale sentinel blocks are stripped
from every message first, messages that carried nothing but a stale block are
dropped, and exactly one fresh block is appended. That makes repair
idempotent and convergent no matter what the compactor did to the old block.

Stripping has a deliberate boundary: sentinel text is reclaimed from
``content`` fields only, because content is where the guard's own carrier
writes. Tool-call arguments are invocation records, what was actually sent
to a tool, and editing them would fabricate history and corrupt argument
JSON, so a marker-shaped string inside tool arguments is left alone even
though it renders as a block. Repeated injection still converges (nothing
accumulates); rendered output simply shows that lookalike alongside the
fresh block, which is the benign-duplication residue this library accepts
everywhere in exchange for never rewriting what actually happened.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from .errors import CodecError
from .render import strip_blocks

__all__ = ["AutoCodec", "ContextCodec", "RenderedContext"]

C = TypeVar("C")


class ContextCodec(Protocol[C]):
    """The guard's only view of a context: text out, block in.

    Either method may raise ``CodecError``. A ``render`` failure downgrades
    every invariant to UNVERIFIABLE, because nothing can be verified against
    text that could not be produced. An ``inject`` failure under REPAIR is a
    hard error: a guard that cannot write cannot pin.
    """

    def render(self, context: C) -> str:
        """Return the text the detectors examine."""
        ...

    def inject(self, context: C, block: str) -> C:
        """Return a new context with ``block`` present exactly once.

        Any existing sentinel block must be replaced, not accumulated, so
        repeated repair converges instead of stacking blocks.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedContext:
    """A rendering plus the structure the flat string throws away.

    ``segments`` holds one entry per message, in order, because region
    attribution in ``diff.py`` works by message digest and a joined string
    has no message boundaries left to digest. ``skipped_blocks`` counts the
    content blocks that had no text to render (images, audio, unknown
    types); the guard surfaces the count in the report so a summary verdict
    over partially rendered context is never silently presented as a verdict
    over all of it.

    This type is deliberately not part of ``ContextCodec``: custom codecs
    owe the guard only ``render`` and ``inject``. A codec that can offer
    segmentation exposes a ``render_details`` method with this return type,
    and the guard uses it when present, falling back to treating the whole
    rendering as one segment.
    """

    text: str
    segments: tuple[str, ...]
    skipped_blocks: int


# Keys that mark a dict as a recognisable message. A dict with none of these
# is refused rather than skipped: it may belong to a schema this codec has
# never seen, and skipping it would mean certifying verdicts against a
# rendering with unknown holes in it.
_MESSAGE_KEYS = ("role", "content", "tool_calls", "function_call", "text", "type")


def _render_tool_call(call: Mapping[str, Any]) -> tuple[str, int]:
    """Render the string arguments and results of one tool-call struct.

    Handles the OpenAI chat shape (``{"function": {"name", "arguments"}}``),
    the flat Responses shape (``{"name", "arguments"}`` or ``{"output"}``),
    and the Anthropic shape (``{"name", "input": {...}}``). Only string
    payloads render; a call whose arguments are all non-string counts as one
    skipped block, because inventing a serialisation here would put text in
    front of the detectors that no model was ever shown.
    """
    function = call.get("function")
    if isinstance(function, Mapping):
        return _render_tool_call(function)
    parts: list[str] = []
    name = call.get("name")
    if isinstance(name, str) and name:
        parts.append(name)
    for key in ("arguments", "input", "output"):
        value = call.get(key)
        if isinstance(value, str):
            if value:
                parts.append(value)
        elif isinstance(value, Mapping):
            parts.extend(v for v in value.values() if isinstance(v, str) and v)
    if parts:
        return " ".join(parts), 0
    return "", 1


def _render_block(block: Any) -> tuple[str, int]:
    """Render one content block to (text, skipped_count)."""
    if isinstance(block, str):
        return block, 0
    if isinstance(block, Mapping):
        text = block.get("text")
        if isinstance(text, str):
            return text, 0
        if "content" in block:
            return _render_content(block["content"])
        if any(key in block for key in ("function", "arguments", "input", "output")):
            return _render_tool_call(block)
        return "", 1
    return "", 1


def _render_content(value: Any) -> tuple[str, int]:
    """Render a message's ``content`` field: str, None, or a block list."""
    if value is None:
        return "", 0
    if isinstance(value, str):
        return value, 0
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        skipped = 0
        for block in value:
            rendered, missed = _render_block(block)
            skipped += missed
            if rendered:
                parts.append(rendered)
        return "\n".join(parts), skipped
    return "", 1


def _render_message(message: Mapping[str, Any], index: int) -> tuple[str, int]:
    """Render one dict-shaped message, refusing dicts that match no known schema."""
    if not any(key in message for key in _MESSAGE_KEYS):
        raise CodecError(
            f"unrecognised message shape at index {index}: dict with keys "
            f"{sorted(str(key) for key in message)} matches no known message schema"
        )
    if "type" in message and all(
        key not in message for key in ("content", "text", "tool_calls", "function_call")
    ):
        # A typed item with no chat payload, e.g. a Responses-style
        # {"type": "function_call", ...}. The block renderer knows the
        # tool-call shapes and counts anything else as skipped.
        return _render_block(message)
    parts: list[str] = []
    skipped = 0
    if "content" in message:
        rendered, missed = _render_content(message["content"])
        skipped += missed
        if rendered:
            parts.append(rendered)
    elif isinstance(message.get("text"), str):
        parts.append(message["text"])
    elif "text" in message:
        skipped += 1
    for key in ("tool_calls",):
        calls = message.get(key)
        if isinstance(calls, (list, tuple)):
            for call in calls:
                if isinstance(call, Mapping):
                    rendered, missed = _render_tool_call(call)
                    skipped += missed
                    if rendered:
                        parts.append(rendered)
                else:
                    skipped += 1
    legacy = message.get("function_call")
    if isinstance(legacy, Mapping):
        rendered, missed = _render_tool_call(legacy)
        skipped += missed
        if rendered:
            parts.append(rendered)
    return "\n".join(parts), skipped


def _render_object(obj: Any, index: int | None) -> tuple[str, int]:
    """Render a duck-typed message object via its ``content`` attribute."""
    content = getattr(obj, "content", None)
    if isinstance(content, str):
        return content, 0
    if isinstance(content, (list, tuple)):
        return _render_content(content)
    where = "" if index is None else f" at index {index}"
    raise CodecError(
        f"unrecognised message object{where}: {type(obj).__name__}.content is "
        f"{type(content).__name__}, expected str or a list of content blocks"
    )


def _strip_message(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Remove sentinel blocks from one message; None means drop the message.

    A message is dropped only when stripping changed it, nothing renderable
    remains, and it carries no tool calls: that combination identifies the
    guard's own previously injected carrier message. A message that was
    empty before stripping is not ours to delete, and an assistant message
    with tool calls must survive with emptied content or its tool results
    would be orphaned.
    """
    content = message.get("content")
    if isinstance(content, str):
        stripped = strip_blocks(content)
        if stripped == content:
            return message
        if stripped.strip() or "tool_calls" in message:
            replaced = dict(message)
            replaced["content"] = stripped
            return replaced
        return None
    if isinstance(content, (list, tuple)):
        blocks: list[Any] = []
        changed = False
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                stripped = strip_blocks(item["text"])
                if stripped != item["text"]:
                    changed = True
                    if stripped.strip():
                        replaced_block = dict(item)
                        replaced_block["text"] = stripped
                        blocks.append(replaced_block)
                    continue
            blocks.append(item)
        if not changed:
            return message
        if blocks or "tool_calls" in message:
            replaced = dict(message)
            replaced["content"] = blocks
            return replaced
        return None
    return message


class AutoCodec:
    """The default codec: recognises common shapes, refuses everything else.

    Stateless and reusable across guards. See the module docstring for the
    recognition table and the refusal rationale. Satisfies
    ``ContextCodec[Any]`` structurally; it is not exported from the package
    because the spec's public surface names only the protocol, but it is
    importable from this module and is the guard's default.
    """

    def render(self, context: Any) -> str:
        """Flat text for the detectors; raises ``CodecError`` on unknown shapes."""
        return self.render_details(context).text

    def render_details(self, context: Any) -> RenderedContext:
        """Render with per-message segments and the skipped-block count.

        The guard prefers this over ``render`` when the codec offers it:
        segments feed region attribution in ``diff.py``, and the skip count
        feeds the report note.
        """
        if isinstance(context, str):
            return RenderedContext(text=context, segments=(context,), skipped_blocks=0)
        if isinstance(context, list):
            segments: list[str] = []
            skipped = 0
            for index, item in enumerate(context):
                if isinstance(item, str):
                    segments.append(item)
                elif isinstance(item, Mapping):
                    rendered, missed = _render_message(item, index)
                    segments.append(rendered)
                    skipped += missed
                elif hasattr(item, "content"):
                    rendered, missed = _render_object(item, index)
                    segments.append(rendered)
                    skipped += missed
                else:
                    raise CodecError(
                        f"unrecognised message shape at index {index}: "
                        f"{type(item).__name__} is not a str, dict, or object "
                        "with a content attribute"
                    )
            text = "\n\n".join(segment for segment in segments if segment.strip())
            return RenderedContext(
                text=text, segments=tuple(segments), skipped_blocks=skipped
            )
        if isinstance(context, Mapping):
            raise CodecError(
                "unrecognised context shape: a bare dict is not a context; "
                "AutoCodec expects str, a list of messages, or an object with "
                "a string content attribute"
            )
        if hasattr(context, "content"):
            rendered, missed = _render_object(context, None)
            return RenderedContext(
                text=rendered, segments=(rendered,), skipped_blocks=missed
            )
        raise CodecError(
            f"unrecognised context shape: {type(context).__name__}; AutoCodec "
            "handles str, list[str], list[dict] chat messages, and objects with "
            "a string content attribute. Supply a ContextCodec for anything else."
        )

    def inject(self, context: Any, block: str) -> Any:
        """Return a new context carrying exactly one current sentinel block.

        Never mutates the input. Writable shapes are ``str`` and lists of
        ``str`` or dict messages; the injected carrier in a message list is
        ``{"role": "user", "content": block}`` because the user role is the
        one placement every chat API accepts mid-conversation (several
        reject mid-list system messages). Lists containing typed objects
        render fine but refuse injection: constructing a foreign message
        type without its framework would be a guess, and a wrong guess here
        corrupts the host's next API call.
        """
        if isinstance(context, str):
            stripped = strip_blocks(context)
            if stripped.strip():
                return stripped.rstrip("\n") + "\n\n" + block
            return block
        if isinstance(context, list):
            return self._inject_list(context, block)
        raise CodecError(
            f"cannot inject into context shape {type(context).__name__}: "
            "AutoCodec writes only str and list contexts. Supply a ContextCodec "
            "that knows how to write this shape."
        )

    def _inject_list(self, context: list[Any], block: str) -> list[Any]:
        if not context:
            raise CodecError(
                "cannot inject into an empty list: no element reveals whether "
                "this is list[str] or a message list, and appending the wrong "
                "shape would corrupt the host's API call"
            )
        has_messages = any(isinstance(item, Mapping) for item in context)
        result: list[Any] = []
        for index, item in enumerate(context):
            if isinstance(item, str):
                stripped = strip_blocks(item)
                if stripped == item:
                    result.append(item)
                elif stripped.strip():
                    result.append(stripped)
                # A string that was nothing but a stale block disappears;
                # the fresh block is appended once, below.
            elif isinstance(item, Mapping):
                kept = _strip_message(item)
                if kept is not None:
                    result.append(kept)
            else:
                raise CodecError(
                    f"cannot inject into a list containing {type(item).__name__} "
                    f"at index {index}: building a message of that type would "
                    "require importing its framework. Supply a ContextCodec or "
                    "use an integration module."
                )
        if has_messages:
            result.append({"role": "user", "content": block})
        else:
            result.append(block)
        return result
