"""The codec matrix: every context shape through render and inject.

Fixture-driven: each JSON file under fixtures/codecs is one shape. Renderable
shapes must render their text (and count what they cannot render); every
writable shape must take an injection and come back with exactly one verified
block; refused shapes must raise CodecError from render, turn into
UNVERIFIABLE findings through the guard, and hard-fail injection under
REPAIR. Duck-typed objects and the empty-list refusal are covered in code
because JSON cannot express them.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from compaction_guard.context import AutoCodec
from compaction_guard.errors import CodecError
from compaction_guard.guard import Guard
from compaction_guard.invariant import Invariant
from compaction_guard.render import expected_checksum, find_blocks, render_block
from compaction_guard.taxonomy import Kind, Policy
from conftest import load_fixture_cases
from stubs import INV_BUDGET, INV_DB, base_messages

CASES = load_fixture_cases("codecs")
RENDERABLE = [c for c in CASES if c[1]["renderable"]]
REFUSED = [c for c in CASES if not c[1]["renderable"]]

BLOCK = render_block([Invariant.parse(INV_DB, id="cafecafecafe")])
CHECKSUM = expected_checksum([Invariant.parse(INV_DB, id="cafecafecafe")])


@pytest.mark.parametrize(("name", "case"), RENDERABLE, ids=[c[0] for c in RENDERABLE])
def test_render_known_shapes(name: str, case: dict[str, Any]) -> None:
    codec = AutoCodec()
    details = codec.render_details(case["context"])
    for expected in case["expect_substrings"]:
        assert expected in details.text, f"{expected!r} missing from rendering"
    assert details.skipped_blocks == case["skipped_blocks"]
    assert codec.render(case["context"]) == details.text


@pytest.mark.parametrize(
    ("name", "case"),
    [c for c in RENDERABLE if c[1]["injectable"]],
    ids=[c[0] for c in RENDERABLE if c[1]["injectable"]],
)
def test_inject_lands_one_verified_block(name: str, case: dict[str, Any]) -> None:
    codec = AutoCodec()
    context = case["context"]
    original = copy.deepcopy(context)
    injected = codec.inject(context, BLOCK)
    assert context == original, "inject must never mutate its input"
    rendered = codec.render(injected)
    blocks = find_blocks(rendered)
    assert len(blocks) == 1
    blocks[0].verify()
    assert blocks[0].header_checksum == CHECKSUM
    for expected in case["expect_substrings"]:
        assert expected in rendered, "injection must not destroy existing content"


@pytest.mark.parametrize(
    ("name", "case"),
    [c for c in RENDERABLE if c[1]["injectable"]],
    ids=[c[0] for c in RENDERABLE if c[1]["injectable"]],
)
def test_inject_idempotent(name: str, case: dict[str, Any]) -> None:
    """Injecting twice converges to one block; repair never accumulates."""
    codec = AutoCodec()
    once = codec.inject(case["context"], BLOCK)
    twice = codec.inject(once, BLOCK)
    assert len(find_blocks(codec.render(twice))) == 1


@pytest.mark.parametrize(("name", "case"), REFUSED, ids=[c[0] for c in REFUSED])
def test_unrecognised_shapes_refuse_render(name: str, case: dict[str, Any]) -> None:
    codec = AutoCodec()
    with pytest.raises(CodecError, match="unrecognised"):
        codec.render(case["context"])


@pytest.mark.parametrize(("name", "case"), REFUSED, ids=[c[0] for c in REFUSED])
def test_unrecognised_output_yields_unverifiable_never_a_guess(
    name: str, case: dict[str, Any]
) -> None:
    """A compactor emitting an unreadable shape gets UNVERIFIABLE findings,

    with the codec failure in the note, and never any stronger verdict.
    """
    guard: Guard[Any] = Guard([INV_DB, INV_BUDGET], policy=Policy.WARN)
    result = guard.compact(base_messages(), lambda messages: case["context"])
    assert result == case["context"]
    report = guard.last_report
    assert report is not None
    assert len(report.findings) == 2
    assert all(f.kind is Kind.UNVERIFIABLE for f in report.findings)
    assert all(f.decided_by == "codec.render" for f in report.findings)
    assert report.note is not None and "codec render failed" in report.note


def test_uninjectable_output_under_repair_is_hard_codec_error() -> None:
    """REPAIR cannot pin what it cannot write; that is a loud failure, not a report."""
    guard: Guard[Any] = Guard([INV_DB], policy=Policy.REPAIR)
    unwritable = {"payload": [1, 2, 3], "meta": "opaque"}
    with pytest.raises(CodecError, match="cannot inject"):
        guard.compact(base_messages(), lambda messages: unwritable)


class _TypedMessage:
    """A duck-typed foreign message: renderable via .content, never constructed."""

    def __init__(self, content: str) -> None:
        self.content = content


def test_duck_typed_objects_render() -> None:
    codec = AutoCodec()
    context = [_TypedMessage("The database orders_prod is production.")]
    assert "orders_prod" in codec.render(context)


def test_duck_typed_objects_refuse_injection() -> None:
    """Building a message of a foreign type would mean guessing a constructor."""
    codec = AutoCodec()
    with pytest.raises(CodecError, match="cannot inject"):
        codec.inject([_TypedMessage("text")], BLOCK)


def test_typed_object_with_non_string_content_refuses() -> None:
    class Broken:
        content = 42

    with pytest.raises(CodecError, match="unrecognised message object"):
        AutoCodec().render([Broken()])


def test_empty_list_refuses_injection() -> None:
    """No element reveals the list's shape; a wrong guess corrupts the API call."""
    with pytest.raises(CodecError, match="empty list"):
        AutoCodec().inject([], BLOCK)


def test_inject_replaces_stale_blocks() -> None:
    """Repair strips whatever blocks exist and writes exactly one fresh one."""
    codec = AutoCodec()
    stale_block = render_block([Invariant.parse(INV_BUDGET, id="0ddba11c0de0")])
    context = [
        {"role": "user", "content": "Real prose."},
        {"role": "user", "content": stale_block},
    ]
    injected = codec.inject(context, BLOCK)
    rendered = codec.render(injected)
    blocks = find_blocks(rendered)
    assert len(blocks) == 1
    assert blocks[0].header_checksum == CHECKSUM
    assert "Real prose." in rendered


def test_inject_drops_only_the_guards_own_carrier() -> None:
    """A message that was nothing but a stale block disappears; a message that

    also carried prose keeps its prose; originally empty messages survive.
    """
    codec = AutoCodec()
    stale_block = render_block([Invariant.parse(INV_BUDGET, id="0ddba11c0de0")])
    context = [
        {"role": "user", "content": ""},
        {"role": "user", "content": f"kept prose\n{stale_block}"},
        {"role": "user", "content": stale_block},
    ]
    injected = codec.inject(context, BLOCK)
    assert {"role": "user", "content": ""} in injected
    assert any(
        isinstance(m.get("content"), str) and "kept prose" in m["content"] for m in injected
    )
    assert len(injected) == 3  # empty + prose + fresh carrier; bare stale carrier gone


def test_str_inject_round_trip() -> None:
    codec = AutoCodec()
    injected = codec.inject("prose context", BLOCK)
    assert isinstance(injected, str)
    assert "prose context" in injected
    assert len(find_blocks(injected)) == 1
