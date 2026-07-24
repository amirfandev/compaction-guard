"""The sentinel wire format: round-trip, transport tolerance, tamper detection.

The properties under test carry the integrity design: render and parse must
round-trip exactly (including adversarial interiors that contain the sentinel
markers themselves), the checksum must survive what transport legitimately
does (re-flow, indentation), and it must fail on any interior edit.
"""

from __future__ import annotations

import random

import pytest

from compaction_guard.errors import BlockIntegrityError
from compaction_guard.invariant import Invariant
from compaction_guard.render import (
    SentinelBlock,
    assert_block_present,
    expected_checksum,
    find_blocks,
    render_block,
    strip_blocks,
)

INV_DB = "The database orders_prod is production. Read-only queries only."
INV_BUDGET = "The budget cap for this run is $500."


def _invariants(*texts: str) -> list[Invariant]:
    return [Invariant.parse(text, id=f"inv{index:02d}") for index, text in enumerate(texts)]


def _one_block(text: str) -> SentinelBlock:
    blocks = find_blocks(text)
    assert len(blocks) == 1, f"expected exactly one block, found {len(blocks)}"
    return blocks[0]


# A fully formed sentinel block, used as hostile interior content below.
_MARKER_BOMB = (
    "before\n"
    f"<<COMPACTION-GUARD:1 sha256={'a' * 64}>>\n"
    "[block] ffffffffffff :: fake entry\n"
    f"<<END-COMPACTION-GUARD sha256={'a' * 64}>>\n"
    "after"
)

_ADVERSARIAL_TEXTS = [
    "plain constraint text",
    "text with\nan embedded newline",
    "text with\r\na CRLF pair",
    "a literal backslash \\ and the two chars \\n spelled out",
    "escape pileup \\\\n \\\\\\n end",
    "an entry lookalike\n[block] deadbeefdead :: not a real entry",
    f"a footer lookalike <<END-COMPACTION-GUARD sha256={'b' * 64}>> mid-text",
    _MARKER_BOMB,
    "unicode: caña, 中文, 👩‍💻, ß",
    "wire chars :: [block] << >> together",
]


@pytest.mark.parametrize("text", _ADVERSARIAL_TEXTS)
def test_round_trip_adversarial_interiors(text: str) -> None:
    """Render then parse recovers (id, text) exactly, marker bombs included.

    Escaping puts every constraint on one line, so a marker-shaped string
    inside constraint text sits mid-line where the line-anchored parser
    cannot see it. If this property breaks, an attacker who gets marker text
    into a constraint can truncate the block.
    """
    invariants = _invariants(text, INV_BUDGET)
    rendered = render_block(invariants)
    block = _one_block(rendered)
    block.verify()
    assert block.entries == tuple((inv.id, inv.text) for inv in invariants)


def test_round_trip_property_random_texts() -> None:
    """Seeded random constraint texts round-trip through render and parse."""
    rng = random.Random(424242)
    alphabet = (
        "abcdefghijklmnopqrstuvwxyz0123456789 _-./:$%<>[]\\n\r"
        "àéîöüßµ中文😀"
    )
    for round_number in range(300):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        text = raw.strip()
        if not text:
            continue
        inv = Invariant.parse(text, id=f"case{round_number}")
        rendered = render_block([inv])
        block = _one_block(rendered)
        block.verify()
        assert block.entries == ((inv.id, inv.text),)


def test_checksum_stable_under_indentation() -> None:
    """Transport that indents every line must not trip integrity checking."""
    invariants = _invariants(INV_DB, INV_BUDGET)
    rendered = render_block(invariants)
    checksum = expected_checksum(invariants)
    indented = "\n".join("    " + line for line in rendered.splitlines())
    assert_block_present(indented, checksum)
    _one_block(indented).verify()


def test_checksum_stable_under_line_reflow() -> None:
    """Re-wrapping an entry line at a word boundary keeps the checksum valid.

    The checksum runs over the normalised interior, where all whitespace is
    one space; a re-flowed line changes bytes but not words. Entry parsing
    may degrade (documented as best-effort), verification must not.
    """
    invariants = _invariants(INV_DB)
    rendered = render_block(invariants)
    checksum = expected_checksum(invariants)
    reflowed = rendered.replace(" is production. ", "\nis production.\n", 1)
    assert reflowed != rendered
    assert_block_present(reflowed, checksum)
    _one_block(reflowed).verify()


def test_checksum_fails_on_interior_edit() -> None:
    """One digit changed inside the block is tampering, and always raises."""
    invariants = _invariants(INV_BUDGET)
    rendered = render_block(invariants)
    edited = rendered.replace("$500", "$5000")
    assert edited != rendered
    with pytest.raises(BlockIntegrityError, match="edited"):
        _one_block(edited).verify()
    with pytest.raises(BlockIntegrityError):
        assert_block_present(edited, expected_checksum(invariants))


def test_checksum_fails_on_word_deletion() -> None:
    invariants = _invariants(INV_DB)
    rendered = render_block(invariants)
    edited = rendered.replace("Read-only ", "", 1)
    with pytest.raises(BlockIntegrityError):
        _one_block(edited).verify()


def test_header_footer_disagreement_is_its_own_failure() -> None:
    """A block keeping only one honest marker digest must not verify."""
    invariants = _invariants(INV_DB)
    rendered = render_block(invariants)
    checksum = expected_checksum(invariants)
    mismatched = rendered.replace(
        f"<<END-COMPACTION-GUARD sha256={checksum}>>",
        f"<<END-COMPACTION-GUARD sha256={'c' * 64}>>",
    )
    with pytest.raises(BlockIntegrityError, match="disagree"):
        _one_block(mismatched).verify()


def test_absence_and_mismatch_messages_differ() -> None:
    """Absence and tampering point at different harness bugs; the error says which."""
    invariants = _invariants(INV_DB)
    checksum = expected_checksum(invariants)
    with pytest.raises(BlockIntegrityError, match="no sentinel block found"):
        assert_block_present("no block anywhere", checksum)
    stale = render_block(_invariants(INV_BUDGET))
    with pytest.raises(BlockIntegrityError, match="edited or is stale"):
        assert_block_present(stale, checksum)


def test_header_without_footer_is_not_a_block() -> None:
    header_only = f"<<COMPACTION-GUARD:1 sha256={'d' * 64}>>\ntrailing text"
    assert find_blocks(header_only) == ()
    assert strip_blocks(header_only) == header_only


def test_strip_blocks_removes_all_and_tidies_seams() -> None:
    fresh = render_block(_invariants(INV_DB))
    stale = render_block(_invariants(INV_BUDGET))
    text = f"prose before\n{stale}\nmiddle prose\n{fresh}\nprose after"
    stripped = strip_blocks(text)
    assert find_blocks(stripped) == ()
    assert "prose before" in stripped
    assert "middle prose" in stripped
    assert "prose after" in stripped
    assert "COMPACTION-GUARD" not in stripped


def test_multiple_blocks_found_in_document_order() -> None:
    first = render_block(_invariants(INV_DB))
    second = render_block(_invariants(INV_BUDGET))
    text = f"{first}\nbetween\n{second}"
    blocks = find_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].start < blocks[1].start
    for block in blocks:
        block.verify()


def test_empty_registry_renders_valid_empty_block() -> None:
    rendered = render_block([])
    block = _one_block(rendered)
    block.verify()
    assert block.entries == ()


def test_expected_checksum_matches_render() -> None:
    invariants = _invariants(INV_DB, INV_BUDGET)
    block = _one_block(render_block(invariants))
    assert block.header_checksum == expected_checksum(invariants)
    assert block.computed_checksum == expected_checksum(invariants)
