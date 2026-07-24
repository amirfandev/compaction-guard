"""The sentinel wire format: render, parse, checksum. In exactly one file.

The block looks like this in context::

    <<COMPACTION-GUARD:1 sha256=3f9ac21e...>>
    [block] a1b2c3d4e5f6 :: The database orders_prod is production. Read-only queries only.
    [block] 9f8e7d6c5b4a :: The budget cap for this run is $500.
    <<END-COMPACTION-GUARD sha256=3f9ac21e...>>

The checksum is sha256 over the interior after ``normalize.normalize()``, and
that choice carries the whole integrity design. Normalising first means the
checksum survives what transport legitimately does to text (line re-flow,
indentation changes, whitespace mangling) while still catching what nothing
legitimate does (edits to the words, ids, or values). A checksum over raw
bytes would scream on every re-wrapped line; no checksum at all would let a
"helpful" summariser trim the block silently. The header and footer both
carry the digest so a truncated block cannot pass by keeping only one marker.

Invariant text is escaped onto a single line (newline, carriage return,
backslash, and the escape mark itself) so that render and parse round-trip
exactly and line-anchored parsing cannot be confused by marker-shaped strings
inside constraint text: a footer lookalike embedded in an invariant sits
mid-line, and markers only match at line starts. The escape mark is U+00A6
(broken bar), not the conventional backslash, for a reason that goes to the
checksum's soundness: ``normalize`` deletes punctuation, so a backslash-built
escape sequence would collapse to the same normalised bytes as its unescaped
lookalike ("line one\\ntwo" versus "line one ntwo"), and two distinct
invariant texts would share one digest. U+00A6 is a symbol, which
``normalize`` keeps, so escape structure survives into the digest and that
equivalence cannot be forged.

A checksum mismatch is always an exception, never a finding. Findings
classify summariser behaviour; an edited sentinel block after the guard's own
repair means something rewrote guard-owned bytes, which is a harness bug.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from .errors import BlockIntegrityError
from .normalize import normalize

if TYPE_CHECKING:
    from .invariant import Invariant

__all__ = [
    "SENTINEL_VERSION",
    "SentinelBlock",
    "assert_block_present",
    "block_checksum",
    "expected_checksum",
    "find_blocks",
    "render_block",
    "strip_blocks",
]

SENTINEL_VERSION = 1

_HEADER_RE = re.compile(
    r"^[ \t]*<<COMPACTION-GUARD:(\d+) sha256=([0-9a-f]{64})>>[ \t\r]*$",
    re.MULTILINE,
)
_FOOTER_RE = re.compile(
    r"^[ \t]*<<END-COMPACTION-GUARD sha256=([0-9a-f]{64})>>[ \t\r]*$",
    re.MULTILINE,
)
# Leading indentation is tolerated (transport may indent quoted blocks); the
# id is a single non-space token; everything after the separator is text.
_ENTRY_RE = re.compile(r"^[ \t]*\[block\] (\S+) :: (.*)$")


# The escape mark must survive normalize() or the checksum cannot see escape
# structure; see the module docstring. U+00A6 is category So, which the
# normalisation pipeline keeps, and it has no NFKC decomposition or case
# mapping, so it is byte-stable through the comparison space.
_ESCAPE_MARK = "¦"


def _escape(text: str) -> str:
    # The mark itself is escaped first so a literal U+00A6 in constraint
    # text can never be misread as the start of an escape sequence.
    return (
        text.replace(_ESCAPE_MARK, _ESCAPE_MARK + _ESCAPE_MARK)
        .replace("\\", _ESCAPE_MARK + "b")
        .replace("\n", _ESCAPE_MARK + "n")
        .replace("\r", _ESCAPE_MARK + "r")
    )


def _unescape(text: str) -> str:
    # A character scanner, not chained str.replace calls: chained replaces
    # can turn an escaped literal back into an active escape and break the
    # round-trip. The scanner consumes each escape exactly once.
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == _ESCAPE_MARK and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == _ESCAPE_MARK:
                out.append(_ESCAPE_MARK)
                i += 2
                continue
            if nxt == "b":
                out.append("\\")
                i += 2
                continue
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "r":
                out.append("\r")
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def block_checksum(interior: str) -> str:
    """sha256 hex digest over the normalised interior text."""
    return sha256(normalize(interior).encode("utf-8")).hexdigest()


def _interior(invariants: Sequence[Invariant]) -> str:
    return "\n".join(
        f"[block] {inv.id} :: {_escape(inv.text)}" for inv in invariants
    )


def render_block(invariants: Sequence[Invariant]) -> str:
    """Render the sentinel block for these invariants, in registry order.

    An empty registry renders header and footer around an empty interior;
    whether to inject anything at all in that case is the guard's decision,
    not this module's. The output has no trailing newline: the injection
    site owns its surrounding whitespace.
    """
    interior = _interior(invariants)
    digest = block_checksum(interior)
    header = f"<<COMPACTION-GUARD:{SENTINEL_VERSION} sha256={digest}>>"
    footer = f"<<END-COMPACTION-GUARD sha256={digest}>>"
    if interior:
        return f"{header}\n{interior}\n{footer}"
    return f"{header}\n{footer}"


def expected_checksum(invariants: Sequence[Invariant]) -> str:
    """The checksum ``render_block`` would stamp for these invariants.

    Exists so callers can verify presence without re-rendering and searching
    for the full block text, which would defeat the re-flow tolerance the
    normalised checksum provides.
    """
    return block_checksum(_interior(invariants))


@dataclass(frozen=True, slots=True)
class SentinelBlock:
    """One sentinel block located in a rendered context.

    ``entries`` is a best-effort recovery of (id, text) pairs and is exact
    for untampered renderer output. It is not the integrity signal: after
    legitimate transport re-flow the entry lines may no longer parse while
    the normalised checksum still verifies. Trust ``verify()``, read
    ``entries`` for diagnostics and round-trip checks.
    """

    start: int
    end: int
    version: int
    header_checksum: str
    footer_checksum: str
    interior: str
    entries: tuple[tuple[str, str], ...]

    @property
    def computed_checksum(self) -> str:
        return block_checksum(self.interior)

    def verify(self) -> None:
        """Raise ``BlockIntegrityError`` unless header, footer and content agree."""
        if self.header_checksum != self.footer_checksum:
            raise BlockIntegrityError(
                "sentinel block header and footer disagree: "
                f"header sha256={self.header_checksum}, footer sha256={self.footer_checksum}"
            )
        computed = self.computed_checksum
        if computed != self.header_checksum:
            raise BlockIntegrityError(
                "sentinel block interior was edited: "
                f"declared sha256={self.header_checksum}, computed sha256={computed}"
            )


def _parse_entries(interior: str) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for line in interior.splitlines():
        match = _ENTRY_RE.match(line.rstrip("\r"))
        if match:
            entries.append((match.group(1), _unescape(match.group(2))))
    return tuple(entries)


def find_blocks(text: str) -> tuple[SentinelBlock, ...]:
    """Locate every sentinel block in ``text``, in document order.

    A header without a following footer is not a block; scanning stops there
    rather than guessing at a boundary, because stripping or verifying
    against a guessed span could destroy user content. Multiple blocks are
    real: a stale block from before compaction can coexist with a fresh one
    until repair strips and replaces it.
    """
    blocks: list[SentinelBlock] = []
    pos = 0
    while True:
        header = _HEADER_RE.search(text, pos)
        if header is None:
            break
        footer = _FOOTER_RE.search(text, header.end())
        if footer is None:
            break
        interior = text[header.end() : footer.start()].strip("\r\n")
        blocks.append(
            SentinelBlock(
                start=header.start(),
                end=footer.end(),
                version=int(header.group(1)),
                header_checksum=header.group(2),
                footer_checksum=footer.group(1),
                interior=interior,
                entries=_parse_entries(interior),
            )
        )
        pos = footer.end()
    return tuple(blocks)


def strip_blocks(text: str) -> str:
    """Remove every sentinel block from ``text``, tidying the seams.

    This is the first half of "inject replaces any existing sentinel": repair
    strips whatever blocks exist (fresh, stale, or corrupted beyond
    verification) and injects exactly one rendered from the current registry.
    That makes repair idempotent and convergent, and it is why a compactor
    instructed by an injected prompt to drop the block cannot win: it never
    has the last write.
    """
    result = text
    for block in reversed(find_blocks(text)):
        before = result[: block.start]
        after = result[block.end :]
        if before.endswith("\n") and after.startswith("\n"):
            after = after[1:]
        elif not before and after.startswith("\n"):
            after = after.lstrip("\n")
        result = before + after
    return result


def assert_block_present(text: str, checksum: str) -> None:
    """Verify that a block with this checksum is present and intact in ``text``.

    The microsecond integrity check behind ``Guard.assert_present``. Success
    requires one block whose declared header and footer digests and computed
    interior digest all equal ``checksum``; anything less raises
    ``BlockIntegrityError`` with a message that distinguishes absence from
    tampering, because the two point at different harness bugs (downstream
    trimming versus content rewriting).
    """
    blocks = find_blocks(text)
    if not blocks:
        raise BlockIntegrityError(
            "no sentinel block found in context; expected a block with "
            f"sha256={checksum}. Something between repair and this check "
            "removed guard-owned text."
        )
    for block in blocks:
        if (
            block.header_checksum == checksum
            and block.footer_checksum == checksum
            and block.computed_checksum == checksum
        ):
            return
    seen = ", ".join(
        f"declared={b.header_checksum[:12]} computed={b.computed_checksum[:12]}"
        for b in blocks
    )
    raise BlockIntegrityError(
        f"no sentinel block matches sha256={checksum}; "
        f"found {len(blocks)} block(s): {seen}. The block present was edited "
        "or is stale relative to the registry."
    )
