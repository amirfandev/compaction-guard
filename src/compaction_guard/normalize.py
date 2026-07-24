"""Text canonicalisation: the one function every exact claim in this library rests on.

``normalize`` is the shared coordinate system. The lexical detector's containment
checks, the sentinel block checksum, and invariant id derivation all compare
strings only after passing them through here. That is deliberate: if two call
sites normalised differently, "verbatim survival" would mean two different
things, and a checksum computed on one side could never verify on the other.

The pipeline, in order:

1. NFKC, then casefold, then NFKC again. The second pass exists because
   casefolding can produce sequences that are no longer in normal form
   (U+0130 casefolds to ``i`` plus a combining dot, for example). One pass
   each approximates Unicode's NFKC_Casefold without shipping its table.
2. Drop format characters (category Cf). Zero-width spaces, joiners, BOMs and
   soft hyphens are the classic way text sneaks past exact matching while
   looking identical on screen. They carry no content, so they are removed
   rather than replaced.
3. Replace control characters (Cc) and all punctuation (P*) with a space.
   Punctuation becomes a space, not the empty string, so that ``read-only``
   and ``read only`` collapse to the same tokens instead of ``readonly``
   versus ``read only``. The alternative (deleting punctuation outright)
   was rejected because it glues hyphenated words together and silently
   changes token boundaries.
4. Collapse every whitespace run to a single ASCII space and strip the ends.

What this function deliberately does NOT do: touch digits. Stripping trailing
zeros or otherwise canonicalising numbers here would let ``3.10`` collapse
into ``3.1``, which is exactly the kind of value mutation the anchor layer
exists to catch. Numeric equivalence rules live in ``anchors.py``, per anchor
kind, where the trade-offs can be stated case by case.

Currency and math symbols (category S*) survive, so ``$500`` normalises to
``$500``. They are content, and the anchor extractor reads them.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize"]


def _one_pass(text: str) -> str:
    """A single application of the pipeline described in the module docstring."""
    folded = unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", text).casefold()
    )
    chars: list[str] = []
    for ch in folded:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        if cat == "Cc" or cat[0] == "P":
            chars.append(" ")
        else:
            chars.append(ch)
    return " ".join("".join(chars).split())


def normalize(text: str) -> str:
    """Return the canonical comparison form of ``text``. Deterministic, idempotent.

    Applied to fixpoint rather than once. Removing a format character can put
    a base letter and a combining mark side by side, and the next NFKC pass
    may then compose them into a character the first pass never saw. A single
    pass would leave ``normalize(normalize(x)) != normalize(x)`` for such
    inputs, and idempotence is a property the test suite (and the checksum
    logic) relies on. Convergence takes two passes for ordinary text; the
    bound of four is generous, and the final iteration is returned unchanged
    even if a pathological input has not settled, because returning something
    stable-ish beats looping forever.
    """
    prev = text
    for _ in range(4):
        cur = _one_pass(prev)
        if cur == prev:
            return cur
        prev = cur
    return prev
