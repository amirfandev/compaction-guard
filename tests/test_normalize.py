"""normalize(): the coordinate system every exact claim rests on.

Groups: idempotence under a seeded generator, the unicode sneak paths
(zero-width characters, NFKC-collapsing pairs, casefold), the punctuation
rule, and the honest boundary (script homoglyphs are NOT folded, and the
tests say so rather than pretending otherwise).
"""

from __future__ import annotations

import random

from compaction_guard.normalize import normalize

# Codepoint pools for the generator: plain text, whitespace zoo, format
# characters, combining marks, compatibility characters, symbols. The mix is
# chosen to hit the pipeline's edge interactions (Cf removal exposing new
# compositions, casefold producing non-NFKC output), not to be realistic prose.
_POOLS = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    " \t\n\r   　",
    "​‌‍⁠﻿­",
    "̧́̇̈",
    "ﬁﬂ①ⅡＡｋKÅ",
    "$€£¥%_-./:;!?()[]{}#@'\"",
    "ßİıΣσς",
    "😀👩‍💻中文カナ",
)


def _random_text(rng: random.Random) -> str:
    length = rng.randint(0, 40)
    chars = []
    for _ in range(length):
        pool = rng.choice(_POOLS)
        chars.append(rng.choice(pool))
    return "".join(chars)


def test_idempotence_property() -> None:
    """normalize(normalize(x)) == normalize(x) over 3000 seeded random inputs.

    Idempotence is what lets the checksum verify text that has already been
    through the pipeline once; a single counterexample breaks integrity
    checking, so the generator leans hard on the interaction-prone pools.
    """
    rng = random.Random(20260724)
    for _ in range(3000):
        text = _random_text(rng)
        once = normalize(text)
        assert normalize(once) == once, f"not idempotent for {text!r}"


def test_output_shape_property() -> None:
    """Output never carries leading/trailing space or doubled interior spaces."""
    rng = random.Random(99)
    for _ in range(1000):
        out = normalize(_random_text(rng))
        assert out == out.strip()
        assert "  " not in out


def test_empty_and_whitespace_only() -> None:
    assert normalize("") == ""
    assert normalize(" \t\n\r  ") == ""


def test_zero_width_characters_are_dropped() -> None:
    """The classic sneak: text that looks identical on screen must compare equal."""
    assert normalize("$5​00") == normalize("$500")
    assert normalize("or­ders_prod") == normalize("orders_prod")  # soft hyphen
    assert normalize("﻿hello") == "hello"  # BOM
    assert normalize("must‌ ‍not") == normalize("must not")  # ZWNJ, ZWJ
    assert normalize("a⁠b") == "ab"  # word joiner


def test_nfkc_collapsing_pairs() -> None:
    """Compatibility characters must land on the same canonical form."""
    assert normalize("ﬁle") == "file"  # fi ligature
    assert normalize("ＲＥＡＤ") == "read"  # fullwidth READ
    assert normalize("Kelvin") == "kelvin"  # Kelvin sign K
    assert normalize("①") == "1"  # circled one
    assert normalize("Ⅱ") == "ii"  # roman numeral two


def test_casefold_edges() -> None:
    assert normalize("ß") == "ss"
    assert normalize("ΣΙΓΜΑ") == normalize("σιγμα")
    # U+0130 casefolds to "i" plus a combining dot; the second NFKC pass in
    # the pipeline exists for exactly this shape, and idempotence must hold.
    dotted = normalize("İstanbul")
    assert normalize(dotted) == dotted


def test_script_homoglyphs_are_not_folded() -> None:
    """Cyrillic and Latin lookalikes stay distinct. This is the honest boundary:

    NFKC does not fold cross-script homoglyphs, so the library does not claim
    to catch them, and a test asserting they collapse would document a defense
    that does not exist.
    """
    assert normalize("о") != normalize("o")  # Cyrillic o vs Latin o
    assert normalize("р") != normalize("p")  # Cyrillic er vs Latin p


def test_punctuation_becomes_space() -> None:
    """read-only and read only must collapse to the same tokens, never readonly."""
    assert normalize("read-only") == "read only"
    assert normalize("read‐only") == "read only"  # unicode hyphen
    assert normalize("don't") == "don t"
    assert normalize("orders_prod") == "orders prod"
    assert normalize("a.b.c") == "a b c"
    assert "readonly" not in normalize("read-only")


def test_control_characters_become_space() -> None:
    assert normalize("a\x00b") == "a b"
    assert normalize("a\x1fb") == "a b"


def test_currency_and_math_symbols_survive() -> None:
    """Symbols are content; the anchor extractor reads them downstream."""
    assert normalize("$500") == "$500"
    assert "€" in normalize("€30")


def test_digits_never_canonicalised() -> None:
    """3.10 must not collapse into 3.1; that mutation is the anchor layer's to catch."""
    assert normalize("3.10") != normalize("3.1")


def test_case_and_whitespace_flattening() -> None:
    left = normalize("The  Database\n ORDERS_PROD   is\tProduction.")
    right = normalize("the database orders_prod is production")
    assert left == right
