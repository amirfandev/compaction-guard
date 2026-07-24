"""Deterministic anchor extraction: values, identifiers, and modality vocabulary.

Anchors are the tokens an invariant cannot afford to lose. They are extracted
once, at registration, by fixed regular expressions and committed vocabulary
tables, never by a model. This is what makes MUTATED and WEAKENED detectable
in the zero-dependency install: the lexical detector extracts anchors from the
summary with the same rules and compares normalised anchor sets, so a swapped
digit or a vanished "must not" is a set difference, not a similarity score.

Design bias, stated once and applied throughout: when a rule is ambiguous,
prefer the error that produces a false alarm over the error that certifies a
broken constraint. A spurious anchor can cause a MUTATED verdict on a harmless
paraphrase, which costs the user an unnecessary look at a report. A missed
anchor can let a real mutation be certified as PRESERVED, which is the one
failure this library exists to prevent. The corpus release gate (false-certify
rate of zero) enforces the same bias downstream.

Per-kind normalisation is deliberately asymmetric:

- Currency amounts are canonicalised through ``Decimal`` ("$500.00", "$500"
  and "$0.5k" all become "500 usd") because money is written many ways and
  the quantity is what is bound.
- Bare numbers are kept verbatim apart from thousands separators, because
  "3.10" is as likely a version as a decimal, and canonicalising it to "3.1"
  would make a real mutation invisible.
- Identifiers are casefolded and otherwise untouched: they are names, and
  names match exactly or not at all.
- Modality phrases live in the ``normalize`` space, where "Read-only" has
  already become "read only" and "don't" has become "don t". The vocabulary
  is written in that space, including the odd-looking apostrophe casualties.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from .normalize import normalize

__all__ = [
    "MODALITY_VOCABULARY",
    "STOPWORDS",
    "Anchor",
    "AnchorKind",
    "extract_anchors",
    "extract_topic",
]


class AnchorKind(StrEnum):
    """The three families of tokens an invariant is anchored to."""

    VALUE = "value"
    """Bound quantities: $500, 3.11, 30d, 20%."""

    IDENTIFIER = "identifier"
    """Code-shaped names: orders_prod, us-east-1, a.py, /etc/passwd."""

    MODALITY = "modality"
    """Obligation force: must not, never, only, read-only, cap, at most."""


@dataclass(frozen=True, slots=True)
class Anchor:
    """One extracted anchor: the raw span and its canonical comparison form.

    ``raw`` is what appeared in the text ("$500"); ``normalized`` is what two
    sides are compared on ("500 usd"). Detectors must compare ``normalized``
    values only; ``raw`` exists for evidence strings a human will read.
    """

    kind: AnchorKind
    raw: str
    normalized: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "raw": self.raw, "normalized": self.normalized}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Anchor:
        return cls(
            kind=AnchorKind(data["kind"]),
            raw=str(data["raw"]),
            normalized=str(data["normalized"]),
        )


# --------------------------------------------------------------------------
# Value extraction tables
# --------------------------------------------------------------------------

# Symbol currencies. ¥ is mapped to jpy; it is also the yuan sign, and picking
# one code deterministically beats guessing from context. Both sides of a
# comparison map it the same way, which is all anchor matching needs.
_CURRENCY_SYMBOLS: dict[str, str] = {"$": "usd", "€": "eur", "£": "gbp", "¥": "jpy"}

# Word currencies. "pounds" is absent on purpose: it collides with the unit of
# mass often enough that a spurious gbp anchor would be common in file-size
# and weight contexts. "gbp" the code is unambiguous and included.
_CURRENCY_WORDS: dict[str, str] = {
    "usd": "usd",
    "eur": "eur",
    "gbp": "gbp",
    "jpy": "jpy",
    "dollar": "usd",
    "dollars": "usd",
    "euro": "eur",
    "euros": "eur",
}

# Order-of-magnitude suffixes, expanded only when currency-marked. A bare
# "30m" is minutes or metres far more often than millions; "$30m" is money.
_MAGNITUDE_EXP: dict[str, int] = {"k": 3, "m": 6, "b": 9, "bn": 9}

# Unit spellings mapped to one canonical form per unit. The canonical form is
# short and lowercase; both sides of a comparison pass through this table, so
# "30 days" and "30d" meet at "30 d". A bare "m" stays "m": it is ambiguous
# between minutes and metres, and resolving it wrongly is worse than carrying
# the ambiguity symmetrically.
_UNIT_CANON: dict[str, str] = {
    "%": "pct",
    "percent": "pct",
    "pct": "pct",
    "ms": "ms",
    "msec": "ms",
    "msecs": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "m": "m",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
    "w": "w",
    "wk": "w",
    "wks": "w",
    "week": "w",
    "weeks": "w",
    "mo": "mo",
    "month": "mo",
    "months": "mo",
    "y": "y",
    "yr": "y",
    "yrs": "y",
    "year": "y",
    "years": "y",
    "kb": "kb",
    "kib": "kib",
    "mb": "mb",
    "mib": "mib",
    "gb": "gb",
    "gib": "gib",
    "tb": "tb",
    "tib": "tib",
    "token": "tokens",
    "tokens": "tokens",
    "row": "rows",
    "rows": "rows",
    "request": "requests",
    "requests": "requests",
    "retry": "retries",
    "retries": "retries",
    "attempt": "attempts",
    "attempts": "attempts",
}

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

_RE_CURRENCY_SYMBOL = re.compile(rf"([$€£¥])\s?({_NUM})\s?([kK]|[mM]|[bB]n?)?\b")
_RE_CURRENCY_WORD = re.compile(
    rf"\b({_NUM})\s?({'|'.join(sorted(_CURRENCY_WORDS, key=len, reverse=True))})\b",
    re.IGNORECASE,
)
_WORD_UNITS = "|".join(
    re.escape(u) for u in sorted((u for u in _UNIT_CANON if u != "%"), key=len, reverse=True)
)
_RE_UNIT = re.compile(rf"\b({_NUM})\s?(%|(?:{_WORD_UNITS})\b)", re.IGNORECASE)
# Multi-dot tails allow versions: 3.11, 1.2.3. The optional v prefix catches
# "v3.11" (no word boundary exists between "v" and "3", so without it the
# whole token would be missed, not partially matched).
_RE_BARE_NUMBER = re.compile(r"\b[vV]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*\b")

# --------------------------------------------------------------------------
# Identifier extraction tables
# --------------------------------------------------------------------------

_RE_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]*@[\w-]+(?:\.[\w-]+)+\b")
_RE_PATH = re.compile(r"(?<![\w./-])/?[\w.+-]+(?:/[\w.+-]+)+")
_RE_DOTTED = re.compile(r"\b[A-Za-z0-9_-]+(?:\.[A-Za-z][A-Za-z0-9_-]*)+\b")
_RE_SNAKE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_RE_KEBAB = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")

# Dotted tokens that are prose abbreviations, not identifiers. Compared after
# casefolding and trailing-dot removal.
_DOTTED_STOP = frozenset({"e.g", "i.e", "a.m", "p.m", "u.s", "u.k"})

# Slash-joined word pairs that are prose, not paths.
_PATH_STOP = frozenset({"and/or", "either/or", "he/she", "him/her", "his/her", "y/n"})

# --------------------------------------------------------------------------
# Modality vocabulary
# --------------------------------------------------------------------------

MODALITY_VOCABULARY: tuple[str, ...] = (
    # Multiword phrases. Matching is longest-first, so "must not" is consumed
    # as one anchor and never leaves a bare "must" behind: a summary that
    # kept "must" but lost "not" therefore shows a modality set difference,
    # which is exactly the WEAKENED signal.
    "no more than",
    "at most",
    "at least",
    "must not",
    "may not",
    "shall not",
    "can not",
    "do not",
    # normalize() turns punctuation into spaces, so "don't" and "can't"
    # arrive here as "don t" and "can t". The vocabulary is written in
    # normalised space, so these entries look odd and are correct.
    "don t",
    "can t",
    "read only",
    "write only",
    "append only",
    # Single words.
    "cannot",
    "must",
    "never",
    "always",
    "only",
    "cap",
    "capped",
    "max",
    "maximum",
    "minimum",
    "forbidden",
    "prohibited",
    "banned",
    "required",
)
"""The committed obligation vocabulary, matched in ``normalize`` space.

This is a list, not an ontology. It covers the modality forms in the spec and
the soft-constraint phrasing seen in the source material (spend caps, scope
limits, permission restrictions). Extending it is a one-line edit plus a
fixture; inferring modality with a model is out of scope by design, because a
vocabulary miss degrades to a weaker verdict while a model dependency would
cost the zero-dependency guarantee. "min" is deliberately absent: it collides
with the minutes unit ("30 min") and "minimum" covers the deontic use.
"""

_RE_MODALITY = re.compile(
    r"\b(?:"
    + "|".join(re.escape(p) for p in sorted(MODALITY_VOCABULARY, key=len, reverse=True))
    + r")\b"
)

STOPWORDS: frozenset[str] = frozenset(
    """
    a an the is are was were be been being am do does did done has have had
    this that these those there here it its they them their we our you your
    i me my he she his her of for to in on at by with from as into onto over
    under and or but if then than so such any all each per via will would
    should could can may might shall s t
    """.split()
)
"""Function words excluded from topic sets.

"not" is intentionally kept out of this list: standalone negation is content.
The single letters "s" and "t" are here because possessives and contractions
shed them under punctuation stripping ("run s", "don t"). Weak modals
(should, could, may) are stopped because bare permission verbs carry little
topical signal; the deontic forms that matter are modality anchors.
"""


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _decimal_str(value: Decimal) -> str:
    """Render a Decimal without exponent notation or trailing zeros."""
    text = format(value.normalize(), "f")
    return text


def _money_amount(number: str, magnitude: str | None) -> str:
    amount = Decimal(number.replace(",", ""))
    if magnitude:
        amount *= Decimal(10) ** _MAGNITUDE_EXP[magnitude.lower()]
    return _decimal_str(amount)


def _overlaps(claimed: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < c_end and end > c_start for c_start, c_end in claimed)


def _claim(claimed: list[tuple[int, int]], start: int, end: int) -> None:
    claimed.append((start, end))


def extract_anchors(text: str) -> tuple[Anchor, ...]:
    """Extract all anchors from ``text``, deterministically.

    Runs identifier patterns before value patterns over the raw text with
    span claiming, so "sha-256" is one identifier rather than an identifier
    plus a spurious bare number, and "$500" is one currency value rather
    than a bare "500". Modality matching runs separately over the normalised
    text, where hyphenation and case have already been flattened.

    Output order: VALUE anchors in first-occurrence order, then IDENTIFIER,
    then MODALITY. Duplicates collapse on (kind, normalized), keeping the
    first raw spelling seen. The same function runs on invariant text at
    registration and on summary text inside the lexical detector; symmetry
    of the rules is what makes anchor-set comparison meaningful.
    """
    claimed: list[tuple[int, int]] = []
    identifiers: list[Anchor] = []
    values: list[Anchor] = []

    # Identifiers, most structured pattern first.
    for match in _RE_EMAIL.finditer(text):
        raw = match.group(0).rstrip(".,;:!?")
        if not raw or _overlaps(claimed, match.start(), match.start() + len(raw)):
            continue
        _claim(claimed, match.start(), match.start() + len(raw))
        identifiers.append(Anchor(AnchorKind.IDENTIFIER, raw, raw.casefold()))
    for match in _RE_PATH.finditer(text):
        raw = match.group(0).rstrip(".,;:!?")
        if not raw or raw.casefold() in _PATH_STOP:
            continue
        if _overlaps(claimed, match.start(), match.start() + len(raw)):
            continue
        _claim(claimed, match.start(), match.start() + len(raw))
        identifiers.append(Anchor(AnchorKind.IDENTIFIER, raw, raw.casefold()))
    for match in _RE_DOTTED.finditer(text):
        raw = match.group(0)
        if raw.casefold().rstrip(".") in _DOTTED_STOP:
            continue
        if _overlaps(claimed, match.start(), match.end()):
            continue
        _claim(claimed, match.start(), match.end())
        identifiers.append(Anchor(AnchorKind.IDENTIFIER, raw, raw.casefold()))
    for match in _RE_SNAKE.finditer(text):
        if _overlaps(claimed, match.start(), match.end()):
            continue
        raw = match.group(0)
        _claim(claimed, match.start(), match.end())
        identifiers.append(Anchor(AnchorKind.IDENTIFIER, raw, raw.casefold()))
    for match in _RE_KEBAB.finditer(text):
        raw = match.group(0)
        # A hyphenated token is an identifier only if it carries a digit
        # (us-east-1, gpt-4, sha-256). English compounds like "read-only" or
        # "long-running" are hyphenated too; without the digit rule every one
        # of them would become a spurious identifier anchor. The cost is that
        # digit-free names like "orders-prod" fall through to the topic set
        # instead, a recorded trade of recall for precision.
        if not any(ch.isdigit() for ch in raw):
            continue
        if _overlaps(claimed, match.start(), match.end()):
            continue
        _claim(claimed, match.start(), match.end())
        identifiers.append(Anchor(AnchorKind.IDENTIFIER, raw, raw.casefold()))

    # Values.
    for match in _RE_CURRENCY_SYMBOL.finditer(text):
        if _overlaps(claimed, match.start(), match.end()):
            continue
        symbol, number, magnitude = match.group(1), match.group(2), match.group(3)
        _claim(claimed, match.start(), match.end())
        normalized = f"{_money_amount(number, magnitude)} {_CURRENCY_SYMBOLS[symbol]}"
        values.append(Anchor(AnchorKind.VALUE, match.group(0), normalized))
    for match in _RE_CURRENCY_WORD.finditer(text):
        if _overlaps(claimed, match.start(), match.end()):
            continue
        number, word = match.group(1), match.group(2)
        _claim(claimed, match.start(), match.end())
        normalized = f"{_money_amount(number, None)} {_CURRENCY_WORDS[word.casefold()]}"
        values.append(Anchor(AnchorKind.VALUE, match.group(0), normalized))
    for match in _RE_UNIT.finditer(text):
        if _overlaps(claimed, match.start(), match.end()):
            continue
        number, unit = match.group(1), match.group(2)
        _claim(claimed, match.start(), match.end())
        normalized = f"{number.replace(',', '')} {_UNIT_CANON[unit.casefold()]}"
        values.append(Anchor(AnchorKind.VALUE, match.group(0), normalized))
    for match in _RE_BARE_NUMBER.finditer(text):
        if _overlaps(claimed, match.start(), match.end()):
            continue
        raw = match.group(0)
        _claim(claimed, match.start(), match.end())
        stripped = raw.replace(",", "")
        if stripped[:1] in ("v", "V") and len(stripped) > 1:
            stripped = stripped[1:]
        values.append(Anchor(AnchorKind.VALUE, raw, stripped.casefold()))

    # Modality, in normalised space. Positions there do not line up with the
    # raw text, so no span claiming against the groups above is needed or
    # possible; the two passes see disjoint vocabularies.
    modality: list[Anchor] = []
    for match in _RE_MODALITY.finditer(normalize(text)):
        phrase = match.group(0)
        modality.append(Anchor(AnchorKind.MODALITY, phrase, phrase))

    seen: set[tuple[AnchorKind, str]] = set()
    ordered: list[Anchor] = []
    for anchor in (*values, *identifiers, *modality):
        key = (anchor.kind, anchor.normalized)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(anchor)
    return tuple(ordered)


def extract_topic(text: str, anchors: Sequence[Anchor] = ()) -> frozenset[str]:
    """Content words of ``text`` minus stopwords and anchor tokens.

    The topic set is what "the subject survives" means for the lexical
    detector: if these tokens appear in the summary but an anchor is missing,
    the constraint was mutated or weakened rather than dropped. Anchor tokens
    are subtracted in both their raw-normalised and canonical spellings, so
    "orders_prod" removes both "orders" and "prod" (normalisation splits on
    the underscore) as well as the joined form. Pure digits and single
    characters are excluded: digits belong to value anchors, and one-letter
    leftovers are punctuation shrapnel, not topics.
    """
    anchor_tokens: set[str] = set()
    for anchor in anchors:
        anchor_tokens.update(normalize(anchor.raw).split())
        anchor_tokens.update(anchor.normalized.split())
    return frozenset(
        token
        for token in normalize(text).split()
        if len(token) > 1
        and not token.isdigit()
        and token not in STOPWORDS
        and token not in anchor_tokens
    )
