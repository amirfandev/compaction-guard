"""Anchor extraction: the deterministic tables MUTATED and WEAKENED stand on.

Fixtures cover the spec's example shapes (currencies, units, versions,
identifiers, modality vocabulary), the canonicalisation asymmetries the
module commits to, and the recorded trades (digit-free kebab names are not
identifiers). Everything here must be deterministic across runs.
"""

from __future__ import annotations

from compaction_guard.anchors import (
    Anchor,
    AnchorKind,
    extract_anchors,
    extract_topic,
)


def _by_kind(text: str, kind: AnchorKind) -> dict[str, str]:
    return {a.raw: a.normalized for a in extract_anchors(text) if a.kind is kind}


def _normalized(text: str, kind: AnchorKind) -> set[str]:
    return {a.normalized for a in extract_anchors(text) if a.kind is kind}


# --- values: currency ---


def test_currency_symbol() -> None:
    assert _by_kind("The budget cap for this run is $500.", AnchorKind.VALUE) == {"$500": "500 usd"}


def test_currency_spellings_meet_at_one_form() -> None:
    """$500, $500.00 and 500 dollars are the same bound quantity."""
    assert _normalized("$500", AnchorKind.VALUE) == {"500 usd"}
    assert _normalized("$500.00", AnchorKind.VALUE) == {"500 usd"}
    assert _normalized("500 dollars", AnchorKind.VALUE) == {"500 usd"}
    assert _normalized("500 USD", AnchorKind.VALUE) == {"500 usd"}


def test_currency_magnitude_suffixes() -> None:
    """k/m/b expand only when currency-marked; $5k and $5000 must collide."""
    assert _normalized("$5k", AnchorKind.VALUE) == {"5000 usd"}
    assert _normalized("$0.5k", AnchorKind.VALUE) == {"500 usd"}
    assert _normalized("$30M", AnchorKind.VALUE) == {"30000000 usd"}
    assert _normalized("$5000", AnchorKind.VALUE) == _normalized("$5k", AnchorKind.VALUE)


def test_bare_magnitude_not_expanded() -> None:
    """A bare 30m is minutes or metres, not millions; expansion needs the mark."""
    assert "30000000" not in " ".join(_normalized("wait 30m", AnchorKind.VALUE))


def test_other_currency_symbols() -> None:
    assert _normalized("€30", AnchorKind.VALUE) == {"30 eur"}
    assert _normalized("£12", AnchorKind.VALUE) == {"12 gbp"}
    assert _normalized("¥1000", AnchorKind.VALUE) == {"1000 jpy"}


# --- values: units, versions, bare numbers ---


def test_duration_units_meet_at_one_form() -> None:
    assert _normalized("retain logs for 30d", AnchorKind.VALUE) == {"30 d"}
    assert _normalized("retain logs for 30 days", AnchorKind.VALUE) == {"30 d"}


def test_percent() -> None:
    assert _normalized("keep error rate under 20%", AnchorKind.VALUE) == {"20 pct"}
    assert _normalized("keep error rate under 20 percent", AnchorKind.VALUE) == {"20 pct"}


def test_versions_kept_verbatim() -> None:
    """3.10 stays 3.10: trailing-zero stripping would hide a real mutation."""
    assert _normalized("Python 3.10", AnchorKind.VALUE) == {"3.10"}
    assert _normalized("Python 3.1", AnchorKind.VALUE) == {"3.1"}
    assert _normalized("Python 3.10", AnchorKind.VALUE) != _normalized("Python 3.1", AnchorKind.VALUE)


def test_v_prefix_and_dotted_versions() -> None:
    assert _normalized("upgrade to v3.11", AnchorKind.VALUE) == {"3.11"}
    assert _normalized("upgrade to 3.11", AnchorKind.VALUE) == {"3.11"}
    assert "1.2.3" in _normalized("pin at 1.2.3", AnchorKind.VALUE)


def test_thousands_separators_removed() -> None:
    assert _normalized("limit is 1,000 rows", AnchorKind.VALUE) == {"1000 rows"}


# --- identifiers ---


def test_identifier_shapes() -> None:
    text = "orders_prod via us-east-1, see a.py and /etc/passwd, mail irfan@chatari.com, use sha-256"
    got = _normalized(text, AnchorKind.IDENTIFIER)
    assert {"orders_prod", "us-east-1", "a.py", "/etc/passwd", "irfan@chatari.com", "sha-256"} <= got


def test_identifiers_casefolded_only() -> None:
    assert _normalized("Orders_Prod", AnchorKind.IDENTIFIER) == {"orders_prod"}


def test_hyphenated_word_without_digit_is_not_identifier() -> None:
    """read-only is English, not a name; the digit rule keeps it out."""
    assert _normalized("read-only long-running", AnchorKind.IDENTIFIER) == set()


def test_digit_free_kebab_name_falls_to_topic() -> None:
    """The recorded trade: orders-prod (no digit) is not an identifier anchor,

    but its words still land in the topic set, so its loss is not invisible.
    """
    assert _normalized("orders-prod", AnchorKind.IDENTIFIER) == set()
    assert {"orders", "prod"} <= extract_topic("protect orders-prod")


def test_prose_dotted_abbreviations_excluded() -> None:
    got = _normalized("e.g. at 9 a.m. in the U.S. office", AnchorKind.IDENTIFIER)
    assert got == set()


def test_prose_slash_pairs_excluded() -> None:
    assert _normalized("and/or either/or", AnchorKind.IDENTIFIER) == set()


def test_currency_not_double_counted_as_bare_number() -> None:
    """$500 is one currency value; span claiming must not add a bare 500."""
    values = _normalized("$500", AnchorKind.VALUE)
    assert values == {"500 usd"}


def test_sha_256_not_split_into_number() -> None:
    anchors = extract_anchors("hash with sha-256")
    kinds = {(a.kind, a.normalized) for a in anchors}
    assert (AnchorKind.IDENTIFIER, "sha-256") in kinds
    assert (AnchorKind.VALUE, "256") not in kinds


# --- modality ---


def test_modality_vocabulary_examples() -> None:
    got = _normalized(
        "You must not delete data. Never bypass review. Read-only mode only. Cap at most usage.",
        AnchorKind.MODALITY,
    )
    assert {"must not", "never", "read only", "only", "cap", "at most"} <= got


def test_must_not_never_leaves_bare_must() -> None:
    """Longest-first matching: losing the "not" must show as a set difference."""
    got = _normalized("You must not write.", AnchorKind.MODALITY)
    assert got == {"must not"}
    assert "must" not in got


def test_apostrophe_casualties() -> None:
    """don't arrives in normalize space as "don t" and the vocabulary knows it."""
    assert _normalized("don't touch prod", AnchorKind.MODALITY) == {"don t"}
    assert _normalized("you can't do that", AnchorKind.MODALITY) == {"can t"}


def test_min_absent_minimum_present() -> None:
    """"min" collides with the minutes unit and is deliberately not modality."""
    assert _normalized("30 min timeout", AnchorKind.MODALITY) == set()
    assert _normalized("a minimum of care", AnchorKind.MODALITY) == {"minimum"}


# --- ordering, duplicates, determinism ---


def test_output_order_and_deduplication() -> None:
    text = "$500 then $500 again for orders_prod, which must not grow"
    anchors = extract_anchors(text)
    kinds = [a.kind for a in anchors]
    assert kinds == sorted(kinds, key=[AnchorKind.VALUE, AnchorKind.IDENTIFIER, AnchorKind.MODALITY].index)
    assert len([a for a in anchors if a.normalized == "500 usd"]) == 1


def test_extraction_deterministic() -> None:
    text = "Cap $5k for orders_prod in us-east-1; must not exceed 30d or 20%."
    assert extract_anchors(text) == extract_anchors(text)


def test_anchor_serde_round_trip() -> None:
    anchor = Anchor(AnchorKind.VALUE, "$500", "500 usd")
    assert Anchor.from_dict(anchor.to_dict()) == anchor


# --- topic ---


def test_topic_excludes_stopwords_and_anchor_tokens() -> None:
    text = "The database orders_prod is production. Read-only queries only."
    anchors = extract_anchors(text)
    topic = extract_topic(text, anchors)
    assert topic == frozenset({"database", "production", "queries"})


def test_topic_excludes_digits_and_single_letters() -> None:
    topic = extract_topic("a 500 x plan")
    assert "500" not in topic
    assert "x" not in topic
    assert "plan" in topic
