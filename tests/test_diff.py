"""Region attribution: retained vs inserted vs evicted, and the site-labeled view.

Synthetic before/after pairs cover the spec's three named scenarios (no-op,
full replacement, partial summarisation with kept tail) plus the properties
that make attribution honest: digest matching over normalised text, multiset
semantics for duplicated messages, and sentinel interiors always labeled
REASSERTION_BLOCK.
"""

from __future__ import annotations

from compaction_guard.detectors.base import SurvivalSite
from compaction_guard.diff import attribute_regions, build_view, view_from_text
from compaction_guard.invariant import Invariant
from compaction_guard.render import render_block

MSG_A = "The database orders_prod is production. Read-only queries only."
MSG_B = "The budget cap for this run is $500."
MSG_C = "Now summarise our progress."
SUMMARY = "Recap: constraints were discussed and work proceeded."


def test_noop_compaction_everything_retained() -> None:
    before = [MSG_A, MSG_B, MSG_C]
    diff = attribute_regions(before, before)
    assert diff.retained == (MSG_A, MSG_B, MSG_C)
    assert diff.inserted == ()
    assert diff.evicted == ()


def test_full_replacement() -> None:
    before = [MSG_A, MSG_B]
    after = [SUMMARY]
    diff = attribute_regions(before, after)
    assert diff.retained == ()
    assert diff.inserted == (SUMMARY,)
    assert diff.evicted == (MSG_A, MSG_B)


def test_partial_summarisation_with_kept_tail() -> None:
    before = [MSG_A, MSG_B, MSG_C]
    after = [SUMMARY, MSG_C]
    diff = attribute_regions(before, after)
    assert diff.retained == (MSG_C,)
    assert diff.inserted == (SUMMARY,)
    assert diff.evicted == (MSG_A, MSG_B)


def test_reflowed_kept_message_still_counts_as_retained() -> None:
    """Digest matching runs over normalize output, so transport reformatting

    of a kept message does not spuriously promote it to summary text.
    """
    reflowed = "the DATABASE orders_prod\nis production.  READ-ONLY queries only."
    diff = attribute_regions([MSG_A], [reflowed])
    assert diff.retained == (reflowed,)
    assert diff.inserted == ()
    assert diff.evicted == ()


def test_duplicate_messages_use_multiset_semantics() -> None:
    """One kept copy must not vouch for many after-side duplicates."""
    diff = attribute_regions([MSG_A], [MSG_A, MSG_A])
    assert diff.retained == (MSG_A,)
    assert diff.inserted == (MSG_A,)

    diff = attribute_regions([MSG_A, MSG_A], [MSG_A])
    assert diff.retained == (MSG_A,)
    assert diff.evicted == (MSG_A,)


def test_empty_segments_appear_nowhere() -> None:
    diff = attribute_regions(["", "   ", MSG_A], [MSG_A, "\n\n"])
    assert diff.retained == (MSG_A,)
    assert diff.inserted == ()
    assert diff.evicted == ()


def test_build_view_labels_sites() -> None:
    before = [MSG_A, MSG_B, MSG_C]
    after = [SUMMARY, MSG_C]
    view = build_view(before, after)
    sites = {sentence.text: sentence.site for sentence in view.sentences}
    assert sites[MSG_C] is SurvivalSite.RETAINED_TAIL
    summary_sentences = [s for s in view.sentences if s.site is SurvivalSite.SUMMARY]
    assert summary_sentences, "inserted summary text must yield SUMMARY sentences"
    assert SUMMARY.split(":")[0] in summary_sentences[0].text


def test_view_from_text_is_all_summary() -> None:
    view = view_from_text("First point. Second point.")
    assert len(view.sentences) == 2
    assert all(s.site is SurvivalSite.SUMMARY for s in view.sentences)
    assert "first" in view.token_set
    assert view.normalized == view.normalized.strip()


def test_sentinel_interior_labeled_reassertion_block() -> None:
    """A constraint alive only inside a prior block is the guard's own echo,

    and the view must say so wherever the block appears.
    """
    inv = Invariant.parse(MSG_B)
    block = render_block([inv])
    text = f"{SUMMARY}\n{block}\ntrailing prose."
    view = view_from_text(text)
    by_site = {site: [] for site in SurvivalSite}
    for sentence in view.sentences:
        by_site[sentence.site].append(sentence.text)
    assert any(MSG_B in text for text in by_site[SurvivalSite.REASSERTION_BLOCK])
    assert any("trailing prose" in text for text in by_site[SurvivalSite.SUMMARY])
    # Marker lines never become sentences: they are wire format, not content.
    assert not any("COMPACTION-GUARD" in s.text for s in view.sentences)
    # Block text still participates in the token set, so containment checks
    # can find it (and report the honest REASSERTION_BLOCK site).
    assert "$500" in view.token_set


def test_block_inside_retained_segment_still_reassertion_block() -> None:
    inv = Invariant.parse(MSG_A)
    block = render_block([inv])
    carrier = f"kept prose.\n{block}"
    view = build_view([carrier], [carrier])
    block_sentences = [s for s in view.sentences if s.site is SurvivalSite.REASSERTION_BLOCK]
    # The interior is sentence-split like everything else; both pieces of the
    # two-sentence constraint must carry the block site.
    assert any("orders_prod is production" in s.text for s in block_sentences)
    assert any("Read-only queries only" in s.text for s in block_sentences)
    kept = [s for s in view.sentences if s.site is SurvivalSite.RETAINED_TAIL]
    assert any("kept prose" in s.text for s in kept)


def test_attribution_deterministic() -> None:
    before = [MSG_A, MSG_B, MSG_C, MSG_B]
    after = [SUMMARY, MSG_B, MSG_C]
    assert attribute_regions(before, after) == attribute_regions(before, after)
