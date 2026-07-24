"""The stdlib detector: containment, anchor diff, modality diff.

This is the default install and the only detector most users will ever run,
so its honesty matters more than its reach. What it can prove, it proves
deterministically: verbatim survival by normalised containment, value and
identifier mutation by anchor-set difference, weakening by modality-anchor
loss. What it cannot tell apart, faithful paraphrase versus absence, it
refuses to guess at: it falls through as None and lets the chain end in
UNVERIFIABLE, because a cheap layer that guessed PRESERVED would be the exact
failure this library exists to catch. The known cost is stated in the spec:
at this tier a good paraphrase reports as DROPPED or UNVERIFIABLE, and REPAIR
makes that harmless (worst case, benign duplication of the constraint).

Rule order, mirroring the chain contract in the spec:

1. exact: the invariant's normalised text contained in one site's text
   (rule 1a), or a near-verbatim match against a short window of contiguous
   sentences (rule 1b): ordered token similarity at or above
   ``containment_threshold`` with every anchor intact inside that window.
   PRESERVED, with the site recorded. The window match is order-sensitive
   and local on purpose. An unordered site-wide token bag would certify
   "queries go to the primary, not replica_02" against a constraint that
   says the reverse, and would let a value anchor in a neighbouring
   sentence vouch for a mutated one; both are confirmed false-certify
   shapes, pinned as fixtures.
2. anchor_diff: the topic survives but a VALUE or IDENTIFIER anchor changed
   or vanished from every topic-bearing sentence. MUTATED. No later layer
   may override this; the chain's short-circuit makes that structural.
3. modality: topic and value anchors survive, a MODALITY anchor is gone
   from every topic-bearing sentence. WEAKENED. Rules 2 and 3 pool anchors
   from topic-bearing sentences only, not the whole view, because anchor
   words are ordinary text: an unrelated sentence containing "never" or a
   stray "$500" must not mask a loss in the sentence that actually
   restates the constraint.
4. miss: nothing above fired. ``examine`` returns None to escalate. When this
   detector is the chain's last layer, the chain calls ``conclude``, which
   issues DROPPED only for a complete miss (topic below threshold and no
   value or identifier anchor present); partial survival stays None, ending
   in UNVERIFIABLE, since partial wreckage is exactly where paraphrase and
   absence are indistinguishable without semantics.

Both sides of every comparison pass through the same machinery: the summary's
anchors come from ``anchors.extract_anchors`` exactly as the invariant's did
at registration. Symmetry of extraction is what makes a set difference mean
"the summariser changed this" rather than "the two sides were parsed
differently".
"""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING

from ..anchors import STOPWORDS, AnchorKind, extract_anchors
from ..normalize import normalize
from ..taxonomy import Kind
from .base import (
    SITE_PREFERENCE,
    LayerVerdict,
    SummaryView,
    SurvivalSite,
    clip,
    contains_tokens,
    site_texts,
    split_sentences,
)

if TYPE_CHECKING:
    from ..invariant import Invariant

__all__ = ["LexicalDetector"]

_AnchorKey = tuple[AnchorKind, str]

_WINDOW_DENSITY = 0.75
"""Minimum fraction of a rule-1b window's content tokens that must belong to
the ordered match. A near-verbatim copy is mostly the invariant's own words;
a window that is mostly other text can still thread a high-recall ordered
subsequence through neighbouring sentences, which is the anchor-pollution
false certify. 0.75 admits cosmetic insertions (a couple of extra words per
sentence) and rejects windows dominated by foreign text."""


@lru_cache(maxsize=8)
def view_anchor_keys(view: SummaryView) -> frozenset[_AnchorKey]:
    """(kind, normalized) anchor keys extracted from the whole view, cached.

    The chain runs every invariant against the same view, and anchor
    extraction is a stack of regex passes over the full text; recomputing it
    per invariant would multiply the cost of the hot path by registry size.
    Cached here, at module level, so the NLI and judge layers can share the
    identical extraction instead of growing their own slightly different one.
    """
    return frozenset(
        (anchor.kind, anchor.normalized) for anchor in extract_anchors(view.full_text)
    )


@lru_cache(maxsize=8)
def _site_data(
    view: SummaryView,
) -> tuple[tuple[SurvivalSite, str, frozenset[str], frozenset[_AnchorKey]], ...]:
    """Per site, in preference order: (site, normalized text, tokens, anchor keys)."""
    rows = []
    for site, text in site_texts(view):
        normalized = normalize(text)
        rows.append(
            (
                site,
                normalized,
                frozenset(normalized.split()),
                frozenset((a.kind, a.normalized) for a in extract_anchors(text)),
            )
        )
    return tuple(rows)


@lru_cache(maxsize=8)
def _sentence_data(
    view: SummaryView,
) -> tuple[tuple[SurvivalSite, tuple[str, ...], frozenset[str], frozenset[_AnchorKey]], ...]:
    """Per sentence, in view order: (site, token sequence, token set, anchor keys).

    The sentence-level table is what makes rules 1b, 2 and 3 local. Site- or
    view-level pooling was the confirmed false-certify and false-mask shape:
    tokens and anchors from one sentence vouching for a different sentence's
    claim. Anchor extraction runs on the raw sentence text, exactly as it ran
    on the invariant at registration.
    """
    rows = []
    for sentence in view.sentences:
        tokens = tuple(sentence.normalized.split())
        rows.append(
            (
                sentence.site,
                tokens,
                frozenset(tokens),
                frozenset((a.kind, a.normalized) for a in extract_anchors(sentence.text)),
            )
        )
    return tuple(rows)


def _anchor_keys(invariant: Invariant) -> frozenset[_AnchorKey]:
    return frozenset((a.kind, a.normalized) for a in invariant.anchors)


def _relevant_anchors(invariant: Invariant, view: SummaryView) -> frozenset[_AnchorKey]:
    """Anchors pooled from the sentences that carry the invariant's topic.

    Reassertion-block sentences never contribute: text inside a prior
    sentinel block is the guard's own earlier write, and letting it satisfy
    an anchor-survival test is how a carried block blinds mutation detection.
    An invariant with an empty topic set (every content word claimed by an
    anchor) pools from every non-block sentence, because with no topic
    tokens to test relevance against, the whole inspectable view is the only
    sound pool left.
    """
    pooled: set[_AnchorKey] = set()
    for site, _tokens, token_set, anchors in _sentence_data(view):
        if site is SurvivalSite.REASSERTION_BLOCK:
            continue
        if not invariant.topic or invariant.topic & token_set:
            pooled |= anchors
    return frozenset(pooled)


def _topic_ratio(invariant: Invariant, tokens: frozenset[str]) -> float:
    """Fraction of the invariant's topic tokens present in ``tokens``.

    An empty topic (every content word claimed by an anchor) counts as fully
    surviving: for such invariants the anchors carry the whole meaning, and
    the anchor rules should decide, not a vacuous ratio.
    """
    if not invariant.topic:
        return 1.0
    return len(invariant.topic & tokens) / len(invariant.topic)


class LexicalDetector:
    """Containment, anchor diff and modality diff over the normalised view.

    ``containment_threshold`` is the ordered token similarity at which a
    sentence window with all anchors intact counts as near-verbatim; the
    default of 0.9 demands essentially the same sentence with cosmetic
    edits, because PRESERVED is a certification and paraphrase must not slip
    in under it. ``topic_threshold`` is the fraction of topic tokens that
    must survive before rules 2 and 3 treat the constraint as still on the
    page; the 0.5 default asks for a majority. Both are design constants,
    adjustable per instance and exercised by the corpus rather than tuned by
    hand.
    """

    name: str = "lexical"
    can_issue: frozenset[Kind] = frozenset(
        {Kind.PRESERVED, Kind.MUTATED, Kind.WEAKENED, Kind.DROPPED}
    )

    def __init__(
        self,
        *,
        containment_threshold: float = 0.9,
        topic_threshold: float = 0.5,
    ) -> None:
        if not 0.0 < containment_threshold <= 1.0:
            raise ValueError("containment_threshold must be in (0, 1]")
        if not 0.0 < topic_threshold <= 1.0:
            raise ValueError("topic_threshold must be in (0, 1]")
        self._containment_threshold = containment_threshold
        self._topic_threshold = topic_threshold

    def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
        inv_norm = normalize(invariant.text)
        sites = _site_data(view)
        inv_anchors = _anchor_keys(invariant)

        # Rule 1a: verbatim survival at one site, in normalize space. Whole
        # tokens only: substring matching would certify "$500" inside
        # "$5000", the exact mutation this layer exists to catch. The
        # reassertion-block site is never a certifiable site here: the chain
        # hands this detector a block-free view, and if a caller feeds a
        # full view directly, the guard's own echo still must not read as
        # summariser fidelity.
        if inv_norm:
            for site, site_norm, _tokens, _anchors in sites:
                if site is SurvivalSite.REASSERTION_BLOCK:
                    continue
                if contains_tokens(site_norm, inv_norm):
                    return LayerVerdict(
                        kind=Kind.PRESERVED,
                        evidence=f'verbatim in {site.value}: "{clip(inv_norm, 120)}"',
                        score=1.0,
                        site=site,
                    )
            # A match spanning a region boundary has no single site. Rare,
            # but refusing to see it would demote verbatim survival to a
            # weaker verdict on a technicality of region bookkeeping. Block
            # sentences are excluded from the joined text for the same
            # reason they are skipped above.
            if any(s.site is SurvivalSite.REASSERTION_BLOCK for s in view.sentences):
                boundary_text = " ".join(
                    s.normalized
                    for s in view.sentences
                    if s.site is not SurvivalSite.REASSERTION_BLOCK
                )
            else:
                boundary_text = view.normalized
            if contains_tokens(boundary_text, inv_norm):
                return LayerVerdict(
                    kind=Kind.PRESERVED,
                    evidence=f'verbatim across region boundary: "{clip(inv_norm, 120)}"',
                    score=1.0,
                    site=None,
                )

        # Rule 1b: near-verbatim inside one short window of contiguous
        # sentences. Order-sensitive similarity, so a word-order inversion
        # that reverses meaning cannot certify; window-local, so a token or
        # anchor in a neighbouring sentence cannot vouch for this one.
        near = self._near_verbatim(invariant, inv_norm, inv_anchors, view)
        if near is not None:
            return near

        # Rules 2 and 3 pool tokens and anchors from topic-bearing sentences
        # only. A value or modality word surviving in an unrelated sentence
        # is not survival of this constraint's binding; the whole-view pool
        # was the confirmed masking bug.
        topic_ratio = _topic_ratio(invariant, view.token_set)
        pooled_anchors = _relevant_anchors(invariant, view)
        value_id = frozenset(
            key for key in inv_anchors if key[0] is not AnchorKind.MODALITY
        )
        missing_value_id = value_id - pooled_anchors

        if topic_ratio >= self._topic_threshold and missing_value_id:
            missing_kinds = {kind for kind, _norm in missing_value_id}
            replacements = sorted(
                norm
                for kind, norm in pooled_anchors - inv_anchors
                if kind in missing_kinds
            )[:6]
            listed = ", ".join(
                sorted(f"{kind.value}:{norm}" for kind, norm in missing_value_id)
            )
            if replacements:
                detail = "topic-sentence same-kind anchors: " + ", ".join(replacements)
            else:
                detail = "no same-kind replacement in any topic-bearing sentence"
            return LayerVerdict(
                kind=Kind.MUTATED,
                evidence=f"anchors missing: {listed}; {detail}",
                score=None,
                site=None,
            )

        modality = inv_anchors - value_id
        missing_modality = modality - pooled_anchors
        if topic_ratio >= self._topic_threshold and not missing_value_id and missing_modality:
            lost = ", ".join(sorted(norm for _kind, norm in missing_modality))
            return LayerVerdict(
                kind=Kind.WEAKENED,
                evidence=(
                    f"modality anchors lost: {lost}; "
                    f"topic survival {round(topic_ratio, 4)}"
                ),
                score=round(topic_ratio, 4),
                site=self._best_topic_site(invariant, sites),
            )

        # Rule 4: no sound verdict here. Escalate; paraphrase and absence
        # look identical from where this layer stands.
        return None

    def _near_verbatim(
        self,
        invariant: Invariant,
        inv_norm: str,
        inv_anchors: frozenset[_AnchorKey],
        view: SummaryView,
    ) -> LayerVerdict | None:
        """Rule 1b: ordered content-token recall over short, dense windows.

        ``difflib.SequenceMatcher`` over content tokens (normalised tokens
        minus stopwords), per window of contiguous same-site sentences, at
        most one sentence longer than the invariant itself. Three
        thresholds, each closing a confirmed false-certify shape:

        - Ordered recall (matched content tokens over invariant content
          length) must reach ``containment_threshold``. Ordered, because a
          permuted token bag scores 1.0 under set containment; the
          inversion "at the primary, not replica_02" against the reverse
          fails here since matching blocks must appear in order.
        - Content tokens, not all tokens, because stopwords pad recall:
          with articles counted, deleting the one scope word from
          "customer tables" still clears 0.9 on a ten-token sentence, and
          scope loss is WEAKENED, not PRESERVED.
        - Window density (matched content tokens over the window's content
          length) must reach ``_WINDOW_DENSITY``. Without it, an ordered
          subsequence threads across sentence boundaries and a "$500" in a
          neighbouring refund sentence vouches for the "$5000" mutation
          next to it; a window mostly made of other text is not a
          near-verbatim copy, whatever recall says.

        Anchors must additionally be intact inside the same window, and
        ``autojunk=False`` because SequenceMatcher's popularity heuristic
        silently drops frequent tokens on long windows, and a containment
        check with input-dependent blind spots is not a containment check.
        """
        all_inv_tokens = tuple(inv_norm.split())
        inv_tokens = tuple(
            token for token in all_inv_tokens if token not in STOPWORDS
        ) or all_inv_tokens
        if not inv_tokens:
            return None
        max_sentences = len(split_sentences(invariant.text)) + 1
        rows = _sentence_data(view)
        per_site: dict[SurvivalSite, list[int]] = {}
        for index, (site, _tokens, _token_set, _anchors) in enumerate(rows):
            per_site.setdefault(site, []).append(index)
        for site in SITE_PREFERENCE:
            if site is SurvivalSite.REASSERTION_BLOCK:
                continue
            indices = per_site.get(site)
            if not indices:
                continue
            for start in range(len(indices)):
                window_tokens: list[str] = []
                window_anchors: set[_AnchorKey] = set()
                for offset in range(max_sentences):
                    position = start + offset
                    if position >= len(indices):
                        break
                    _row_site, tokens, _token_set, anchors = rows[indices[position]]
                    window_tokens.extend(
                        token for token in tokens if token not in STOPWORDS
                    )
                    window_anchors |= anchors
                    if not window_tokens or not inv_anchors <= window_anchors:
                        continue
                    matcher = SequenceMatcher(
                        None, inv_tokens, tuple(window_tokens), autojunk=False
                    )
                    matched = sum(block.size for block in matcher.get_matching_blocks())
                    recall = matched / len(inv_tokens)
                    density = matched / len(window_tokens)
                    if recall >= self._containment_threshold and density >= _WINDOW_DENSITY:
                        return LayerVerdict(
                            kind=Kind.PRESERVED,
                            evidence=(
                                f"ordered content-token recall {round(recall, 4)} at "
                                f"window density {round(density, 4)} over {offset + 1} "
                                f"sentence(s) in {site.value}, all {len(inv_anchors)} "
                                "anchors intact in the window"
                            ),
                            score=round(recall, 4),
                            site=site,
                        )
        return None

    def conclude(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
        """Terminal-only rule 4: a complete miss is DROPPED when nothing can
        escalate further.

        The chain calls this only when this detector is the last layer and
        ``examine`` returned None. A complete miss means the topic fell below
        threshold and not one bound value or identifier survived; with no
        semantic layer left to consult, absence of every lexical trace is the
        soundest available reading. Anything short of that stays None: a
        partial wreck could be a paraphrase, and certifying its death would
        be a guess wearing a verdict.
        """
        topic_ratio = _topic_ratio(invariant, view.token_set)
        if topic_ratio >= self._topic_threshold:
            return None
        value_id = frozenset(
            key for key in _anchor_keys(invariant) if key[0] is not AnchorKind.MODALITY
        )
        if value_id & view_anchor_keys(view):
            return None
        return LayerVerdict(kind=Kind.DROPPED, evidence="lexical_only", score=None, site=None)

    def _best_topic_site(
        self,
        invariant: Invariant,
        sites: tuple[tuple[SurvivalSite, str, frozenset[str], frozenset[_AnchorKey]], ...],
    ) -> SurvivalSite | None:
        """The site holding the most topic tokens; preference order breaks ties.

        WEAKENED is a survival kind, so the report should say where the
        weakened residue lives; the site with the densest topic overlap is
        the best deterministic stand-in for "where the constraint ended up".
        """
        best: SurvivalSite | None = None
        best_count = 0
        for site, _norm, tokens, _anchors in sites:
            count = len(invariant.topic & tokens)
            if count > best_count:
                best = site
                best_count = count
        return best
