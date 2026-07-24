"""Region attribution across a compaction, and the SummaryView detectors examine.

The wrapper holds both sides of every compaction, and this module is where
that advantage is cashed in. Each side is a sequence of segments (one per
message, from ``AutoCodec.render_details``; one per context for codecs that
cannot segment). Segments are matched by digest of their ``normalize`` form:
an after-segment whose digest appears on the before side was kept verbatim,
everything else is inserted summary text, and before-segments with no
after-side match were evicted.

Why this matters: PRESERVED alone lies. A constraint that survived only in
the kept-verbatim tail was never processed by the summariser at all, and the
next compaction will feed exactly that tail through it. Reporting such a
constraint as simply preserved would teach users their compactor is safe one
step before it proves otherwise. Site attribution is what lets a finding
carry ``survived_in`` and ``at_risk`` instead.

Matching on normalised digests rather than raw bytes is a deliberate lean in
the library's stated direction: transport-level reformatting of a kept
message still counts it as retained. The failure modes are asymmetric. A
retained message misread as summary would merely lose an at-risk flag; a
summary misread as retained adds one. The second error is the false alarm,
and false alarms are the side this library chooses everywhere.

Text inside a sentinel block is attributed to REASSERTION_BLOCK regardless of
which segment carried it: survival inside a prior block is the guard's own
echo, not evidence the summariser preserved anything.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256

from .detectors.base import Sentence, SummaryView, SurvivalSite
from .normalize import normalize
from .render import find_blocks

__all__ = ["RegionDiff", "attribute_regions", "build_view", "view_from_text"]

# Sentence boundaries: terminal punctuation followed by whitespace, plus
# newlines (handled by splitting lines first). Deliberately dumb: an
# abbreviation like "e.g." over-splits, which costs nothing because
# containment checks also run against the full text, while a smarter
# splitter would be a heuristic in the one layer that must stay exact.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+")


def _digest(normalized_segment: str) -> str:
    return sha256(normalized_segment.encode("utf-8")).hexdigest()


def _digest_counts(segments: Iterable[str]) -> Counter[str]:
    """Multiset of segment digests, ignoring segments with no content.

    A Counter rather than a set because transcripts repeat messages
    (identical retries, repeated tool outputs), and set semantics would let
    one kept copy vouch for any number of after-side duplicates.
    """
    counts: Counter[str] = Counter()
    for segment in segments:
        normalized = normalize(segment)
        if normalized:
            counts[_digest(normalized)] += 1
    return counts


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionDiff:
    """What the compaction did to the message regions, by digest.

    ``retained`` and ``inserted`` partition the after side's non-empty
    segments; ``evicted`` is the before side's remainder. Segments whose
    normalised form is empty appear nowhere: they carry nothing a detector
    could compare. This record is diagnostic; verdicts come from the
    ``SummaryView`` built by ``build_view``, which does the same matching
    with per-sentence site labels.
    """

    retained: tuple[str, ...]
    inserted: tuple[str, ...]
    evicted: tuple[str, ...]


def attribute_regions(
    before_segments: Sequence[str], after_segments: Sequence[str]
) -> RegionDiff:
    """Partition both sides of a compaction into retained, inserted, evicted.

    Matching is greedy in document order over digest multisets, so duplicate
    messages resolve deterministically: the first after-side copy consumes
    the first before-side copy. A no-op compaction yields everything
    retained and nothing inserted or evicted, which is the honest answer,
    not a special case.
    """
    remaining_before = _digest_counts(before_segments)
    retained: list[str] = []
    inserted: list[str] = []
    for segment in after_segments:
        normalized = normalize(segment)
        if not normalized:
            continue
        digest = _digest(normalized)
        if remaining_before[digest] > 0:
            remaining_before[digest] -= 1
            retained.append(segment)
        else:
            inserted.append(segment)

    remaining_after = _digest_counts(after_segments)
    evicted: list[str] = []
    for segment in before_segments:
        normalized = normalize(segment)
        if not normalized:
            continue
        digest = _digest(normalized)
        if remaining_after[digest] > 0:
            remaining_after[digest] -= 1
        else:
            evicted.append(segment)
    return RegionDiff(
        retained=tuple(retained), inserted=tuple(inserted), evicted=tuple(evicted)
    )


def build_view(
    before_segments: Sequence[str], after_segments: Sequence[str]
) -> SummaryView:
    """The site-labeled view of the after side, for the detector chain.

    Computed once per compaction and shared across every invariant and
    detector layer, because sentence splitting and normalisation are the
    same work regardless of which constraint is being checked. Sentences in
    kept-verbatim segments carry RETAINED_TAIL, sentences in inserted
    segments carry SUMMARY, and sentinel-block interiors carry
    REASSERTION_BLOCK wherever they appear.
    """
    remaining_before = _digest_counts(before_segments)
    sited: list[tuple[str, SurvivalSite]] = []
    for segment in after_segments:
        normalized = normalize(segment)
        if not normalized:
            continue
        digest = _digest(normalized)
        if remaining_before[digest] > 0:
            remaining_before[digest] -= 1
            sited.append((segment, SurvivalSite.RETAINED_TAIL))
        else:
            sited.append((segment, SurvivalSite.SUMMARY))
    return _assemble_view(sited)


def view_from_text(text: str) -> SummaryView:
    """The view for bare summary text, where there is no before side.

    This is what ``check()`` uses in REASSERTED mode: everything is summary
    except sentinel-block interiors, which still label as REASSERTION_BLOCK
    so a constraint surviving only inside a previously injected block is
    reported as exactly that and never as summariser fidelity.
    """
    return _assemble_view([(text, SurvivalSite.SUMMARY)])


def _assemble_view(segments: Sequence[tuple[str, SurvivalSite]]) -> SummaryView:
    sentences: list[Sentence] = []
    for segment, site in segments:
        sentences.extend(_segment_sentences(segment, site))
    full_text = "\n\n".join(segment for segment, _ in segments if segment.strip())
    normalized = normalize(full_text)
    return SummaryView(
        full_text=full_text,
        normalized=normalized,
        sentences=tuple(sentences),
        token_set=frozenset(normalized.split()),
    )


def _segment_sentences(segment: str, site: SurvivalSite) -> list[Sentence]:
    """Split one segment into site-labeled sentences, carving out sentinel blocks.

    Block interiors are read through the parsed entries when they parse (the
    entry text is the actual constraint, free of wire markers); a block
    mangled past entry parsing falls back to its raw interior lines, which
    still carry the constraint words even if wrapped in format noise. Either
    way the site is REASSERTION_BLOCK, never the surrounding segment's.
    """
    sentences: list[Sentence] = []
    position = 0
    for block in find_blocks(segment):
        _extend_sentences(sentences, segment[position : block.start], site)
        if block.entries:
            for _invariant_id, text in block.entries:
                _extend_sentences(sentences, text, SurvivalSite.REASSERTION_BLOCK)
        else:
            _extend_sentences(
                sentences, block.interior, SurvivalSite.REASSERTION_BLOCK
            )
        position = block.end
    _extend_sentences(sentences, segment[position:], site)
    return sentences


def _extend_sentences(
    sentences: list[Sentence], text: str, site: SurvivalSite
) -> None:
    for line in text.splitlines():
        for part in _SENTENCE_BOUNDARY.split(line):
            piece = part.strip()
            if not piece:
                continue
            normalized = normalize(piece)
            if normalized:
                sentences.append(
                    Sentence(text=piece, normalized=normalized, site=site)
                )
