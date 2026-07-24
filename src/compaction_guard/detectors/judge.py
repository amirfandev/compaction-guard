"""The judge layer: an injected completion callable behind a checkable contract.

The judge is the only model surface in the package and it is deliberately
untrusted. Models acquiesce, drift between runs, and quote text that is not
there. So the contract converts a soft judgment into a checkable one: forced
choice among five verdicts, evidence before verdict in the reply shape, and a
cited span that must re-verify against the actual text by exact match. A
fabricated span, an unparseable reply, a verdict outside the forced choices,
or a raised callable all degrade to UNVERIFIABLE with the reason attached,
never to a trusted verdict and never to a crash on the compaction path.

PRESERVED and MUTATED are not choices, by matrix and by rubric. Verbatim
survival and value changes are decided by layers that recompute; a model
asserting either would be overriding deterministic evidence with a guess.
For the same reason a PARAPHRASED claim must carry the invariant's bound
values and identifiers inside the cited span itself: a paraphrase that lost
the number is not a paraphrase, whatever the judge thinks.

No SDK is imported here or anywhere in the package. The caller's closure
(``lambda prompt: client.complete(prompt)``) is the entire integration, which
also makes testing offline and trivial: inject a lambda returning a canned
JSON string.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..anchors import AnchorKind, extract_anchors
from ..normalize import normalize
from ..taxonomy import Kind
from .base import (
    JudgeFn,
    LayerVerdict,
    SummaryView,
    SurvivalSite,
    clip,
    contains_tokens,
    site_texts,
)

if TYPE_CHECKING:
    from ..invariant import Invariant

__all__ = ["JUDGE_PROMPT_TEMPLATE", "JudgeDetector", "build_prompt"]


JUDGE_PROMPT_TEMPLATE = """\
You are auditing what a context compaction did to one registered constraint.

THE CONSTRAINT:
{constraint}

THE POST-COMPACTION TEXT:
{context}

Task: decide what happened to the constraint in the post-compaction text.
Find your evidence first, then commit to a verdict. Reply with one JSON
object and nothing else, in exactly this shape:

{{"span": "<verbatim quote from the post-compaction text, or null>", \
"verdict": "<paraphrased|weakened|contradicted|dropped|unverifiable>", \
"reason": "<one short sentence>"}}

Forced choices:
- "paraphrased": the quoted span restates the constraint with the same
  values, names, and force, in different words.
- "weakened": the quoted span keeps the subject but loses obligation force
  or widens what is permitted.
- "contradicted": the quoted span asserts the opposite of the constraint or
  grants what it forbids.
- "dropped": no trace of the constraint remains; "span" must be null.
- "unverifiable": you cannot decide; say why in "reason".

Rules:
- The span must be copied character for character from the post-compaction
  text. It is re-checked mechanically; a span that does not match voids your
  verdict.
- Never answer "preserved" or "mutated". Verbatim survival and value changes
  are decided by deterministic layers, not by you.
- When torn between two verdicts, pick the one that reports more damage."""


def build_prompt(constraint: str, context: str) -> str:
    """Render the rubric for one invariant against one post-compaction text.

    Public because the judge calibration script in ``evidence/`` must send
    byte-identical prompts to a user-supplied callable; two renderings of
    "the same" rubric would make agreement numbers incomparable.
    """
    return JUDGE_PROMPT_TEMPLATE.format(constraint=constraint, context=context)


_FORCED_CHOICES: dict[str, Kind] = {
    Kind.PARAPHRASED.value: Kind.PARAPHRASED,
    Kind.WEAKENED.value: Kind.WEAKENED,
    Kind.CONTRADICTED.value: Kind.CONTRADICTED,
    Kind.DROPPED.value: Kind.DROPPED,
    Kind.UNVERIFIABLE.value: Kind.UNVERIFIABLE,
}


def _extract_json(raw: str) -> dict[str, Any] | None:
    """First JSON object in the reply, tolerant of fences and preambles.

    Models wrap JSON in code fences and prose no matter how firmly the
    prompt forbids it. Scanning for the first parseable object costs a few
    lines and removes the single most common cause of spurious
    UNVERIFIABLE results.
    """
    decoder = json.JSONDecoder()
    index = raw.find("{")
    while index != -1:
        try:
            obj, _end = decoder.raw_decode(raw, index)
        except ValueError:
            index = raw.find("{", index + 1)
            continue
        if isinstance(obj, dict):
            return obj
        index = raw.find("{", index + 1)
    return None


class JudgeDetector:
    """Forced-choice adjudication with span re-verification. Never a default.

    Construct with any ``JudgeFn``. The detector always returns a verdict
    (worst case UNVERIFIABLE), so a chain ending in a judge never reaches
    exhaustion silently: whatever went wrong is named in the evidence.
    """

    name: str = "judge"
    can_issue: frozenset[Kind] = frozenset(
        {
            Kind.PARAPHRASED,
            Kind.WEAKENED,
            Kind.CONTRADICTED,
            Kind.DROPPED,
            Kind.UNVERIFIABLE,
        }
    )

    def __init__(self, judge: JudgeFn) -> None:
        self._judge = judge

    def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
        prompt = build_prompt(invariant.text, view.full_text)
        try:
            raw = self._judge(prompt)
        except Exception as exc:
            # The callable is user code on the synchronous compaction path.
            # Its failure is this layer running out of soundness, which is
            # exactly what UNVERIFIABLE names; crashing compact() over a
            # judge would invert the layer's optional status.
            return self._unverifiable(
                f"judge callable raised {type(exc).__name__}: {exc}"
            )
        reply = _extract_json(raw)
        if reply is None:
            return self._unverifiable("judge reply contained no parseable JSON object")
        verdict_raw = reply.get("verdict")
        if not isinstance(verdict_raw, str) or verdict_raw.casefold() not in _FORCED_CHOICES:
            return self._unverifiable(
                f"judge verdict {verdict_raw!r} is not one of the forced choices"
            )
        kind = _FORCED_CHOICES[verdict_raw.casefold()]
        reason_raw = reply.get("reason")
        reason = clip(reason_raw.strip(), 200) if isinstance(reason_raw, str) else ""
        span = reply.get("span")

        if kind is Kind.UNVERIFIABLE:
            return self._unverifiable(
                f"judge abstained: {reason or 'no reason given'}"
            )
        if kind is Kind.DROPPED:
            if isinstance(span, str) and span.strip():
                return self._unverifiable(
                    "judge answered dropped but cited a span; a drop claim must cite nothing"
                )
            evidence = "judge found no surviving span"
            if reason:
                evidence = f"{evidence}; reason: {reason}"
            return LayerVerdict(kind=Kind.DROPPED, evidence=evidence, score=None, site=None)

        if not isinstance(span, str) or not span.strip():
            return self._unverifiable(
                f"judge verdict {kind.value!r} requires a cited span and none was given"
            )
        # Whole-token containment in normalize space, not raw substring: raw
        # matching tolerates nothing the transport does (line re-flow, case)
        # and accepts things it must not ("cap $500" inside "cap $5000").
        # Normalize-space boundary matching is the library's one notion of
        # "the same text", and the judge gets no looser a notion than the
        # lexical layer holds itself to.
        span_norm = normalize(span)
        if not contains_tokens(view.normalized, span_norm):
            return self._unverifiable(
                f'cited span failed re-verification: "{clip(span, 160)}"'
            )
        if kind is Kind.PARAPHRASED:
            required = frozenset(
                (anchor.kind, anchor.normalized)
                for anchor in invariant.anchors
                if anchor.kind is not AnchorKind.MODALITY
            )
            span_anchors = frozenset(
                (anchor.kind, anchor.normalized) for anchor in extract_anchors(span)
            )
            missing = required - span_anchors
            if missing:
                listed = ", ".join(
                    sorted(f"{anchor_kind.value}:{norm}" for anchor_kind, norm in missing)
                )
                return self._unverifiable(
                    f"paraphrase span lacks bound anchors: {listed}"
                )
        evidence = f'cited span verified: "{clip(span, 160)}"'
        if reason:
            evidence = f"{evidence}; reason: {reason}"
        return LayerVerdict(
            kind=kind,
            evidence=evidence,
            score=None,
            site=self._locate(span_norm, view),
        )

    @staticmethod
    def _unverifiable(evidence: str) -> LayerVerdict:
        return LayerVerdict(kind=Kind.UNVERIFIABLE, evidence=evidence, score=None, site=None)

    @staticmethod
    def _locate(span_norm: str, view: SummaryView) -> SurvivalSite | None:
        """Attribute a verified span to a site, preference order deciding ties."""
        for site, text in site_texts(view):
            if contains_tokens(normalize(text), span_norm):
                return site
        return None
