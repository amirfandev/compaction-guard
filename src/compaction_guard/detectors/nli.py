"""The NLI layer: bidirectional entailment over sentence windows.

The only offline layer that can say CONTRADICTED, and the layer that turns
"the words changed but did the meaning survive" into an answerable question.
It runs an ONNX export of ``cross-encoder/nli-deberta-v3-xsmall`` through
onnxruntime, pinned by commit, so a fixed install scores identically forever.

Direction semantics, stated carefully because they carry the verdicts. For a
window W (sentences of the view) and the invariant's declarative form H:

- W entails H: the constraint's full content is derivable from the summary.
- H entails W: the window claims no more than the constraint does.
- Both: the window restates the constraint. PARAPHRASED, provided every
  VALUE and IDENTIFIER anchor also survives lexically, because NLI is
  numerically insensitive ($500 entails $5000 both ways in practice) and a
  certifying verdict must never rest on a layer blind to the thing it would
  be certifying.
- H entails W but W does not entail H: the window is a strictly weaker
  consequence of the constraint ("be careful with orders_prod" from a
  read-only rule). WEAKENED. This is the one-direction case the spec's rule
  table names; the other lone direction (W entails H only, a window that
  says more than the constraint) falls through as None, since PARAPHRASED
  requires equivalence and nothing weaker may certify.
- Contradiction label with W as premise: CONTRADICTED, first window wins.
- Neutral everywhere: None, escalate.

MUTATED never appears here and cannot: the escalation matrix bars it, and the
chain's short-circuit means a lexical MUTATED was final before this layer ran.

Requires the ``nli`` extra (onnxruntime, tokenizers, huggingface-hub); the
heaviest extra, as the docs say. Without it, ``examine`` returns None and the
reason lands in ``unavailable`` for the chain's exhaustion evidence.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

from ..anchors import STOPWORDS, AnchorKind
from ..normalize import normalize
from ..taxonomy import Kind
from .base import LayerVerdict, SummaryView, SurvivalSite, clip, split_sentences
from .lexical import view_anchor_keys

if TYPE_CHECKING:
    from ..invariant import Invariant

__all__ = [
    "HYPOTHESIS_REWRITES",
    "NLI_ONNX_REPO",
    "NLI_REVISION",
    "NLI_SOURCE_MODEL",
    "NLIDetector",
    "template_hypothesis",
]

NLI_SOURCE_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
"""The model whose behaviour this layer runs. Kept for provenance."""

NLI_ONNX_REPO = "Xenova/nli-deberta-v3-xsmall"
"""ONNX export of NLI_SOURCE_MODEL (states its base model in its card).

Used instead of the source repo because it ships ``onnx/model.onnx`` and a
``tokenizer.json`` loadable by the ``tokenizers`` library, which keeps the
extra free of torch and of the transformers stack.
"""

NLI_REVISION = "2a4f614a701367a02d51389039afc998faeda637"
"""Pinned commit of NLI_ONNX_REPO on the Hugging Face hub."""


# ---------------------------------------------------------------------------
# Hypothesis templating: the rewrite-rule table. Never a model call.
# ---------------------------------------------------------------------------

HYPOTHESIS_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:please\s+)?do\s+not\s+(.+)$", re.IGNORECASE), "It is forbidden to {rest}"),
    (re.compile(r"^(?:please\s+)?don'?t\s+(.+)$", re.IGNORECASE), "It is forbidden to {rest}"),
    (
        re.compile(r"^(?:please\s+)?never\s+(.+)$", re.IGNORECASE),
        "It is forbidden to ever {rest}",
    ),
    (
        re.compile(r"^(?:please\s+)?always\s+(.+)$", re.IGNORECASE),
        "It is required to always {rest}",
    ),
    (
        re.compile(r"^(?:please\s+)?avoid\s+(.+)$", re.IGNORECASE),
        "It is required to avoid {rest}",
    ),
    (
        re.compile(r"^(?:please\s+)?(?:use\s+only|only\s+use)\s+(.+)$", re.IGNORECASE),
        "Only {rest} may be used",
    ),
    (re.compile(r"^no\s+(\w+ing\b.*)$", re.IGNORECASE), "{rest} is forbidden"),
    (
        re.compile(r"^(?:please\s+)?stay\s+(?:within|under|below)\s+(.+)$", re.IGNORECASE),
        "It is required to stay within {rest}",
    ),
    (re.compile(r"^(?:please\s+)?keep\s+(.+)$", re.IGNORECASE), "It is required to keep {rest}"),
)
"""Imperative openings rewritten to declarative statements, first match wins.

NLI models are trained on declarative premise-hypothesis pairs; a bare
imperative ("Do not deploy to eu-west-1") reads as an instruction, not a
proposition, and entailment against it is noisier. The table covers the
imperative openings that constraint prose actually uses; a sentence matching
no rule passes through unchanged, which is correct for the most common case
("The budget cap for this run is $500.") that is already declarative. This is
a rule table and not grammar: extending it is one line plus a fixture, and a
miss degrades to a slightly noisier NLI call, never to a wrong templating.
"""


def _template_sentence(sentence: str) -> str:
    body = sentence.strip().rstrip(".!?").strip()
    if not body:
        return ""
    for pattern, template in HYPOTHESIS_REWRITES:
        match = pattern.match(body)
        if match:
            return template.replace("{rest}", match.group(1).strip()) + "."
    return body + "."


def template_hypothesis(text: str) -> str:
    """The invariant's declarative form, per sentence, deterministically."""
    parts = [part for part in (_template_sentence(s) for s in split_sentences(text)) if part]
    if not parts:
        return text.strip()
    return " ".join(parts)


class NLIDetector:
    """Sentence-windowed bidirectional NLI. See the module docstring.

    Windows are single sentences plus adjacent same-site pairs (a constraint
    restated across a sentence boundary would otherwise never assemble a
    premise that carries it). Candidate windows are ranked by content-token
    overlap with the invariant and capped at ``max_windows``, because this
    layer sits on the synchronous compaction path and every window costs two
    model calls.

    ``min_entailment`` gates only the certifying verdict: PARAPHRASED needs
    argmax entailment in both directions at or above it. WEAKENED and
    CONTRADICTED are loss reports, safe to issue on argmax alone; demanding
    extra confidence before reporting damage would bias the layer toward
    silence exactly where silence costs the most.
    """

    name: str = "nli"
    can_issue: frozenset[Kind] = frozenset(
        {Kind.PARAPHRASED, Kind.WEAKENED, Kind.CONTRADICTED}
    )

    def __init__(
        self,
        *,
        repo_id: str = NLI_ONNX_REPO,
        revision: str = NLI_REVISION,
        max_windows: int = 8,
        min_entailment: float = 0.5,
        max_length: int = 512,
    ) -> None:
        if max_windows < 1:
            raise ValueError("max_windows must be at least 1")
        if not 0.0 <= min_entailment <= 1.0:
            raise ValueError("min_entailment must be in [0, 1]")
        self._repo_id = repo_id
        self._revision = revision
        self._max_windows = max_windows
        self._min_entailment = min_entailment
        self._max_length = max_length
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: tuple[str, ...] = ()
        self._label_index: dict[str, int] = {}
        self._load_attempted = False
        self.unavailable: str | None = None

    def _load(self) -> bool:
        """Import and fetch lazily; record failure instead of raising.

        The label order is read from the pinned ``config.json`` rather than
        hardcoded: the pin makes it constant, but reading it keeps the code
        honest if anyone re-points ``repo_id`` at a different export.
        """
        if self._load_attempted:
            return self._session is not None
        self._load_attempted = True
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            self.unavailable = (
                f"nli extra is not installed ({exc}); install compaction-guard[nli]"
            )
            return False
        try:
            model_path = hf_hub_download(
                self._repo_id, "onnx/model.onnx", revision=self._revision
            )
            tokenizer_path = hf_hub_download(
                self._repo_id, "tokenizer.json", revision=self._revision
            )
            config_path = hf_hub_download(
                self._repo_id, "config.json", revision=self._revision
            )
        except Exception as exc:
            self.unavailable = (
                f"could not obtain {self._repo_id}@{self._revision[:12]}: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        id2label = {
            str(label).casefold(): int(index)
            for index, label in config.get("id2label", {}).items()
        }
        required = {"contradiction", "entailment", "neutral"}
        if not required <= id2label.keys():
            self.unavailable = (
                f"{self._repo_id} config.json lacks NLI labels {sorted(required)}; "
                f"found {sorted(id2label)}"
            )
            return False
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=self._max_length)
        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = tokenizer
        self._session = session
        self._input_names = tuple(inp.name for inp in session.get_inputs())
        self._label_index = id2label
        return True

    def _probs(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """(contradiction, entailment, neutral) probabilities for one pair."""
        import numpy as np

        encoding = self._tokenizer.encode(premise, hypothesis)
        feed: dict[str, Any] = {}
        for name in self._input_names:
            if name == "input_ids":
                feed[name] = np.asarray([encoding.ids], dtype=np.int64)
            elif name == "attention_mask":
                feed[name] = np.asarray([encoding.attention_mask], dtype=np.int64)
            elif name == "token_type_ids":
                feed[name] = np.asarray([encoding.type_ids], dtype=np.int64)
        logits = [float(value) for value in self._session.run(None, feed)[0][0]]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        total = sum(exps)
        probs = [value / total for value in exps]
        return (
            probs[self._label_index["contradiction"]],
            probs[self._label_index["entailment"]],
            probs[self._label_index["neutral"]],
        )

    @staticmethod
    def _windows(view: SummaryView) -> tuple[tuple[str, frozenset[str], SurvivalSite], ...]:
        singles = [
            (sentence.text, frozenset(sentence.normalized.split()), sentence.site)
            for sentence in view.sentences
        ]
        pairs = []
        for left, right in zip(view.sentences, view.sentences[1:], strict=False):
            if left.site is right.site:
                pairs.append(
                    (
                        f"{left.text} {right.text}",
                        frozenset(left.normalized.split()) | frozenset(right.normalized.split()),
                        left.site,
                    )
                )
        return tuple(singles + pairs)

    def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
        if not self._load():
            return None
        probe = frozenset(
            token for token in normalize(invariant.text).split() if token not in STOPWORDS
        ) or frozenset(normalize(invariant.text).split())
        ranked = sorted(
            (
                (len(probe & tokens), index, text, site)
                for index, (text, tokens, site) in enumerate(self._windows(view))
            ),
            key=lambda row: (-row[0], row[1]),
        )
        candidates = [
            (text, site) for overlap, _index, text, site in ranked if overlap > 0
        ][: self._max_windows]
        if not candidates:
            # No window shares a single content token with the invariant.
            # Entailment from such windows is noise; absence is the lexical
            # and embedding layers' call to make, not this one's.
            return None

        hypothesis = template_hypothesis(invariant.text)
        value_id_intact = frozenset(
            (anchor.kind, anchor.normalized)
            for anchor in invariant.anchors
            if anchor.kind is not AnchorKind.MODALITY
        ) <= view_anchor_keys(view)

        best_paraphrase: tuple[float, str, SurvivalSite] | None = None
        best_weakened: tuple[float, str, SurvivalSite] | None = None
        for text, site in candidates:
            contra_f, entail_f, neutral_f = self._probs(text, hypothesis)
            if contra_f > entail_f and contra_f > neutral_f:
                return LayerVerdict(
                    kind=Kind.CONTRADICTED,
                    evidence=(
                        f'contradiction p={round(contra_f, 4):.4f} on window: '
                        f'"{clip(text, 120)}"'
                    ),
                    score=round(contra_f, 4),
                    site=site,
                )
            forward = entail_f > contra_f and entail_f > neutral_f
            contra_b, entail_b, neutral_b = self._probs(hypothesis, text)
            backward = entail_b > contra_b and entail_b > neutral_b
            if forward and backward:
                confidence = round(min(entail_f, entail_b), 4)
                if confidence >= self._min_entailment and value_id_intact:
                    if best_paraphrase is None or confidence > best_paraphrase[0]:
                        best_paraphrase = (confidence, text, site)
            elif backward and not forward:
                confidence = round(entail_b, 4)
                if best_weakened is None or confidence > best_weakened[0]:
                    best_weakened = (confidence, text, site)

        if best_paraphrase is not None:
            confidence, text, site = best_paraphrase
            return LayerVerdict(
                kind=Kind.PARAPHRASED,
                evidence=(
                    f'bidirectional entailment p={confidence:.4f}, value and identifier '
                    f'anchors intact, window: "{clip(text, 120)}"'
                ),
                score=confidence,
                site=site,
            )
        if best_weakened is not None:
            confidence, text, site = best_weakened
            return LayerVerdict(
                kind=Kind.WEAKENED,
                evidence=(
                    f'one-way entailment p={confidence:.4f} (constraint entails window, '
                    f'window does not entail constraint), window: "{clip(text, 120)}"'
                ),
                score=confidence,
                site=site,
            )
        return None
