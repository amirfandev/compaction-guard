"""The embedding layer: a similarity floor that confirms absence, nothing else.

Cosine similarity is negation-blind: "read-only queries only" and "writes are
fine here" share almost every token's neighborhood, and a static embedding
scores them as close. So this layer is one-directional by construction. Max
cosine of the invariant against every sentence below the floor upgrades a
lexical miss to DROPPED with a score attached; anything at or above the floor
escalates, because high similarity is compatible with paraphrase, weakening
and contradiction alike and certifying any of them from a cosine would be
laundering. The ``can_issue`` whitelist in the escalation matrix makes this
enforcement, not etiquette: the chain rejects a PRESERVED from this layer.

Requires the ``embeddings`` extra (model2vec, numpy). Without it, ``examine``
returns None and records why in ``unavailable``, which the chain quotes in
the exhaustion verdict, so a bare install degrades to UNVERIFIABLE with the
missing extra named instead of failing or, worse, pretending to have looked.

The model is pinned by commit so a fixed install produces identical scores
forever; a moving revision would make the committed calibration fixtures
unreproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..taxonomy import Kind
from .base import LayerVerdict, SummaryView

if TYPE_CHECKING:
    from ..invariant import Invariant

__all__ = [
    "EMBEDDING_REPO",
    "EMBEDDING_REVISION",
    "EmbeddingDetector",
]

EMBEDDING_REPO = "minishlab/potion-base-8M"
"""The static embedding model. Small, CPU-cheap, deterministic per revision."""

EMBEDDING_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
"""Pinned commit of EMBEDDING_REPO on the Hugging Face hub."""


class EmbeddingDetector:
    """DROPPED-only confirmation by similarity floor. See the module docstring.

    ``floor`` is a design constant, not a measured quantity: it trades alarm
    volume against UNVERIFIABLE volume and can never cause a false
    certification, because this layer certifies nothing. Set too high, near
    misses get called DROPPED (a false alarm the report absorbs); set too
    low, the layer rarely fires and more cases exhaust to UNVERIFIABLE. The
    calibration fixtures shipped with the extra pin the behaviour of the
    default at the pinned revision.
    """

    name: str = "embedding"
    can_issue: frozenset[Kind] = frozenset({Kind.DROPPED})

    def __init__(
        self,
        *,
        floor: float = 0.35,
        repo_id: str = EMBEDDING_REPO,
        revision: str = EMBEDDING_REVISION,
    ) -> None:
        if not 0.0 <= floor <= 1.0:
            raise ValueError("floor must be in [0, 1]")
        self._floor = floor
        self._repo_id = repo_id
        self._revision = revision
        self._model: Any = None
        self._load_attempted = False
        self.unavailable: str | None = None

    def _load(self) -> bool:
        """Import and fetch lazily; record failure instead of raising.

        Two failure modes collapse into ``unavailable`` on purpose. A missing
        extra is a configuration the user chose; a fetch failure on a cold
        cache is an environment problem. Either way this layer cannot answer,
        the chain must keep moving on the critical path, and UNVERIFIABLE
        with the reason attached is the honest output. The pinned revision
        goes through ``snapshot_download`` because model2vec's loader does
        not take a revision itself; downloading the pinned snapshot and
        loading from the local path is the pin.
        """
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True
        try:
            from huggingface_hub import snapshot_download
            from model2vec import StaticModel
        except ImportError as exc:
            self.unavailable = (
                f"embeddings extra is not installed ({exc}); "
                "install compaction-guard[embeddings]"
            )
            return False
        try:
            path = snapshot_download(
                repo_id=self._repo_id,
                revision=self._revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt"],
            )
            self._model = StaticModel.from_pretrained(path)
        except Exception as exc:  # degrade, never crash the compaction path
            self.unavailable = (
                f"could not load {self._repo_id}@{self._revision[:12]}: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        return True

    def examine(self, invariant: Invariant, view: SummaryView) -> LayerVerdict | None:
        if not self._load():
            return None
        if not view.sentences:
            # Nothing to compare against is the strongest possible absence
            # signal this layer can produce.
            return LayerVerdict(
                kind=Kind.DROPPED,
                evidence="no sentences in view; nothing to compare against",
                score=0.0,
                site=None,
            )
        import numpy as np

        texts = [invariant.text] + [sentence.text for sentence in view.sentences]
        vectors = np.asarray(self._model.encode(texts), dtype=np.float64)
        inv_vec = vectors[0]
        sent_vecs = vectors[1:]
        inv_norm = float(np.linalg.norm(inv_vec))
        if inv_norm == 0.0:
            max_sim = 0.0
        else:
            sent_norms = np.maximum(np.linalg.norm(sent_vecs, axis=1), 1e-12)
            sims = (sent_vecs @ inv_vec) / (sent_norms * inv_norm)
            max_sim = float(np.max(sims))
        max_sim = round(max_sim, 4)
        if max_sim < self._floor:
            return LayerVerdict(
                kind=Kind.DROPPED,
                evidence=(
                    f"max cosine {max_sim:.4f} below floor {self._floor:.2f} "
                    f"across {len(view.sentences)} sentences"
                ),
                score=max_sim,
                site=None,
            )
        # At or above the floor: some semantic trace exists. What kind of
        # trace is beyond a cosine; route onward.
        return None
