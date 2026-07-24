"""compaction-guard: keeps registered constraints present across context compaction.

When a long-running agent compacts its history, the summariser can silently
drop, weaken, or mutate the constraints it was given, and the agent then acts
unbound by rules it still appears to follow. This package wraps the user's own
compactor at the compaction boundary: it registers constraint text, diffs both
sides of every compaction, classifies what happened to each constraint, and
under the default REPAIR policy re-injects the registered text verbatim in a
checksummed sentinel block. The core is stdlib only and makes no network or
model calls; heavier detection layers are optional extras.

This module re-exports the public surface and contains no logic. Anything not
importable from here is an implementation detail, whatever its name looks
like.

The four shipped detector classes are exported alongside the ``Detector``
protocol because ``Guard(detectors=...)`` replaces the default chain rather
than extending it: appending a judge means restating the lexical layer, and
wiring a ``JudgeFn`` at all means constructing ``JudgeDetector``. An
extension surface whose only implementations are private is not a surface.
The embedding and NLI modules import their heavy dependencies lazily, so
these re-exports cost a bare install nothing.
"""

from .anchors import Anchor, AnchorKind
from .check import check
from .context import ContextCodec
from .detectors.base import (
    Detector,
    JudgeFn,
    LayerVerdict,
    Sentence,
    SummaryView,
    SurvivalSite,
)
from .detectors.embedding import EmbeddingDetector
from .detectors.judge import JudgeDetector
from .detectors.lexical import LexicalDetector
from .detectors.nli import NLIDetector
from .errors import (
    BlockIntegrityError,
    BudgetExceeded,
    CodecError,
    CompactionGuardError,
    DuplicateInvariantId,
    InvariantViolation,
)
from .guard import Guard
from .invariant import Invariant
from .report import BlockBudget, CompactionReport, Finding
from .taxonomy import GATING_KINDS, SEVERITY_ORDER, Kind, Mode, Policy, Severity

__all__ = [
    "GATING_KINDS",
    "SEVERITY_ORDER",
    "Anchor",
    "AnchorKind",
    "BlockBudget",
    "BlockIntegrityError",
    "BudgetExceeded",
    "CodecError",
    "CompactionGuardError",
    "CompactionReport",
    "ContextCodec",
    "Detector",
    "DuplicateInvariantId",
    "EmbeddingDetector",
    "Finding",
    "Guard",
    "Invariant",
    "InvariantViolation",
    "JudgeDetector",
    "JudgeFn",
    "Kind",
    "LayerVerdict",
    "LexicalDetector",
    "Mode",
    "NLIDetector",
    "Policy",
    "Sentence",
    "Severity",
    "SummaryView",
    "SurvivalSite",
    "check",
]
