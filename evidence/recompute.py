"""Run every installed detector tier against the committed corpus. Write results.json.

This script is the only permitted source of numbers about this library. It
runs each install tier (core, core+embeddings, core+embeddings+nli) over the
committed corpus and writes ``evidence/results.json`` with per-kind precision
and recall, the full confusion matrix, the false-certify rate, the decided_by
distribution, and the pinning prevention check. A tier whose optional extra is
not importable, or whose model cannot load, is reported with
``status: "not_run"`` and the detector's own reason string; it is never
silently skipped, because a missing row and a clean row must not look alike.

Every case goes through ``Guard.compact`` with a stub compactor that returns
the case's committed after-side, so the measured pipeline is the shipped one:
codec rendering, diff-based site attribution, the detector chain, and REPAIR
injection. The prevention check asserts, on the same pass, that the canonical
invariant text is present in the returned context and that ``assert_present``
verifies the injected block. That is the pinning guarantee demonstrated as an
executable count rather than a citation.

Determinism: the corpus is regenerated in memory and compared byte-for-byte
against the committed file before anything runs; drift is a hard failure with
instructions, not a warning. results.json contains counts and rounded ratios
only, no durations and no timestamps, so two runs on the same install are
byte-identical and CI can diff the committed copy.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT / "evidence")):
    if path not in sys.path:
        sys.path.insert(0, path)

from make_corpus import CORPUS_PATH, SEEDS, build_corpus, corpus_jsonl  # noqa: E402

from compaction_guard.context import AutoCodec  # noqa: E402
from compaction_guard.detectors.base import (  # noqa: E402
    Detector,
    SummaryView,
    contains_tokens,
)
from compaction_guard.detectors.embedding import EmbeddingDetector  # noqa: E402
from compaction_guard.detectors.lexical import LexicalDetector  # noqa: E402
from compaction_guard.detectors.nli import NLIDetector  # noqa: E402
from compaction_guard.errors import BlockIntegrityError  # noqa: E402
from compaction_guard.guard import Guard  # noqa: E402
from compaction_guard.invariant import Invariant  # noqa: E402
from compaction_guard.normalize import normalize  # noqa: E402
from compaction_guard.taxonomy import Kind  # noqa: E402

RESULTS_PATH = ROOT / "evidence" / "results.json"

# Ground-truth labels cover the six things a summariser can do to a constraint.
# UNVERIFIABLE is a verdict, not an event, so it appears in prediction columns
# and never as a truth row.
TRUTH_KINDS = (
    Kind.PRESERVED,
    Kind.PARAPHRASED,
    Kind.WEAKENED,
    Kind.MUTATED,
    Kind.CONTRADICTED,
    Kind.DROPPED,
)
CERTIFY_KINDS = frozenset({Kind.PRESERVED, Kind.PARAPHRASED})
DAMAGE_KINDS = frozenset({Kind.MUTATED, Kind.CONTRADICTED, Kind.DROPPED})


def verify_corpus() -> tuple[list[dict[str, Any]], str]:
    """Regenerate the corpus and refuse to run if the committed file drifted.

    The committed jsonl is the corpus of record; the in-memory regeneration
    exists to prove the file still matches the generator. Any mismatch means
    someone edited one without the other, and numbers computed over an
    unverified corpus would be unreproducible by construction.
    """
    cases = build_corpus()
    text = corpus_jsonl(cases)
    if not CORPUS_PATH.exists():
        sys.exit(
            f"{CORPUS_PATH} is missing. Run `python evidence/make_corpus.py` "
            "and commit the output before recomputing."
        )
    committed = CORPUS_PATH.read_text(encoding="utf-8")
    if committed != text:
        sys.exit(
            f"{CORPUS_PATH} does not match the generator's output. "
            "Rerun `python evidence/make_corpus.py`, review the diff, and commit "
            "both the corpus and the regenerated results together."
        )
    return cases, sha256(text.encode("utf-8")).hexdigest()


def probe_unavailable(detector: Detector) -> str | None:
    """One tiny examine() call, then read the detector's own reason string.

    Layers behind extras degrade rather than raise: a failed import or model
    load makes ``examine`` return None and record why in ``unavailable``.
    Probing with a throwaway invariant surfaces that reason up front, so a
    tier can be marked not_run with the same message a user would see in an
    exhaustion finding.
    """
    invariant = Invariant.parse("The probe budget cap is $1.")
    view = SummaryView.from_summary("The probe budget cap is $1.")
    detector.examine(invariant, view)
    reason: str | None = getattr(detector, "unavailable", None)
    return reason


def build_tiers() -> list[tuple[str, tuple[Detector, ...], list[str]]]:
    """The three install tiers, each with its blocking reasons (empty = runnable).

    Tiers are cumulative because that is how the extras install: nli users
    have the lexical layer by definition and almost always embeddings too,
    and the eval's job is to price each install choice, not every subset.
    """
    embedding = EmbeddingDetector()
    nli = NLIDetector()
    embedding_reason = probe_unavailable(embedding)
    nli_reason = probe_unavailable(nli)

    tiers: list[tuple[str, tuple[Detector, ...], list[str]]] = [
        ("core", (LexicalDetector(),), []),
        (
            "core+embeddings",
            (LexicalDetector(), embedding),
            [r for r in (embedding_reason,) if r],
        ),
        (
            "core+embeddings+nli",
            (LexicalDetector(), embedding, nli),
            [r for r in (embedding_reason, nli_reason) if r],
        ),
    ]
    return tiers


def evaluate_tier(
    detectors: tuple[Detector, ...], cases: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one tier over every case through Guard.compact. Returns (metrics, prevention).

    A fresh Guard per case keeps registries independent; the compactor is a
    closure returning a deep copy of the committed after-side, so no case can
    leak mutations into another. Prevention is measured on the same contexts
    the metrics come from: the returned post-REPAIR context must contain the
    canonical text and pass assert_present.
    """
    confusion: Counter[tuple[str, str]] = Counter()
    decided_by: Counter[str] = Counter()
    survived_in: Counter[str] = Counter()
    at_risk = 0
    false_certify: list[str] = []
    present = 0
    assert_failures = 0

    for case in cases:
        guard: Guard[list[dict[str, str]]] = Guard(
            [str(case["invariant"])], detectors=detectors
        )
        before = copy.deepcopy(case["before"])
        after = case["after"]

        def compactor(
            _context: list[dict[str, str]], _after: Any = after
        ) -> list[dict[str, str]]:
            result: list[dict[str, str]] = copy.deepcopy(_after)
            return result

        out = guard.compact(before, compactor)
        report = guard.last_report
        assert report is not None
        finding = report.findings[0]

        truth = str(case["label"])
        predicted = finding.kind.value
        confusion[(truth, predicted)] += 1
        decided_by[finding.decided_by] += 1
        if finding.survived_in is not None:
            survived_in[finding.survived_in.value] += 1
        if finding.at_risk:
            at_risk += 1
        if Kind(truth) in DAMAGE_KINDS and finding.kind in CERTIFY_KINDS:
            false_certify.append(str(case["case_id"]))

        rendered = AutoCodec().render(out)
        if contains_tokens(normalize(rendered), normalize(str(case["invariant"]))):
            present += 1
        try:
            guard.assert_present(out)
        except BlockIntegrityError:
            assert_failures += 1

    per_kind: dict[str, dict[str, Any]] = {}
    all_kinds = [kind.value for kind in Kind]
    for kind in all_kinds:
        support = sum(count for (t, _p), count in confusion.items() if t == kind)
        predicted_n = sum(count for (_t, p), count in confusion.items() if p == kind)
        correct = confusion[(kind, kind)]
        per_kind[kind] = {
            "support": support,
            "predicted": predicted_n,
            "correct": correct,
            "precision": round(correct / predicted_n, 4) if predicted_n else None,
            "recall": round(correct / support, 4) if support else None,
        }

    matrix: dict[str, dict[str, int]] = {}
    for (truth, predicted), count in sorted(confusion.items()):
        matrix.setdefault(truth, {})[predicted] = count

    damage_total = sum(
        count for (t, _p), count in confusion.items() if Kind(t) in DAMAGE_KINDS
    )
    metrics = {
        "confusion": matrix,
        "per_kind": per_kind,
        "false_certify": {
            "definition": (
                "cases labeled mutated, contradicted or dropped that this tier "
                "verdicts as preserved or paraphrased"
            ),
            "count": len(false_certify),
            "rate": round(len(false_certify) / damage_total, 4) if damage_total else 0.0,
            "cases": sorted(false_certify),
        },
        "decided_by": dict(sorted(decided_by.items())),
        "survived_in": dict(sorted(survived_in.items())),
        "at_risk_findings": at_risk,
    }
    prevention = {
        "policy": "repair",
        "cases": len(cases),
        "canonical_text_present": present,
        "present_rate": round(present / len(cases), 4) if cases else 0.0,
        "assert_present_failures": assert_failures,
    }
    return metrics, prevention


def main() -> None:
    cases, digest = verify_corpus()

    labels: Counter[str] = Counter(str(case["label"]) for case in cases)
    scaffolds: Counter[str] = Counter(str(case["scaffold"]) for case in cases)
    operators: Counter[str] = Counter(str(case["operator"]) for case in cases)

    tiers_out: dict[str, Any] = {}
    prevention: dict[str, Any] | None = None
    for name, detectors, blockers in build_tiers():
        if blockers:
            tiers_out[name] = {
                "status": "not_run",
                "detectors": [detector.name for detector in detectors],
                "reasons": blockers,
            }
            continue
        metrics, tier_prevention = evaluate_tier(detectors, cases)
        tiers_out[name] = {
            "status": "ran",
            "detectors": [detector.name for detector in detectors],
            **metrics,
        }
        if prevention is None:
            # The prevention number is a property of REPAIR, not of the
            # detector stack; the first tier that runs (always core) owns it.
            prevention = tier_prevention

    results = {
        "schema_version": 1,
        "generated_by": "python evidence/recompute.py",
        "corpus": {
            "file": "evidence/corpus.jsonl",
            "sha256": digest,
            "cases": len(cases),
            "seeds": len(SEEDS),
            "labels": dict(sorted(labels.items())),
            "scaffolds": dict(sorted(scaffolds.items())),
            "operators": dict(sorted(operators.items())),
        },
        "tiers": tiers_out,
        "prevention": prevention,
        "note": (
            "Tiers marked not_run lacked their optional extra in the "
            "environment that produced this file. Install the extra and rerun "
            "python evidence/recompute.py to fill them in; committed numbers "
            "for a tier exist only if that tier actually ran."
        ),
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {RESULTS_PATH}")
    for name, tier in tiers_out.items():
        if tier["status"] == "ran":
            certify = tier["false_certify"]
            print(f"  {name}: ran, false-certify {certify['count']} ({certify['rate']})")
        else:
            print(f"  {name}: not run ({'; '.join(tier['reasons'])})")
    if prevention is not None:
        print(
            f"  prevention: {prevention['canonical_text_present']}/{prevention['cases']} "
            f"present after REPAIR, {prevention['assert_present_failures']} "
            "assert_present failures"
        )


if __name__ == "__main__":
    main()
