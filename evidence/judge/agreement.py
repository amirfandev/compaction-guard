"""Judge calibration: a labeled subset and an agreement script, no results.

The repo commits the fixtures and this script, never any judge numbers,
because judge output depends on a model this package refuses to ship or
call. A user who injects a ``JudgeFn`` runs::

    python evidence/judge/agreement.py --judge mymodule:my_judge

and gets agreement counts for their judge against ground truth, computed
through the shipped ``JudgeDetector`` (forced choice, span re-verification,
degradation to UNVERIFIABLE), so the number describes the judge as the
library would actually use it, not the raw model.

``--write-subset`` regenerates ``subset.jsonl`` from the deterministic
corpus generator; a committed subset that drifted from the generator refuses
to run, the same discipline ``recompute.py`` applies to the corpus.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for path in (str(ROOT / "src"), str(ROOT / "evidence")):
    if path not in sys.path:
        sys.path.insert(0, path)

from make_corpus import build_corpus  # noqa: E402

from compaction_guard.detectors.base import JudgeFn, SummaryView  # noqa: E402
from compaction_guard.detectors.judge import JudgeDetector  # noqa: E402
from compaction_guard.invariant import Invariant  # noqa: E402

SUBSET_PATH = Path(__file__).resolve().parent / "subset.jsonl"

# Five cases per ground-truth label, first occurrence order over the
# deterministic corpus. Small on purpose: judge calls cost money and the
# subset exists to estimate agreement, not to re-run the whole eval.
PER_LABEL = 5


def build_subset() -> list[dict[str, str]]:
    taken: Counter[str] = Counter()
    subset: list[dict[str, str]] = []
    for case in build_corpus():
        label = str(case["label"])
        if taken[label] >= PER_LABEL:
            continue
        taken[label] += 1
        subset.append(
            {
                "case_id": str(case["case_id"]),
                "label": label,
                "invariant": str(case["invariant"]),
                "summary": str(case["summary"]),
            }
        )
    return subset


def subset_jsonl(cases: list[dict[str, str]]) -> str:
    return "".join(
        json.dumps(case, sort_keys=True, ensure_ascii=False) + "\n" for case in cases
    )


def verify_subset() -> list[dict[str, str]]:
    cases = build_subset()
    text = subset_jsonl(cases)
    if not SUBSET_PATH.exists():
        sys.exit(
            f"{SUBSET_PATH} is missing. Run "
            "`python evidence/judge/agreement.py --write-subset` and commit it."
        )
    if SUBSET_PATH.read_text(encoding="utf-8") != text:
        sys.exit(
            f"{SUBSET_PATH} does not match the generator. Rerun with "
            "--write-subset, review the diff, and commit."
        )
    return cases


def load_judge(spec: str) -> JudgeFn:
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        sys.exit("--judge takes module:callable, e.g. myjudge:complete")
    module = importlib.import_module(module_name)
    judge = getattr(module, attribute)
    if not callable(judge):
        sys.exit(f"{spec} is not callable")
    return judge  # type: ignore[no-any-return]


def run_agreement(judge: JudgeFn) -> None:
    cases = verify_subset()
    detector = JudgeDetector(judge)
    agree = 0
    abstain = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for case in cases:
        invariant = Invariant.parse(case["invariant"])
        view = SummaryView.from_summary(case["summary"])
        verdict = detector.examine(invariant, view)
        predicted = "escalate" if verdict is None else verdict.kind.value
        confusion[(case["label"], predicted)] += 1
        if predicted == case["label"]:
            agree += 1
        if predicted == "unverifiable":
            abstain += 1
    total = len(cases)
    print(f"cases: {total}")
    print(f"agreement with ground truth: {agree}/{total}")
    print(f"degraded to unverifiable (failed contract, abstained): {abstain}")
    print("confusion (truth -> predicted):")
    for (truth, predicted), count in sorted(confusion.items()):
        print(f"  {truth} -> {predicted}: {count}")
    print(
        "Note: the judge cannot answer preserved or mutated by contract; "
        "those truths counting against agreement is expected and correct."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", help="module:callable implementing JudgeFn")
    parser.add_argument(
        "--write-subset",
        action="store_true",
        help="regenerate subset.jsonl from the corpus generator",
    )
    args = parser.parse_args()
    if args.write_subset:
        cases = build_subset()
        SUBSET_PATH.write_text(subset_jsonl(cases), encoding="utf-8")
        print(f"wrote {len(cases)} cases to {SUBSET_PATH}")
        return
    if not args.judge:
        parser.error("either --judge or --write-subset is required")
    run_agreement(load_judge(args.judge))


if __name__ == "__main__":
    main()
