---
name: Bug report
about: Report a defect in compaction-guard
title: ""
labels: bug
assignees: ""
---

## What happened

A short description of the defect.

## False certification check

Did the guard report `preserved` or `paraphrased` for a constraint that was in
fact mutated, contradicted, or dropped? If yes, say so here and include the
exact invariant text and the summary text it was checked against. False
certification is treated as a release blocker, and confirmed cases become
permanent fixtures before the fix ships.

## Minimal reproduction

```python
# Smallest snippet that reproduces the problem.
# Include the registered invariant text and a stub compactor if relevant.
```

## The report line

Paste the output of `report.to_json()` for the failing compaction, if you have
it. It is a single JSON line and it usually answers most questions.

```
```

## Environment

- compaction-guard version:
- Python version:
- Installed extras (none, embeddings, nli, langchain, openai-agents, anthropic):
- Policy in use (REPAIR, RAISE, WARN):
- Context shape (str, list of dicts, content blocks, custom codec):
- Framework integration in use, if any:

## Expected behaviour

What you expected instead, and why.
