"""Detection layers: the chain that decides what compaction did to each invariant.

``base`` holds the shared vocabulary (the ``Detector`` protocol, ``SummaryView``,
``LayerVerdict``, the escalation matrix) and ``DetectorChain``, which runs the
layers cheap to expensive and enforces each layer's whitelist. ``lexical`` is
the stdlib default and the only detector most installs will ever run.
``embedding`` and ``nli`` are optional extras behind lazy imports; ``judge``
wraps a caller-supplied completion callable and ships with no client.

Nothing is re-exported here on purpose: the package's public surface is defined
in ``compaction_guard/__init__.py``, and detector internals are addressed by
their module paths (``compaction_guard.detectors.lexical.LexicalDetector``).
"""
