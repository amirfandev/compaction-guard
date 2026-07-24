# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-24

First release.

### Added

- `Guard`, generic over the caller's context type, with `REPAIR`, `RAISE`, and
  `WARN` policies applied at the compaction boundary.
- Checksummed sentinel block wire format with idempotent injection: every
  `compact()` exits with exactly one current block, regardless of what the
  compactor did.
- Failure taxonomy (`preserved`, `paraphrased`, `weakened`, `mutated`,
  `contradicted`, `dropped`, `unverifiable`) with survival-site attribution in
  every finding.
- Stdlib-only lexical detector in the core install; embedding and NLI
  detectors behind the `embeddings` and `nli` extras, loaded lazily.
- `check()` functional core, `assert_present()` per-turn integrity check,
  `reassertion_block()` for opaque compaction, and a JSONL-ready
  `CompactionReport.to_json()`.
- Deterministic evidence corpus and recompute script; the only source of any
  number in the README.
- Integrations for LangChain, the OpenAI Agents SDK, and the Anthropic
  Messages API, each behind its own extra.
- `py.typed` marker, public exports for the four shipped detector classes,
  per-extra calibration fixtures with measured scores at the pinned model
  revisions, a judge agreement script under `evidence/judge/`, integration
  tests against protocol fakes, and `docs/DESIGN.md`.

### Fixed

Findings from pre-release adversarial review, each pinned as a permanent
fixture:

- Near-verbatim detection now uses ordered content-token recall over short,
  dense sentence windows instead of an unordered token bag, closing false
  PRESERVED certifications on word-order inversions and on value anchors
  borrowed from neighbouring sentences.
- Value and modality anchor survival is tested against topic-bearing
  sentences only, so an unrelated occurrence of a common word ("never", a
  stray "$500") no longer masks a weakening or mutation.
- Detectors no longer see the guard's own carried sentinel block; block
  survival is reported by an explicit chain rule (`chain.block_echo`) that
  never overrides damage found in the summary. Previously a compactor that
  kept the tail blinded every compaction after the first repair.
- The sentinel escape mark is now U+00A6, which survives text
  normalisation, so distinct multi-line constraint texts can no longer
  collide to one block checksum.
- `assert_present` is quiet until the guard has issued a block, so the
  canonical every-turn loop no longer crashes before the first compaction;
  the stale-after-`add()` error now names the re-pin paths.
- Module-level `check()` accepts bare constraint strings, matching the
  `Guard` constructor.

[Unreleased]: https://github.com/amirfandev/compaction-guard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/amirfandev/compaction-guard/releases/tag/v0.1.0
