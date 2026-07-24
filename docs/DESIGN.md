# Design decisions and rejected alternatives

This file records why the library is shaped the way it is, which
alternatives were considered and discarded, what each discarded alternative
would have bought, and every place where the implementation deliberately
departs from the letter of the spec it was built to. Nothing here is
aspirational; each decision names the mechanism that enforces it, and the
final section names what would prove each of the riskiest ones wrong.

## The spine: a wrapper around the user's compactor

Three architectures were designed in full before implementation:

1. **Standalone pinning** that owns the summariser call: the guard runs
   summarisation itself and injects the block into output it fully
   controls. It would have bought the strongest possible invariant (the
   guard sees and writes everything) at the price of restructuring the
   host's loop around the guard's own message type and handing it the
   summariser call. Prevention is identical to the wrapper's; adoption is
   not.
2. **The wrapper**: the host keeps its compactor, its loop, and its context
   type; the guard wraps the one call where compaction happens.
3. **A passive observer** that watches transcripts and raises alarms. It
   would have bought zero-risk adoption (it never touches context) and a
   telemetry surface, and it prevents nothing: compaction-induced errors
   cluster in the first few post-compaction steps, so an alarm that arrives
   after the next tool call is an autopsy.

The wrapper won. With `REPAIR` as the default policy it is constraint
pinning executed at the compaction boundary, which is the intervention with
published evidence behind it (Governance Decay, arXiv 2606.22528: 0%
violation with the constraint text present, restored by verbatim
re-injection), and its adoption cost is one changed line.

Grafted from the pinning design: the checksummed sentinel wire format,
`assert_present()` as a per-turn integrity check, the token budget with
loud refusal at `add()` time, and the `Mode` enum so degradation against
opaque compaction is a visible field on every report. Grafted from the
observer: survival-site attribution (`survived_in`, `at_risk`) and
`Report.to_json()`, because `PRESERVED` without a site lies: a constraint
that survived only in the kept-verbatim tail is one compaction from death.
The observer posture itself survives as `policy=WARN` plus `on_report`,
a policy value instead of a second architecture.

Discarded by name: sink frameworks, OTel, corpus stores, redaction policy,
replay CLIs, tap-mode compaction inference, and Slipstream-style async
validation. Async validation trades the guarantee for latency: the wrapper
must return context before the next step, or the first post-compaction
tool call runs unguarded, which is precisely when the published failure
concentrates.

## Why the core has zero dependencies

The core installs with nothing and imports nothing beyond the standard
library (`difflib`, `hashlib`, `unicodedata`, `re`, `dataclasses`, `json`).
Three reasons, in descending order of weight:

1. The library's product is trust in a verification step. Every dependency
   is a party to that trust, and a resolver conflict at install time is a
   user who never reaches the first `compact()`. A safety library that
   fails to install protects nothing.
2. The guard has to run in constrained places: hook processes (the
   Anthropic SessionStart helper is designed to run in one), CI sandboxes,
   air-gapped evaluation rigs. Stdlib-only means "Python 3.11 is present"
   is the entire requirements analysis.
3. Determinism. The lexical tier's verdicts are a function of committed
   code and nothing else, so the corpus numbers cannot drift under a
   transitive upgrade.

The rejected alternative was bundling a small paraphrase model in the
core, which would have bought recall on the paraphrased and contradicted
rows of the tier table. It was rejected because it converts the default
install from a pure function into a weights download, and because the
asymmetry is survivable: the core tier's failure mode is a false alarm
(`dropped`/`unverifiable` on faithful paraphrase), never a false
certification, and REPAIR converts the false alarm into benign
duplication. Users who need the recall install `[nli]`; the cost sits with
the person who chose it. The model layers hide behind lazy imports so the
bare install never pays for their existence.

## Soundness architecture: detectors escalate, they do not vote

The failure this library must never exhibit is a false certification: a
mutated, contradicted, or dropped constraint reported as preserved or
paraphrased. The detector design starts from that requirement.

A voting ensemble was rejected first. Votes average sound and unsound
opinions: cosine similarity scores "$500 cap" and "$5000 cap" as
near-identical, so in any weighted vote the embedding layer actively
argues for certifying a mutation, and the weights that suppress it need
calibration data that does not exist for arbitrary user constraints.
Blind spots here are not noise to be averaged out; they are structural,
known per layer, ahead of time. So the architecture encodes them as
executable restrictions instead:

1. The chain runs cheap to expensive (lexical, embedding, NLI, judge) and
   short-circuits on the first sound verdict. A layer that cannot decide
   returns `None` and the question escalates.
2. Each layer carries a `can_issue` whitelist enforced by `DetectorChain`
   at construction and at verdict time (`detectors/base.py`, the matrix as
   data). Embeddings may only confirm absence (`DROPPED`), because
   similarity can prove nothing about fidelity. NLI may not issue
   `MUTATED` or override a lexical one, because entailment models are
   numerically insensitive. A judge verdict stands only with a cited span
   that re-verifies against the actual text; a fabricated span degrades to
   `UNVERIFIABLE`.
3. Every finding records `decided_by` as a single rule name, so every
   aggregate number decomposes by layer and every verdict is recomputable
   from its evidence.

The corpus release gate closes the loop: `evidence/recompute.py` computes
the false-certify rate on every run, the gate is zero, and any
counterexample becomes a permanent fixture before the fix ships.

## Why repair verifies rather than trusting injection

Under REPAIR the guard does not stop at calling `codec.inject`. It
re-renders the returned context and asserts, by checksum, that exactly the
block it built is present. A miss raises `BlockIntegrityError`, hard,
always.

The alternative, trusting the injection call to have worked, was rejected
because `inject` is the single most user-extensible point in the library.
A custom codec that silently returns its input unchanged, appends the
block somewhere the renderer cannot see, or mangles it in transit would
turn REPAIR into fake protection while every report says `repaired=True`.
That is a silent failure of exactly the kind this library exists to catch,
inside the library itself. Verification costs one render and one hash;
the asymmetry between that cost and the cost of unverified fake repair is
the whole argument.

The same reasoning keeps integrity failures out of the findings taxonomy.
A finding describes summariser behaviour and flows into a report a host
may only log; converting a failed repair into an `UNVERIFIABLE` finding
would let it be ignored by default. Exceptions mean the machinery cannot
be trusted; findings mean the summariser did something. The guard never
converts one into the other.

## Why the checksum lives over the normalised interior

The sentinel block's sha256 is computed over the interior lines after
`normalize.normalize()` (NFKC, casefold, zero-width strip, whitespace
collapse, punctuation strip), not over raw bytes.

A raw-byte checksum was rejected because it screams on every re-wrapped
line: transports re-flow text, harnesses re-indent quoted blocks, and a
per-turn integrity check that false-alarms on legitimate transport gets
disabled by the first annoyed operator. No checksum at all was rejected
because it lets a "helpful" summariser trim the block silently. The
normalised interior is the point between: the digest survives what
transport legitimately does to text and catches what nothing legitimate
does, an edit to the words, ids, or values. The header and footer both
carry the digest so a truncated block cannot pass by keeping one marker.

The choice has a subtlety that adversarial review caught: whatever escape
convention puts multi-line invariant text onto single block lines must
itself survive normalisation, or the checksum goes blind to escape
structure. Conventional backslash escapes fail this: `normalize()` deletes
punctuation, so `"a\nb"` and `"a nb"` collapsed to identical normalised
bytes and shared a digest, a forgeable equivalence inside the integrity
primitive. The escape mark is therefore U+00A6 (broken bar): category So,
which the normalisation pipeline keeps, with no NFKC decomposition and no
case mapping, and the mark itself is escaped first. Distinct canonical
texts now get distinct checksums, and the round-trip property tests plus
forged-interior fixtures in `tests/test_render.py` and
`tests/test_adversarial.py` pin it.

## Why MUTATED outranks DROPPED

`SEVERITY_ORDER` places `MUTATED` above `DROPPED`, and the ordering is a
judgement encoded once so no caller re-derives it: a wrong live value
drives confident wrong action. An agent holding "$5000 cap" spends against
it without hesitating; an agent holding nothing at least sometimes asks,
because it knows it does not know. Commission beats omission in the damage
ranking even though omission is far more common in practice.

The same judgement is why `MUTATED` exists as its own kind rather than
folding into `CONTRADICTED`: value confusion is a failure class that
semantic layers are structurally blind to (embedding and NLI models score
`$500` against `$5000` as near-identical), so it must be owned by
deterministic anchor comparison, and the `can_issue` matrix bars the
semantic layers from ever overriding it. `UNVERIFIABLE` sits between
`WEAKENED` and `PARAPHRASED` in the same ordering: worse than any verified
survival, better than any verified loss, because it asserts nothing.

## Why UNVERIFIABLE is a verdict rather than an error

The core tier exhausts without a sound answer on 57 of the 300 corpus
cases. That is not a malfunction; it is the honest output of a tiered
system whose cheap tier refuses to guess. Three designs were considered
for that outcome:

1. Raise an exception. Rejected: exhaustion is routine on bare installs,
   so users would wrap every `compact()` in try/except and discard exactly
   the signal that says "install a deeper tier or accept uncertainty".
   Exceptions are reserved for machinery failures.
2. Default to a real verdict. Defaulting to `DROPPED` manufactures false
   alarms with a confident name; defaulting to `PRESERVED` is a false
   certification, the one prohibited failure.
3. A first-class verdict. Chosen: `UNVERIFIABLE` ranks in the severity
   order, counts in reports, and is consumed by policy: `fail_closed=True`
   makes it gate like a loss under RAISE, so a security-posture user can
   refuse uncertainty while the default user just sees it reported.

The same honesty rule generalises: a codec that cannot render produces
`UNVERIFIABLE` for every invariant with the failure named in the report,
because no verdict may be stronger than the rendering supports. The
UNOBSERVED integrations cap every finding at `UNVERIFIABLE` for the same
reason: nothing was inspectable, and the report says so instead of
guessing.

## Why auto-intake and the state ledger were discarded

**Auto-intake** (scanning turns for constraint-shaped text and pinning it
automatically) would have bought adoption with zero registration calls. It
was discarded because recognising constraints is a detection problem with
an unbounded recall claim: every missed pin is silent no-protection that
the user believes exists, which is worse than the disease. The library's
contract is crisp precisely because everything it asserts is something the
host explicitly declared. Host code calls `guard.add()`, and the README
shows where.

**The state ledger** (`set_state("spent", 310)` style mutable run state)
would have bought protection for evolving quantities, which genuinely do
decay under compaction. It was discarded because pinning machinery
guarantees immutability: the checksum is over canonical text, and a value
that changes every turn either churns the block (making the integrity
check meaningless as a drift detector) or freezes a stale number into
protected text (pinning a lie). Mutable run state is a ledger problem with
its own consistency semantics, one problem per repo, out of scope here
forever.

## Decisions taken during adversarial review

An adversarial review executed the library against constructed
counterexamples and confirmed four defect families (the fourth, the escape
mark, is covered in the checksum section above). The fixes are
load-bearing design, so they are recorded here; every counterexample is a
permanent fixture in `tests/fixtures/verdicts/` or
`tests/test_adversarial.py`.

### Near-verbatim matching is ordered, content-weighted, and window-local

The spec's rule 1 reads "token-set containment above threshold with all
anchors intact". Implemented literally, an unordered token bag over a
whole site certifies word-order inversions that reverse meaning ("queries
go to the primary, not replica_02" against the reverse), and site-level
anchor pooling lets a `$500` in a neighbouring sentence vouch for a
`$5000` mutation. Both are exactly the counterexample class the spec's own
falsification list names, and the zero false-certify gate outranks the
rule's phrasing. Rule 1b is therefore: ordered content-token recall
(`difflib.SequenceMatcher`, stopwords excluded, `autojunk=False`) at or
above the containment threshold, at window density of at least 0.75, over
windows of contiguous same-site sentences at most one sentence longer than
the invariant, with every anchor intact inside the window. Each term was
chosen against a specific counterexample: order kills inversions, content
weighting kills stopword-padded deletion (scope loss scoring above intact
text), density kills subsequences threaded across sentence boundaries.

### Anchor survival for rules 2 and 3 is topic-local

Value, identifier, and modality anchors are ordinary text. Tested against
the whole view, an unrelated sentence containing "never" masks a modality
loss, and a stray value in another constraint's restatement masks a
mutation; both were confirmed executable. Rules 2 and 3 now pool anchors
from topic-bearing sentences only (sentences sharing at least one topic
token with the invariant; all non-block sentences when the topic set is
empty). The cost is a false alarm when a value legitimately survives in a
sentence with no topic overlap, which is the side this library chooses
everywhere.

### The guard's own block is invisible to detectors

The injected sentinel block is appended as the last message, the position
compactors most commonly keep verbatim. Crediting it as a survival site,
even last in preference order, meant every compaction after the first
repair reported PRESERVED from the carried block, and the RAISE gate was
dead from the second compaction onward. Now the chain strips
reassertion-block sentences before any detector runs, and block survival
is reported only by the chain's own echo rule: when the outcome would
otherwise be DROPPED or UNVERIFIABLE and the invariant sits verbatim in
the carried block, the finding is PRESERVED with
`survived_in=reassertion_block` and `decided_by=chain.block_echo`.
Positive damage verdicts from the summary are never overridden by the
echo. DROPPED is downgraded because it would be false: the text
demonstrably is in context. The residue: at the core tier, a summary that
contradicts a block-carried constraint still reads PRESERVED-in-block,
because lexical detection cannot see contradiction with or without the
block; the NLI tier sees the contradiction in the stripped view and
reports it.

### `assert_present` tracks what the guard has issued

The spec requires both "raises on absence" and "safe to call every turn",
and its canonical loop calls `assert_present` before any compaction has
happened; implemented naively these cannot all hold, and the method
crashed on turn one of the spec's own example. Resolution: the guard
records the checksum of the last block it issued (via repair or
`reassertion_block()`). Before anything was issued, nothing is owed and
the check returns quietly. After issue, absence and edits raise as before.
The stale case (registry grew after the last issue) still raises, because
an intact old block is not protection for the constraint just added, but
the error now names both re-pin paths. `reassertion_block()` counts as
issuing on purpose: a host that asked for the block intends it to be in
context, which is what keeps the UNOBSERVED integrations' promise
checkable.

### The functional core accepts strings

`check()` takes `Invariant | str`, matching the `Guard` constructor, and
treats a bare string as one constraint. The previous signature took only
`Invariant` and died on the natural first call with an `AttributeError`
deep inside a generator. The spec's signature said `Sequence[Invariant]`;
widening is compatible and removes a confirmed trap.

### The shipped detector classes are public

The spec's public API list left `LexicalDetector`, `EmbeddingDetector`,
`NLIDetector`, and `JudgeDetector` private. But `Guard(detectors=...)`
replaces the default chain rather than extending it, so composing any
custom chain requires naming the lexical layer, and wiring a `JudgeFn` (a
headline feature) requires constructing `JudgeDetector`. An extension
surface whose only implementations are private is not a surface. The four
classes are exported from the package root; the spec's list should be
amended to match.

## Deliberate divergences from the spec's letter, for ratification

Beyond the review-driven fixes above, three places implement something
other than what the spec literally says. Each is flagged here rather than
silently absorbed.

1. **NLI weakening direction.** The spec's rule table defines premise =
   window, hypothesis = invariant, then "Forward only -> WEAKENED". The
   implementation issues WEAKENED when the invariant entails the window
   and not conversely: the window is a strictly weaker consequence of the
   constraint ("be careful with credentials" from a prohibition). A window
   that entails the full constraint is not a weakening of it. The
   calibration fixture
   `tests/fixtures/calibration/nli/weakened_backward_only.json` pins the
   implemented direction with measured behaviour at the pinned revision.
   The spec's rule table should be corrected or this line reverted
   knowingly.
2. **`Guard.check` bookkeeping.** The spec annotates `check()` "No side
   effects". The method updates `last_report`, fires `on_report`, and
   drains the removal ledger, because a host that only ever re-asserts
   (the pause-after-compaction flow) would otherwise lose its eviction
   trace and telemetry, which the spec's own `removed` contract promises.
   The module-level functional `check()` is pure as specified.
3. **Judge span re-verification.** The spec says "re-verifies by exact
   match". The implementation re-verifies by whole-token containment in
   normalize space: raw byte equality tolerates nothing transport
   legitimately does (re-flow, case) while token-boundary matching still
   rejects `$500` inside `$5000`. The judge gets exactly the library's own
   notion of "the same text", no looser.

## Evidence discipline

No number about this library appears anywhere that
`python evidence/recompute.py` does not produce from the committed corpus.
The corpus generator is deterministic (fixed seed, committed digest), the
recompute script refuses to run over a drifted corpus, and CI regenerates
both and diffs against the committed copies. The committed `results.json`
tier rows for the model layers were produced against the pinned revisions
named in the detector modules; the CI evidence job recomputes them on
linux, so any cross-platform numeric instability in the model stacks would
surface as a diff failure there rather than stand uncorrected. The judge
tier commits fixtures and an agreement script, never results, because a
model number this repo did not produce is a number it will not print.

## The riskiest decisions, and what falsifies each

1. **REPAIR as the default policy** rests on the presence-is-sufficient
   result: a re-injected verbatim block restores compliance even when a
   lossy or contradicting summary sits beside it. Falsified if episodes
   with an intact block plus a contradicting summary show material
   violation rates, the two-authorities residue turning out to be
   load-bearing. If that evidence appears, RAISE becomes the default and
   the README's central claim changes.
2. **The zero-dependency tier reports faithful paraphrase as
   `dropped`/`unverifiable`.** The bet is that REPAIR makes this harmless
   and honest asymmetry beats fake coverage. Falsified by users on good
   compactors abandoning the library over alarm noise rather than
   installing `[nli]`; the corpus paraphrase slice's verdict distribution
   is published so the noise level is inspectable in advance. The remedy
   would be a bundled paraphrase layer in the default detector set, and
   the zero-dependency section above would need rewriting, not
   footnoting.
3. **AutoCodec duck-typing across context shapes.** The bet is that
   refusing unrecognised shapes plus covering str, lists, OpenAI-style
   dicts, and `.content` objects spans real usage. Falsified by any
   mainstream shape that AutoCodec mis-renders while claiming success
   (certifying against garbage text); one confirmed mis-render is a
   release blocker, where mere refusal is a fixture request.
4. **Synchronous verification on the critical path.** The bet: lexical is
   microseconds, NLI is tens of milliseconds per invariant, and compaction
   is rare next to the summarisation LLM call it wraps. Falsified if
   realistic registries (tens of invariants, long summaries, windowed
   bidirectional NLI) push verification past about a second per
   compaction; the chain would then need a latency budget and an escape
   hatch, revisited here rather than papered over.
5. **The `can_issue` matrix as the whole defense against unsound
   layers.** The claim: no reachable path lets embeddings certify
   presence, NLI overturn a lexical `MUTATED`, or an uncited judge verdict
   stand, so no false certification can be laundered through the chain.
   Falsified by a single case where the false-certify rate leaves zero.
   That number is computed on every CI run, the release gate is that it
   stays zero, and any counterexample becomes a permanent fixture before
   the fix ships.
