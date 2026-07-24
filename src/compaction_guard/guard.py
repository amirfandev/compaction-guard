"""Guard: the registry, the compaction boundary, and policy application.

This is the only stateful module in the package. Everything it holds is
either the registry (what must survive), the removal ledger (what was
evicted, with reasons), or the last report (what happened most recently).
Detection, rendering, diffing and budget arithmetic all live in stateless
modules; the guard sequences them and applies policy.

``compact()`` is the one call that matters. It renders the before side,
runs the user's compactor, renders the after side, attributes regions by
message digest, runs the detector chain per invariant, and then applies the
configured policy. Under REPAIR it re-injects the sentinel block rendered
from the current registry and verifies by checksum that the injection
landed. That verification is not optional ceremony: repair without proof of
repair is a silent failure of exactly the kind this library exists to
catch, so a missing or edited block after the guard's own write raises
``BlockIntegrityError`` instead of producing a finding.

Two failure classes stay strictly apart here. Findings describe summariser
behaviour and flow into reports; exceptions mean the machinery cannot be
trusted or refuses to proceed (budget overflow at ``add()``, a codec that
cannot write under REPAIR, a block that fails its checksum after repair).
The guard never converts one into the other.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Generic, TypeVar

from .budget import ensure_fits, measure
from .check import check as check_summary
from .check import examine_view
from .context import AutoCodec, ContextCodec, RenderedContext
from .detectors.base import Detector, DetectorChain
from .diff import build_view
from .errors import (
    BlockIntegrityError,
    CodecError,
    CompactionGuardError,
    DuplicateInvariantId,
    InvariantViolation,
)
from .invariant import Invariant
from .render import assert_block_present, expected_checksum, find_blocks, render_block
from .report import BlockBudget, CompactionReport, Finding
from .taxonomy import GATING_KINDS, Kind, Mode, Policy, Severity

__all__ = ["Guard"]

C = TypeVar("C")

# Findings synthesised when the codec cannot render carry this attribution
# instead of a detector rule name: no detector ran, and pretending one did
# would make the decided_by distribution lie about the layer that failed.
_CODEC_DECIDED_BY = "codec.render"

_EMPTY_REGISTRY_NOTE = "no invariants registered"


class Guard(Generic[C]):
    """Wraps the user's compactor and keeps registered constraints present.

    The guard is generic over the context type ``C`` and touches contexts
    only through its codec, so adoption is one changed line: hand
    ``compact()`` the same context and compactor the host already has. The
    default policy is REPAIR because re-injecting the registered text
    verbatim is the intervention with published evidence behind it
    (Governance Decay, arXiv 2606.22528) and it never breaks a healthy run.

    Instances are not thread-safe. The registry, removal ledger and last
    report are plain mutable state, sequenced by the caller's loop, which is
    how agent loops actually run compaction.
    """

    def __init__(
        self,
        invariants: Iterable[Invariant | str] = (),
        *,
        policy: Policy = Policy.REPAIR,
        detectors: Sequence[Detector] | None = None,
        codec: ContextCodec[C] | None = None,
        fail_closed: bool = False,
        max_block_tokens: int = 1024,
        token_estimator: Callable[[str], int] | None = None,
        on_report: Callable[[CompactionReport], None] | None = None,
    ) -> None:
        """Configure the guard and register the initial invariants.

        Initial registration goes through the same path as ``add()``, so a
        duplicate id or a budget overflow fails here, at the line that
        caused it, not at the first compaction mid-run. For the same reason
        a custom detector list is validated against the escalation matrix
        now: a layer declaring kinds outside its row is a construction bug,
        and surfacing it at the first ``compact()`` would attribute it to
        whatever the summariser happened to do.

        ``token_estimator=None`` keeps the deliberately pessimistic default
        from ``budget.py``; a caller with a real tokeniser passes one
        callable and every number the guard reports uses it.
        """
        self._policy = Policy(policy)
        self._detectors: tuple[Detector, ...] | None = (
            tuple(detectors) if detectors is not None else None
        )
        if self._detectors is not None:
            # Constructed for validation only: DetectorChain enforces the
            # escalation matrix at construction time, and failing here beats
            # failing inside the first compaction.
            DetectorChain(self._detectors)
        self._codec: ContextCodec[C] = codec if codec is not None else AutoCodec()
        self._fail_closed = fail_closed
        self._max_block_tokens = max_block_tokens
        self._estimator = token_estimator
        self._on_report = on_report
        self._registry: dict[str, Invariant] = {}
        self._removed: list[tuple[str, str]] = []
        self._last_report: CompactionReport | None = None
        # The checksum of the block this guard most recently handed out,
        # via repair or reassertion_block(). None until then, which is what
        # lets assert_present distinguish "no block owed yet" from "a block
        # was owed and is gone".
        self._issued_checksum: str | None = None
        for item in invariants:
            if isinstance(item, Invariant):
                self._register(item)
            else:
                self._register(Invariant.parse(item, token_estimator=self._estimator))

    # --- registry ---

    def add(
        self,
        text: str,
        *,
        id: str | None = None,
        severity: Severity = Severity.BLOCK,
        source: str | None = None,
    ) -> Invariant:
        """Register a constraint. Refuses loudly rather than ever truncating.

        Raises ``BudgetExceeded`` if the rendered sentinel block would pass
        ``max_block_tokens`` with this invariant included, before anything
        is committed to the registry. Raises ``DuplicateInvariantId`` on id
        collision, which with derived ids usually means the same constraint
        text was registered twice.

        A constraint added mid-run is pinned at the next ``compact()``. A
        host that calls ``assert_present`` every turn and cannot wait for
        the next real compaction should re-pin immediately:
        ``context = guard.compact(context, compactor=lambda c: c)``; until
        then ``assert_present`` reports the old block as stale, because an
        intact stale block is not protection for the constraint just added.
        """
        invariant = Invariant.parse(
            text,
            id=id,
            severity=Severity(severity),
            source=source,
            token_estimator=self._estimator,
        )
        return self._register(invariant)

    def remove(self, id: str, *, reason: str) -> None:
        """Evict one invariant. The only way out of the registry.

        The reason is required and non-empty because eviction must leave a
        trace: the (id, reason) pair rides in the next report's ``removed``
        field, so a run log shows not just what was protected but what
        stopped being protected and why. Removing an unregistered id raises,
        since recording an eviction that did not happen would fabricate the
        trace this method exists to keep.
        """
        if not reason.strip():
            raise ValueError(
                "remove() requires a non-empty reason; eviction must leave a trace"
            )
        if id not in self._registry:
            raise CompactionGuardError(
                f"no invariant with id {id!r} is registered; nothing was removed"
            )
        del self._registry[id]
        self._removed.append((id, reason))

    def invariants(self) -> tuple[Invariant, ...]:
        """The registry, in registration order."""
        return tuple(self._registry.values())

    def budget(self) -> BlockBudget:
        """Current token accounting for the sentinel block."""
        return measure(
            self.invariants(),
            max_tokens=self._max_block_tokens,
            estimator=self._estimator,
        )

    @property
    def last_report(self) -> CompactionReport | None:
        """The most recent report this guard emitted, or None before the first."""
        return self._last_report

    # --- the compaction boundary (OWNED mode) ---

    def compact(self, context: C, compactor: Callable[[C], C]) -> C:
        """Run the user's compactor, verify what survived, apply policy.

        The before side is rendered before the compactor runs, not after:
        compactors are allowed to mutate their input in place, and a diff
        against a mutated before side would attribute the compactor's own
        edits to the retained region. Exceptions from the compactor itself
        propagate untouched; the guard wraps the boundary, not the failure
        modes of the user's summariser call.

        With an empty registry the compactor's output is returned unchanged
        and the report says so. No block is injected and no findings are
        computed, because pretending to protect what nobody declared is
        fake protection.

        A codec that cannot render makes every invariant UNVERIFIABLE with
        the failure named in the report note; the run continues. Under
        REPAIR (and under RAISE once the gate passes) a codec that cannot
        inject, or a block that fails checksum verification after
        injection, raises instead: ``CodecError`` because a guard that
        cannot write cannot pin, ``BlockIntegrityError`` because repair
        without verified repair is the silent failure this library exists
        to catch. Hard failures emit no report and leave the removal ledger
        intact for the next report that does get emitted.
        """
        started = time.perf_counter()
        before, before_error = self._try_render(context)
        out = compactor(context)
        after, after_error = self._try_render(out)

        registered = self.invariants()
        chars_before = None if before is None else len(before.text)
        chars_after = 0 if after is None else len(after.text)

        if not registered:
            report = self._report(
                started,
                findings=(),
                chars_before=chars_before,
                chars_after=chars_after,
                repaired=False,
                block_checksum=None,
                note=_EMPTY_REGISTRY_NOTE,
            )
            self._emit(report)
            return out

        if before is not None and after is not None:
            view = build_view(before.segments, after.segments)
            findings = examine_view(registered, view, detectors=self._detectors)
            note = (
                f"codec skipped {after.skipped_blocks} unrenderable content block(s)"
                if after.skipped_blocks
                else None
            )
        else:
            failure = before_error if before_error is not None else after_error
            message = str(failure) if failure is not None else "codec produced no rendering"
            findings = self._unverifiable_findings(registered, message)
            note = f"codec render failed: {message}"

        if self._policy is Policy.WARN:
            report = self._report(
                started,
                findings=findings,
                chars_before=chars_before,
                chars_after=chars_after,
                repaired=False,
                block_checksum=None,
                note=note,
            )
            self._emit(report)
            return out

        if self._policy is Policy.RAISE:
            gating = self._gating(findings)
            if gating:
                report = self._report(
                    started,
                    findings=findings,
                    chars_before=chars_before,
                    chars_after=chars_after,
                    repaired=False,
                    block_checksum=None,
                    note=note,
                )
                self._emit(report)
                details = ", ".join(
                    f"{finding.invariant_id}={finding.kind.value}" for finding in gating
                )
                raise InvariantViolation(
                    f"policy RAISE refused this compaction: {len(gating)} gating "
                    f"finding(s) on BLOCK invariants: {details}. The compacted "
                    "context was not returned; the caller still holds the original.",
                    report,
                )

        # REPAIR, or RAISE with a clean gate. RAISE is a gate on top of
        # pinning, not instead of it: once nothing gates, it repairs.
        block = render_block(registered)
        checksum = expected_checksum(registered)
        repaired = self._codec.inject(out, block)
        final_text = self._codec.render(repaired)
        assert_block_present(final_text, checksum)
        self._issued_checksum = checksum
        report = self._report(
            started,
            findings=findings,
            chars_before=chars_before,
            chars_after=len(final_text),
            repaired=True,
            block_checksum=checksum,
            note=note,
        )
        self._emit(report)
        return repaired

    # --- pure verification (REASSERTED mode) ---

    def check(self, summary_text: str) -> CompactionReport:
        """Verify summary text against the registry. No compaction, no injection.

        Delegates to the functional core with this guard's detector list.
        The registry and the context are untouched; what this method does
        share with ``compact()`` is report bookkeeping: pending removals
        drain into the report, ``last_report`` updates, and ``on_report``
        fires. A host that only ever re-asserts (the pause-after-compaction
        flow) still gets a complete removal trace and telemetry stream.
        """
        report = check_summary(self.invariants(), summary_text, detectors=self._detectors)
        removed = self._drain_removed()
        if removed:
            report = replace(report, removed=removed)
        self._emit(report)
        return report

    # --- opaque-compaction support (UNOBSERVED mode) ---

    def reassertion_block(self) -> str:
        """The sentinel-wrapped invariant block, for host code to inject.

        For the places the guard cannot reach: SessionStart hooks, input
        items appended after a server-side compaction. Rendered fresh from
        the current registry on every call, so it is always current. An
        empty registry still renders a valid empty block; whether injecting
        one is worth the tokens is the host's call, and the host can ask
        ``invariants()`` first. Calling this counts as issuing a block:
        ``assert_present`` starts demanding presence afterwards, because a
        host that asked for the block intends it to be in context.
        """
        registered = self.invariants()
        block = render_block(registered)
        if registered:
            self._issued_checksum = expected_checksum(registered)
        return block

    def assert_present(self, context: C) -> None:
        """Cheap per-turn integrity check: is the block this guard owes intact.

        Safe to call every turn, including before the first compaction: a
        guard that has never issued a block (no repair yet, no
        ``reassertion_block()`` call) is owed nothing, so this returns
        without rendering. The canonical loop calls this unconditionally
        and must not crash on turn one.

        Once a block has been issued, absence or a checksum-breaking edit
        raises ``BlockIntegrityError``: something downstream trimmed or
        rewrote guard-owned text, a harness bug. The stale case also
        raises: when ``add()`` grew the registry after the last issue, the
        old block being intact is not protection for the new constraint,
        and the error says how to re-pin (run a compaction, a no-op
        ``guard.compact(context, compactor=lambda c: c)`` included, or
        inject a fresh ``reassertion_block()``). An empty registry returns
        without rendering: no block is owed, so there is nothing whose
        absence would mean anything.
        """
        registered = self.invariants()
        if not registered:
            return
        if self._issued_checksum is None:
            return
        text = self._codec.render(context)
        current = expected_checksum(registered)
        blocks = find_blocks(text)
        for block in blocks:
            if (
                block.header_checksum == current
                and block.footer_checksum == current
                and block.computed_checksum == current
            ):
                return
        issued = self._issued_checksum
        if issued != current and any(
            block.header_checksum == issued
            and block.footer_checksum == issued
            and block.computed_checksum == issued
            for block in blocks
        ):
            raise BlockIntegrityError(
                "the sentinel block in context is stale relative to the "
                f"registry: it verifies as sha256={issued[:12]}... but the "
                f"registry now renders sha256={current[:12]}.... A constraint "
                "added or removed since the last injection is not pinned yet. "
                "Re-pin by compacting (a no-op compactor works: "
                "guard.compact(context, compactor=lambda c: c)) or by "
                "injecting guard.reassertion_block() where the guard cannot "
                "write."
            )
        assert_block_present(text, current)

    # --- internals ---

    def _register(self, invariant: Invariant) -> Invariant:
        existing = self._registry.get(invariant.id)
        if existing is not None:
            if existing.text == invariant.text:
                detail = "the same text is already registered under this id"
            else:
                detail = f"already registered with different text: {existing.text!r}"
            raise DuplicateInvariantId(
                f"invariant id {invariant.id!r}: {detail}. Derived ids hash the "
                "normalised text, so cosmetic variants of one constraint collide "
                "here instead of pinning twice."
            )
        # Budget check before commit: on refusal the registry is exactly as
        # it was, so a caller catching BudgetExceeded can retry after a
        # remove() without wondering what state the failed add left behind.
        ensure_fits(
            (*self._registry.values(), invariant),
            max_tokens=self._max_block_tokens,
            estimator=self._estimator,
        )
        self._registry[invariant.id] = invariant
        return invariant

    def _try_render(self, context: C) -> tuple[RenderedContext | None, CodecError | None]:
        try:
            return self._render_details(context), None
        except CodecError as exc:
            return None, exc

    def _render_details(self, context: C) -> RenderedContext:
        """Render with segmentation when the codec offers it.

        ``render_details`` is not part of the ``ContextCodec`` protocol, so
        it is looked up duck-typed: ``AutoCodec`` provides it, and custom
        codecs that can segment may too. Codecs that cannot are treated as
        one segment, which degrades region attribution (everything after
        compaction looks inserted) but never invents message boundaries.
        """
        details_fn = getattr(self._codec, "render_details", None)
        if callable(details_fn):
            details = details_fn(context)
            if isinstance(details, RenderedContext):
                return details
        text = self._codec.render(context)
        return RenderedContext(text=text, segments=(text,), skipped_blocks=0)

    def _unverifiable_findings(
        self, invariants: tuple[Invariant, ...], message: str
    ) -> tuple[Finding, ...]:
        """One UNVERIFIABLE finding per invariant when rendering failed.

        No detector ran, so no verdict stronger than UNVERIFIABLE is sound:
        the design rule is that no verdict exceeds what the rendering
        supports, and here there was no rendering at all.
        """
        return tuple(
            Finding(
                invariant_id=invariant.id,
                kind=Kind.UNVERIFIABLE,
                severity=invariant.severity,
                decided_by=_CODEC_DECIDED_BY,
                evidence=f"codec render failed: {message}",
                score=None,
                survived_in=None,
                at_risk=False,
            )
            for invariant in invariants
        )

    def _gating(self, findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
        """The findings that gate under RAISE, honouring fail_closed.

        ``CompactionReport.gating`` deliberately excludes UNVERIFIABLE
        because the report does not carry the fail_closed flag; the guard
        is the one place that knows it, so the extension happens here.
        """
        kinds: frozenset[Kind] = GATING_KINDS
        if self._fail_closed:
            kinds = kinds | {Kind.UNVERIFIABLE}
        return tuple(
            finding
            for finding in findings
            if finding.severity is Severity.BLOCK and finding.kind in kinds
        )

    def _report(
        self,
        started: float,
        *,
        findings: tuple[Finding, ...],
        chars_before: int | None,
        chars_after: int,
        repaired: bool,
        block_checksum: str | None,
        note: str | None,
    ) -> CompactionReport:
        return CompactionReport(
            mode=Mode.OWNED,
            findings=findings,
            chars_before=chars_before,
            chars_after=chars_after,
            repaired=repaired,
            block_checksum=block_checksum,
            removed=self._drain_removed(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            note=note,
        )

    def _drain_removed(self) -> tuple[tuple[str, str], ...]:
        drained = tuple(self._removed)
        self._removed.clear()
        return drained

    def _emit(self, report: CompactionReport) -> None:
        # last_report is set before on_report runs so the callback observes
        # current state, and so a raising callback cannot lose the report.
        self._last_report = report
        if self._on_report is not None:
            self._on_report(report)
