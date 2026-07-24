"""Framework adapters: the guard, attached to real interception surfaces.

Each module in this package adapts one stack's compaction surface to the
guard, and each states plainly how much protection that surface permits.
The Mode on every report is the honesty mechanism: OWNED where the adapter
runs the compactor and verifies its own injection by checksum, REASSERTED
where the summary text was inspectable but injection is the host's act,
UNOBSERVED where compaction is opaque and the only true claims are "a
block was re-asserted" and "nothing about the summary was verified". An
adapter that cannot see the summary reports every invariant UNVERIFIABLE
rather than clean; no adapter grades its own visibility upward.

Framework imports are lazy, inside functions. ``langchain`` is the one
module gated at import time, because everything in it needs framework
types. ``openai_agents`` and ``anthropic`` import cleanly without their
SDKs: their surfaces are a duck-typed protocol and plain strings, so both
run against fakes in an install with no extras, and a hook script needs
nothing but this library.

No adapter re-implements detection or policy. They sequence guard calls,
plus the same report bookkeeping ``Guard`` itself uses, so ``last_report``,
``on_report`` and the removal ledger behave identically whichever door a
compaction came through.
"""

from __future__ import annotations

from importlib.util import find_spec

from ..errors import CompactionGuardError

__all__ = ["MissingIntegrationError"]


class MissingIntegrationError(CompactionGuardError, ModuleNotFoundError):
    """An integration needs a framework package that is not installed.

    Subclasses ``ModuleNotFoundError`` so ``except ImportError`` probing by
    host code keeps working, and ``CompactionGuardError`` so the library's
    single catch-all net still catches it. The message always names the
    extra to install, because the fix is always the same one line.
    """


def _require(module: str, *, extra: str) -> None:
    """Refuse at import time when a required framework is absent.

    ``find_spec`` looks the module up without importing it, so the check is
    cheap and cannot trigger framework side effects. Integration modules
    call this at top level only when nothing in them works without the
    framework: failing at import with the install command beats failing at
    first use with a traceback that points into a lazy import.
    """
    if find_spec(module) is not None:
        return
    raise MissingIntegrationError(
        f"this integration requires the {module!r} package, which is not "
        f"installed. Install the extra: pip install 'compaction-guard[{extra}]'"
    )
