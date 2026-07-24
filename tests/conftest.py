"""Shared scaffolding: import path, the offline contract, fixture loading.

The suite runs against the src/ tree directly, so no editable install is
needed. Every test runs under the autouse ``no_network`` fixture: the core's
zero-network promise is a release claim, and a suite that could silently
reach a socket would be incapable of noticing the claim breaking.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse any socket use. Accidental network access fails the test loudly."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "a test attempted network access; this suite runs offline by contract"
        )

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def load_fixture_cases(subdir: str) -> list[tuple[str, dict[str, Any]]]:
    """All JSON fixture files under ``tests/fixtures/<subdir>``, sorted by path.

    Returns (relative path, parsed payload) pairs so parametrised tests get
    readable ids and a failing case names its file.
    """
    root = FIXTURES / subdir
    cases: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json")):
        with path.open(encoding="utf-8") as handle:
            cases.append((str(path.relative_to(root)), json.load(handle)))
    return cases
