from __future__ import annotations

from typing import Protocol

from dork.models import CandidatePaper


class SourceAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def fetch(self) -> list[CandidatePaper]: ...
