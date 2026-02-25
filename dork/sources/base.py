from __future__ import annotations

from datetime import date
from typing import Protocol

from dork.models import CandidatePaper


class SourceAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def fetch(self, since: date | None = None) -> list[CandidatePaper]: ...
