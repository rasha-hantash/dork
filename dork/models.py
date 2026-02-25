from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    ACCEPT = "accept"
    BORDERLINE = "borderline"
    REJECT = "reject"


class CandidatePaper(BaseModel):
    source: str
    source_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: date
    categories: list[str] = Field(default_factory=list)

    @property
    def dedup_key(self) -> str:
        return f"{self.source}:{self.source_id}"


class RelevanceScore(BaseModel):
    score: float = Field(ge=0, le=1)
    topics: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ScoredPaper(BaseModel):
    source: str
    source_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: date
    categories: list[str] = Field(default_factory=list)
    relevance: RelevanceScore
    decision: Decision
    scored_at: datetime = Field(default_factory=datetime.now)
    pr_number: int | None = None
    pr_status: str | None = None

    @property
    def dedup_key(self) -> str:
        return f"{self.source}:{self.source_id}"


class PipelineRun(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    sources_fetched: int = 0
    candidates_after_dedup: int = 0
    accepted: int = 0
    borderline: int = 0
    rejected: int = 0
    pr_number: int | None = None
    dry_run: bool = False
