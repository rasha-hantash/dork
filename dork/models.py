from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

ARXIV_URL_PATTERN = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")
ARXIV_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})")
ARXIV_VERSION_PATTERN = re.compile(r"v(\d+)")


def extract_arxiv_id(url: str, source: str, source_id: str) -> str | None:
    """Extract arXiv ID from URL or source metadata."""
    if source == "arxiv":
        match = ARXIV_ID_PATTERN.match(source_id)
        if match:
            return match.group(1)
    match = ARXIV_URL_PATTERN.search(url)
    if match:
        return match.group(1)
    return None


def extract_arxiv_version(source_id: str) -> int:
    """Extract version number from arXiv source_id (e.g. '2401.12345v2' → 2). Defaults to 1."""
    match = ARXIV_VERSION_PATTERN.search(source_id)
    if match:
        return int(match.group(1))
    return 1


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
    def arxiv_id(self) -> str | None:
        return extract_arxiv_id(self.url, self.source, self.source_id)

    @property
    def arxiv_version(self) -> int:
        if self.source == "arxiv":
            return extract_arxiv_version(self.source_id)
        return 1

    @property
    def dedup_key(self) -> str:
        aid = self.arxiv_id
        if aid:
            return f"arxiv:{aid}"
        return f"{self.source}:{self.source_id}"


class RelevanceScore(BaseModel):
    score: float = Field(ge=0, le=1)
    topics: list[str] = Field(default_factory=list)
    reasoning: str = ""


class NoveltyScore(BaseModel):
    score: float = Field(ge=0, le=1)
    contradiction: bool = False
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
    novelty: NoveltyScore | None = None
    decision: Decision
    is_update: bool = False
    previous_version: int | None = None
    scored_at: datetime = Field(default_factory=datetime.now)
    pr_number: int | None = None
    pr_status: str | None = None

    @property
    def arxiv_id(self) -> str | None:
        return extract_arxiv_id(self.url, self.source, self.source_id)

    @property
    def arxiv_version(self) -> int:
        if self.source == "arxiv":
            return extract_arxiv_version(self.source_id)
        return 1

    @property
    def dedup_key(self) -> str:
        aid = self.arxiv_id
        if aid:
            return f"arxiv:{aid}"
        return f"{self.source}:{self.source_id}"

    @property
    def combined_score(self) -> float:
        """Combined relevance + novelty score. Falls back to relevance only."""
        if self.novelty is None:
            return self.relevance.score
        return 0.6 * self.relevance.score + 0.4 * self.novelty.score


class PipelineRun(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    sources_fetched: int = 0
    candidates_after_dedup: int = 0
    accepted: int = 0
    borderline: int = 0
    rejected: int = 0
    embedding_rejected: int = 0
    pr_number: int | None = None
    dry_run: bool = False
