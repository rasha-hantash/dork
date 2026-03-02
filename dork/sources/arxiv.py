from __future__ import annotations

import logging
from datetime import date, timedelta

import arxiv

from dork.config import ArxivSourceConfig
from dork.models import CandidatePaper

log = logging.getLogger(__name__)


class ArxivSource:
    def __init__(self, config: ArxivSourceConfig) -> None:
        self.config = config
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3,
        )

    @property
    def name(self) -> str:
        return "arxiv"

    def fetch(self, since: date | None = None) -> list[CandidatePaper]:
        query = self._build_query()
        cutoff = since or (date.today() - timedelta(days=self.config.days_back))
        log.info("fetching arxiv papers", extra={"query": query, "since": cutoff.isoformat(), "max_results": self.config.max_results})

        search = arxiv.Search(
            query=query,
            max_results=self.config.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers: list[CandidatePaper] = []
        for result in self.client.results(search):
            pub_date = result.published.date()
            if pub_date < cutoff:
                continue
            paper = CandidatePaper(
                source="arxiv",
                source_id=result.get_short_id(),
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                published=pub_date,
                categories=result.categories,
            )
            papers.append(paper)

        log.info("fetched arxiv papers", extra={"count": len(papers)})
        return papers

    def _build_query(self) -> str:
        return " OR ".join(f"cat:{cat}" for cat in self.config.categories)
