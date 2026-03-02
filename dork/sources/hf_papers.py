from __future__ import annotations

import logging
import re
from datetime import date

import httpx

from dork.config import HuggingFaceSourceConfig
from dork.models import CandidatePaper

log = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api/daily_papers"
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


class HuggingFaceSource:
    def __init__(self, config: HuggingFaceSourceConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "huggingface"

    def fetch(self, since: date | None = None) -> list[CandidatePaper]:
        log.info("fetching huggingface daily papers")

        try:
            resp = httpx.get(HF_API_URL, params={"limit": 100}, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error("huggingface api error", extra={"error": str(e)})
            return []

        entries = resp.json()
        papers: list[CandidatePaper] = []

        for entry in entries:
            paper_data = entry.get("paper", {})
            arxiv_id = paper_data.get("id", "")
            title = paper_data.get("title", "")
            abstract = paper_data.get("summary", "")
            authors = [a.get("name", "") for a in paper_data.get("authors", []) if a.get("name")]
            published_str = paper_data.get("publishedAt", "")

            if not arxiv_id or not title:
                continue

            # Parse published date
            try:
                pub_date = date.fromisoformat(published_str[:10])
            except (ValueError, IndexError):
                pub_date = date.today()

            if since and pub_date < since:
                continue

            url = f"https://arxiv.org/abs/{arxiv_id}"

            paper = CandidatePaper(
                source="huggingface",
                source_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                published=pub_date,
            )
            papers.append(paper)

        log.info("fetched huggingface papers", extra={"count": len(papers)})
        return papers
