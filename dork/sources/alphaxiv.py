from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import arxiv
import httpx

from dork.config import AlphaXivSourceConfig
from dork.models import CandidatePaper

log = logging.getLogger(__name__)

ALPHAXIV_BASE = "https://www.alphaxiv.org"
ARXIV_ID_RE = re.compile(r"/abs/(\d{4}\.\d{4,5}(?:v\d+)?)")

# Browser-like headers to avoid 403 blocks.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; dork-paper-discovery/0.1; "
        "+https://github.com/rasha-hantash/dork)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class AlphaXivSource:
    def __init__(self, config: AlphaXivSourceConfig) -> None:
        self.config = config
        self.client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

    @property
    def name(self) -> str:
        return "alphaxiv"

    def fetch(self, since: date | None = None) -> list[CandidatePaper]:
        log.info("fetching alphaxiv trending papers")

        arxiv_ids = self._scrape_paper_ids()
        if not arxiv_ids:
            log.warning("no paper ids scraped from alphaxiv")
            return []

        log.info("scraped alphaxiv paper ids", extra={"count": len(arxiv_ids)})

        cutoff = since or (date.today() - timedelta(days=self.config.days_back))
        papers = self._fetch_metadata(arxiv_ids, cutoff)

        log.info("fetched alphaxiv papers", extra={"count": len(papers)})
        return papers

    # ------------------------------------------------------------------

    def _scrape_paper_ids(self) -> list[str]:
        """Fetch the alphaxiv explore page and extract arXiv IDs from /abs/ links."""
        all_ids: list[str] = []
        seen: set[str] = set()

        for sort in self.config.sorts:
            url = f"{ALPHAXIV_BASE}/?sort={sort}"
            try:
                resp = httpx.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                log.error("alphaxiv fetch error", extra={"url": url, "error": str(e)})
                continue

            for match in ARXIV_ID_RE.finditer(resp.text):
                raw_id = match.group(1)
                # Strip version suffix for dedup (we'll get latest from arXiv).
                base_id = re.sub(r"v\d+$", "", raw_id)
                if base_id not in seen:
                    seen.add(base_id)
                    all_ids.append(base_id)

        return all_ids[: self.config.max_results]

    def _fetch_metadata(
        self, arxiv_ids: list[str], cutoff: date
    ) -> list[CandidatePaper]:
        """Use the arXiv API to retrieve full metadata for discovered IDs."""
        search = arxiv.Search(id_list=arxiv_ids)
        papers: list[CandidatePaper] = []

        try:
            results = list(self.client.results(search))
        except Exception as e:
            log.error("arxiv api error for alphaxiv ids", extra={"error": str(e)})
            return []

        for result in results:
            pub_date = result.published.date()
            if pub_date < cutoff:
                continue

            paper = CandidatePaper(
                source="alphaxiv",
                source_id=result.get_short_id(),
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                published=pub_date,
                categories=result.categories,
            )
            papers.append(paper)

        return papers
