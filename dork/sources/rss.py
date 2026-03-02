from __future__ import annotations

import logging
import re
from datetime import date
from time import mktime

import feedparser

from dork.config import RssSourceConfig
from dork.models import CandidatePaper

log = logging.getLogger(__name__)

ARXIV_URL_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")


class RssSource:
    def __init__(self, config: RssSourceConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "rss"

    def fetch(self, since: date | None = None) -> list[CandidatePaper]:
        papers: list[CandidatePaper] = []

        for feed_url in self.config.feeds:
            log.info("fetching rss feed", extra={"url": feed_url})
            try:
                fetched = self._fetch_feed(feed_url, since)
                papers.extend(fetched)
            except Exception as e:
                log.error("rss feed error", extra={"url": feed_url, "error": str(e)})

        log.info("fetched rss papers", extra={"count": len(papers)})
        return papers

    def _fetch_feed(self, feed_url: str, since: date | None) -> list[CandidatePaper]:
        feed = feedparser.parse(feed_url)
        papers: list[CandidatePaper] = []

        for entry in feed.entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")
            authors = _parse_authors(entry)

            if not title or not link:
                continue

            # Parse publication date
            pub_date = _parse_date(entry)
            if since and pub_date < since:
                continue

            # Check if this links to an arXiv paper
            arxiv_match = ARXIV_URL_RE.search(link)
            if arxiv_match:
                source_id = arxiv_match.group(1)
            else:
                # Use a hash of the URL as source_id for non-arXiv entries
                source_id = link

            paper = CandidatePaper(
                source="rss",
                source_id=source_id,
                title=title,
                authors=authors,
                abstract=summary,
                url=link,
                published=pub_date,
            )
            papers.append(paper)

        return papers


def _parse_authors(entry: dict) -> list[str]:
    """Extract author names from a feed entry."""
    # feedparser may put authors in different fields
    if "authors" in entry:
        return [a.get("name", "") for a in entry["authors"] if a.get("name")]
    if "author" in entry:
        return [entry["author"]]
    return []


def _parse_date(entry: dict) -> date:
    """Extract publication date from a feed entry."""
    for field in ("published_parsed", "updated_parsed"):
        parsed = entry.get(field)
        if parsed:
            try:
                return date.fromtimestamp(mktime(parsed))
            except (ValueError, OverflowError):
                pass
    return date.today()
