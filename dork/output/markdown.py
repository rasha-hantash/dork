from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import anthropic

from dork.config import DorkConfig
from dork.models import ScoredPaper

log = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
You are a technical writer for AI engineers. Given a research paper's metadata and abstract, \
generate a structured markdown summary.

Respond with ONLY the markdown content (no fences), using this exact structure:

# {title}

## TL;DR

One to two sentences explaining what this means for someone building AI systems.

## Key Findings

- Bullet points of the most important results

## Practical Implications

What to change in your systems based on this paper.

## Limitations

Where results might not generalize."""


def generate_markdown(paper: ScoredPaper, config: DorkConfig) -> str:
    client = anthropic.Anthropic()

    user_message = (
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors[:10])}\n"
        f"Abstract: {paper.abstract}"
    )

    log.debug("generating summary", extra={"source_id": paper.source_id})

    response = client.messages.create(
        model=config.scoring.model,
        max_tokens=2048,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    body = response.content[0].text
    frontmatter = _build_frontmatter(paper)
    return frontmatter + "\n" + body + "\n"


def paper_path(paper: ScoredPaper, kb_path: Path) -> Path:
    slug = _slugify(paper.title)
    short_id = paper.source_id.replace("/", "-").replace(".", "")
    yy = paper.published.strftime("%Y")
    mm = paper.published.strftime("%m")
    return kb_path / "papers" / yy / mm / f"{short_id}-{slug}.md"


def _build_frontmatter(paper: ScoredPaper) -> str:
    fm = {
        "title": paper.title,
        "authors": paper.authors[:10],
        "date": paper.published.isoformat(),
        "source": paper.source,
        "source_id": paper.source_id,
        "url": paper.url,
        "topics": paper.relevance.topics,
        "relevance_score": round(paper.relevance.score, 2),
    }
    lines = ["---"]
    for key, value in fm.items():
        lines.append(f"{key}: {json.dumps(value)}")
    lines.append("---")
    return "\n".join(lines)


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:80]
