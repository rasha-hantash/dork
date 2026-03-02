from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


def generate_index(kb_path: Path) -> Path:
    """Scan papers/**/*.md, group by topic, write papers/index.md."""
    papers_dir = kb_path / "papers"
    if not papers_dir.exists():
        log.info("no papers directory found, skipping index")
        return papers_dir / "index.md"

    # topic -> list of (title, relative_path, score)
    topic_map: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    paper_count = 0

    for md_file in sorted(papers_dir.rglob("*.md")):
        if md_file.name == "index.md":
            continue

        frontmatter = _parse_frontmatter(md_file)
        if not frontmatter:
            continue

        paper_count += 1
        title = frontmatter.get("title", md_file.stem)
        topics = frontmatter.get("topics", [])
        score = frontmatter.get("relevance_score", 0.0)
        rel_path = md_file.relative_to(papers_dir)

        for topic in topics:
            topic_map[topic].append((title, str(rel_path), score))

    # Sort topics alphabetically, papers within each topic by score descending
    lines = ["# Paper Index\n"]
    for topic in sorted(topic_map):
        lines.append(f"## {topic}\n")
        entries = sorted(topic_map[topic], key=lambda x: x[2], reverse=True)
        for title, rel_path, score in entries:
            lines.append(f"- [{title}]({rel_path}) — {score:.2f}")
        lines.append("")

    index_path = papers_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n")
    log.info("generated topic index", extra={"topics": len(topic_map), "papers": paper_count})
    return index_path


def _parse_frontmatter(md_file: Path) -> dict | None:
    """Extract YAML frontmatter from a markdown file."""
    try:
        text = md_file.read_text()
    except OSError:
        return None

    if not text.startswith("---"):
        return None

    end = text.find("---", 3)
    if end == -1:
        return None

    fm_block = text[3:end].strip()
    result: dict = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        try:
            result[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            result[key] = value

    return result
