"""Convention doc output — extract actionable rules from papers/articles and append to brain-os docs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from dork.config import DorkConfig
from dork.models import ContentType, ConventionDocEntry, ScoredPaper

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Part A — LLM extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are an AI engineering knowledge curator. Given a research paper or blog article, \
extract 0–3 actionable convention rules that a practitioner should follow.

A good convention rule:
- Is specific and actionable ("Prefer X over Y because Z")
- Would change how someone builds, deploys, or evaluates AI systems
- Is non-obvious — not something a competent engineer would already know
- Stands alone without needing the full paper context

If the content doesn't yield actionable rules (pure theory, incremental results, \
narrow domain), return an empty list. Most papers will yield 0 rules — that's fine.

You will be given the list of existing convention docs with their ## section headings. \
Place each rule under the most relevant doc and section. If no existing doc fits, \
suggest a new doc path (e.g. "ai-engineering/ai-engineering-conventions.md") and section.

Return structured JSON output."""

EXTRACTION_FEW_SHOT = """
Example input (paper abstract about retrieval-augmented generation chunk sizing):
"We find that chunk sizes of 256-512 tokens with 10% overlap consistently outperform \
both smaller and larger chunks across 8 QA benchmarks..."

Example output:
[
  {
    "target_doc": "rag-conventions.md",
    "section": "Chunking",
    "rule": "Default to 256–512 token chunks with ~10% overlap for QA workloads. Smaller chunks fragment context; larger chunks dilute relevance signal.",
    "citation_desc": "Chunk size benchmarking across 8 QA datasets"
  }
]

Example input (paper about a novel attention mechanism with marginal improvements):
Output: []
"""


class ConventionExtractionResult(BaseModel):
    entries: list[ConventionDocEntry] = Field(default_factory=list)


def extract_convention_entries(
    paper: ScoredPaper,
    kb_path: Path,
    config: DorkConfig,
) -> list[ConventionDocEntry]:
    """Use LLM to extract 0–3 convention doc entries from a scored paper/article."""
    doc_listing = _build_doc_listing(kb_path)
    if not doc_listing:
        log.warning("no convention docs found, skipping extraction")
        return []

    client = anthropic.Anthropic()

    content_label = "blog article" if paper.content_type == ContentType.BLOG else "research paper"
    user_message = (
        f"Content type: {content_label}\n"
        f"Title: {paper.title}\n"
        f"URL: {paper.url}\n\n"
        f"Abstract/Content:\n{paper.abstract}\n\n"
        f"---\n\n"
        f"Existing convention docs and their sections:\n{doc_listing}\n\n"
        f"{EXTRACTION_FEW_SHOT}"
    )

    log.debug("extracting convention entries", extra={"source_id": paper.source_id})

    response = client.messages.parse(
        model=config.scoring.model,
        max_tokens=config.scoring.max_tokens,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_format=ConventionExtractionResult,
    )

    if response.parsed_output is not None:
        entries = response.parsed_output.entries
        log.info(
            "extracted convention entries",
            extra={"source_id": paper.source_id, "count": len(entries)},
        )
        return entries

    log.warning("failed to parse convention extraction", extra={"source_id": paper.source_id})
    return []


def _build_doc_listing(kb_path: Path) -> str:
    """Build a listing of existing convention docs with their ## headings."""
    index_path = kb_path / "index.md"
    if not index_path.exists():
        return ""

    index_text = index_path.read_text()

    # Parse doc paths from the code block in index.md
    # Format: "path/to/doc.md — topic1, topic2"
    doc_paths: list[str] = []
    in_code_block = False
    for line in index_text.splitlines():
        if line.strip() == "```":
            in_code_block = not in_code_block
            continue
        if in_code_block and line.strip():
            # Extract path before " — "
            path = line.split(" — ")[0].strip()
            if path.endswith(".md"):
                doc_paths.append(path)

    # Read ## headings from each doc
    lines: list[str] = []
    for doc_rel in doc_paths:
        doc_file = kb_path / doc_rel
        if not doc_file.exists():
            continue
        headings = _extract_headings(doc_file)
        heading_str = ", ".join(headings) if headings else "(no sections)"
        lines.append(f"- {doc_rel}: {heading_str}")

    return "\n".join(lines)


def _extract_headings(doc_path: Path) -> list[str]:
    """Extract ## level headings from a markdown file."""
    headings: list[str] = []
    try:
        for line in doc_path.read_text().splitlines():
            if line.startswith("## "):
                headings.append(line[3:].strip())
    except OSError:
        pass
    return headings


# ---------------------------------------------------------------------------
# Part B — Convention doc writer
# ---------------------------------------------------------------------------

FOOTNOTE_RE = re.compile(r"^\[\^(\d+)\]:", re.MULTILINE)


def write_convention_entries(
    entries: list[ConventionDocEntry],
    paper: ScoredPaper,
    kb_path: Path,
) -> list[Path]:
    """Append convention entries to their target docs. Returns list of modified paths."""
    if not entries:
        return []

    # Group entries by target doc
    by_doc: dict[str, list[ConventionDocEntry]] = {}
    for entry in entries:
        by_doc.setdefault(entry.target_doc, []).append(entry)

    modified: list[Path] = []

    for doc_rel, doc_entries in by_doc.items():
        doc_path = kb_path / doc_rel
        is_new = not doc_path.exists()

        if is_new:
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            # Scaffold new doc
            doc_name = doc_path.stem.replace("-", " ").replace("_", " ").title()
            doc_path.write_text(f"# {doc_name}\n")
            # Add to index.md
            _add_to_index(kb_path, doc_rel, doc_entries)
            log.info("created new convention doc", extra={"path": doc_rel})

        text = doc_path.read_text()

        # Find max existing footnote number
        footnote_nums = [int(m.group(1)) for m in FOOTNOTE_RE.finditer(text)]
        next_footnote = max(footnote_nums, default=0) + 1

        # Build citation based on content type
        citation = _build_citation(paper, next_footnote)

        for entry in doc_entries:
            text = _append_entry(text, entry, next_footnote, citation)
            next_footnote += 1

        doc_path.write_text(text)
        modified.append(doc_path)
        log.info(
            "wrote convention entries",
            extra={"doc": doc_rel, "count": len(doc_entries)},
        )

    return modified


def _build_citation(paper: ScoredPaper, footnote_num: int) -> str:
    """Build a footnote citation string."""
    if paper.content_type == ContentType.BLOG:
        slug = re.sub(r"[^a-z0-9]+", "-", paper.title.lower()).strip("-")[:40]
        return f"[^{footnote_num}]: article:{slug} {paper.url} \"{paper.title[:80]}\""
    else:
        arxiv_id = paper.arxiv_id or paper.source_id
        return f"[^{footnote_num}]: paper:{arxiv_id} {paper.url} \"{paper.title[:80]}\""


def _append_entry(
    text: str,
    entry: ConventionDocEntry,
    footnote_num: int,
    citation: str,
) -> str:
    """Append a rule under the specified ## section and add citation at end of file."""
    section_header = f"## {entry.section}"
    rule_line = f"- {entry.rule}[^{footnote_num}]"

    # Find the section
    section_idx = text.find(section_header)
    if section_idx == -1:
        # Section doesn't exist — append new section before footnotes (or at end)
        footnote_match = FOOTNOTE_RE.search(text)
        if footnote_match:
            insert_pos = footnote_match.start()
            text = text[:insert_pos] + f"\n{section_header}\n\n{rule_line}\n\n" + text[insert_pos:]
        else:
            text = text.rstrip("\n") + f"\n\n{section_header}\n\n{rule_line}\n"
    else:
        # Find end of section (next ## or end of file, but before footnotes)
        after_header = section_idx + len(section_header)
        next_section = re.search(r"\n## ", text[after_header:])
        footnote_match = FOOTNOTE_RE.search(text[after_header:])

        if next_section and footnote_match:
            insert_offset = min(next_section.start(), footnote_match.start())
        elif next_section:
            insert_offset = next_section.start()
        elif footnote_match:
            insert_offset = footnote_match.start()
        else:
            insert_offset = len(text) - after_header

        insert_pos = after_header + insert_offset
        text = text[:insert_pos].rstrip("\n") + f"\n{rule_line}\n" + text[insert_pos:]

    # Append citation at end of file
    text = text.rstrip("\n") + f"\n{citation}\n"

    return text


def _add_to_index(kb_path: Path, doc_rel: str, entries: list[ConventionDocEntry]) -> None:
    """Add a new doc entry to index.md."""
    index_path = kb_path / "index.md"
    if not index_path.exists():
        return

    topics = ", ".join(set(e.section.lower() for e in entries))
    new_line = f"{doc_rel} — {topics}"

    text = index_path.read_text()
    # Insert before the closing ``` of the code block
    # Find last ``` in the file
    last_fence = text.rfind("```")
    if last_fence == -1:
        text = text.rstrip("\n") + f"\n{new_line}\n"
    else:
        text = text[:last_fence] + new_line + "\n" + text[last_fence:]

    index_path.write_text(text)
