from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

from dork.config import DorkConfig
from dork.models import Decision, PipelineRun, ScoredPaper
from dork.output.markdown import generate_markdown, paper_path
from dork.output.pr import create_pr
from dork.scoring.llm import LLMScorer
from dork.sources.arxiv import ArxivSource
from dork.store import PaperStore

log = logging.getLogger(__name__)


def run_pipeline(config: DorkConfig, dry_run: bool = False) -> PipelineRun:
    run = PipelineRun(
        run_id=uuid.uuid4().hex[:12],
        started_at=datetime.now(),
        dry_run=dry_run,
    )

    store = PaperStore(config.data_path)
    scorer = LLMScorer(config.scoring)

    # --- Fetch ---
    candidates = []
    if config.sources.arxiv.enabled:
        source = ArxivSource(config.sources.arxiv)
        candidates.extend(source.fetch())

    run.sources_fetched = len(candidates)
    log.info("fetched candidates", extra={"count": run.sources_fetched})

    # --- Dedup ---
    new_candidates = [c for c in candidates if not store.is_seen(c.dedup_key)]
    run.candidates_after_dedup = len(new_candidates)
    log.info("after dedup", extra={"new": len(new_candidates), "dupes": run.sources_fetched - len(new_candidates)})

    if not new_candidates:
        log.info("no new papers to process")
        run.finished_at = datetime.now()
        store.append_run(run)
        return run

    # --- Score ---
    scored: list[ScoredPaper] = []
    for candidate in new_candidates:
        paper = scorer.score_paper(candidate)
        scored.append(paper)
        store.append_paper(paper)

    accepted = [p for p in scored if p.decision == Decision.ACCEPT]
    borderline = [p for p in scored if p.decision == Decision.BORDERLINE]
    rejected = [p for p in scored if p.decision == Decision.REJECT]

    run.accepted = len(accepted)
    run.borderline = len(borderline)
    run.rejected = len(rejected)

    log.info(
        "scoring complete",
        extra={"accepted": run.accepted, "borderline": run.borderline, "rejected": run.rejected},
    )

    if dry_run:
        _print_dry_run(accepted, borderline, rejected)
        run.finished_at = datetime.now()
        store.append_run(run)
        return run

    # --- Output ---
    papers_to_publish = accepted + borderline
    if not papers_to_publish:
        log.info("no papers above threshold, skipping PR")
        run.finished_at = datetime.now()
        store.append_run(run)
        return run

    kb_path = config.knowledge_base_path
    file_paths: list[Path] = []

    for paper in papers_to_publish:
        md_content = generate_markdown(paper, config)
        fp = paper_path(paper, kb_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(md_content)
        file_paths.append(fp)
        log.info("wrote paper", extra={"path": str(fp)})

    # --- PR ---
    pr_number = create_pr(papers_to_publish, rejected, file_paths, config)
    run.pr_number = pr_number

    run.finished_at = datetime.now()
    store.append_run(run)

    log.info(
        "pipeline complete",
        extra={"run_id": run.run_id, "pr_number": pr_number, "accepted": run.accepted},
    )
    return run


def _print_dry_run(
    accepted: list[ScoredPaper],
    borderline: list[ScoredPaper],
    rejected: list[ScoredPaper],
) -> None:
    print(f"\n{'='*60}")
    print(f"DRY RUN RESULTS")
    print(f"{'='*60}")

    if accepted:
        print(f"\nACCEPTED ({len(accepted)}):")
        for p in accepted:
            print(f"  [{p.relevance.score:.2f}] {p.title[:70]}")
            print(f"         topics: {', '.join(p.relevance.topics[:5])}")

    if borderline:
        print(f"\nBORDERLINE ({len(borderline)}):")
        for p in borderline:
            print(f"  [{p.relevance.score:.2f}] {p.title[:70]}")

    if rejected:
        print(f"\nREJECTED ({len(rejected)}):")
        for p in rejected[:10]:
            print(f"  [{p.relevance.score:.2f}] {p.title[:70]}")
        if len(rejected) > 10:
            print(f"  ... and {len(rejected) - 10} more")

    print(f"\n{'='*60}")
    print(f"Total: {len(accepted)} accepted, {len(borderline)} borderline, {len(rejected)} rejected")
