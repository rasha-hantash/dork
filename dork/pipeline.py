from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

from dork.config import DorkConfig
from dork.models import CandidatePaper, Decision, PipelineRun, ScoredPaper
from dork.output.index import generate_index
from dork.output.markdown import generate_markdown, paper_path
from dork.output.pr import create_pr
from dork.scoring.embeddings import fetch_embedding, max_similarity
from dork.scoring.llm import LLMScorer
from dork.scoring.reference_set import ReferenceSet
from dork.sources.arxiv import ArxivSource
from dork.sources.hf_papers import HuggingFaceSource
from dork.sources.rss import RssSource
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

    # --- Fetch from all enabled sources ---
    last_run = store.last_run_date()
    if last_run:
        log.info("last run date", extra={"date": last_run.isoformat()})

    candidates: list[CandidatePaper] = []

    if config.sources.arxiv.enabled:
        source = ArxivSource(config.sources.arxiv)
        candidates.extend(source.fetch(since=last_run))

    if config.sources.huggingface.enabled:
        source_hf = HuggingFaceSource(config.sources.huggingface)
        candidates.extend(source_hf.fetch(since=last_run))

    if config.sources.rss.enabled:
        source_rss = RssSource(config.sources.rss)
        candidates.extend(source_rss.fetch(since=last_run))

    run.sources_fetched = len(candidates)
    log.info("fetched candidates", extra={"count": run.sources_fetched})

    # --- Cross-source dedup ---
    seen_dedup_keys: set[str] = set()
    unique_candidates: list[CandidatePaper] = []
    for c in candidates:
        key = c.dedup_key
        if key not in seen_dedup_keys:
            seen_dedup_keys.add(key)
            unique_candidates.append(c)

    dupes_cross_source = len(candidates) - len(unique_candidates)
    if dupes_cross_source > 0:
        log.info("cross-source dedup", extra={"removed": dupes_cross_source})

    # --- Dedup against store (with version awareness) ---
    new_candidates: list[CandidatePaper] = []
    version_updates: dict[str, int] = {}  # dedup_key -> previous version

    for c in unique_candidates:
        key = c.dedup_key
        prev_version = store.seen_version(key)
        if prev_version is None:
            # Never seen
            new_candidates.append(c)
        elif c.arxiv_version > prev_version:
            # Newer version of a previously seen paper
            new_candidates.append(c)
            version_updates[key] = prev_version
            log.info(
                "version update detected",
                extra={"source_id": c.source_id, "old_version": prev_version, "new_version": c.arxiv_version},
            )

    run.candidates_after_dedup = len(new_candidates)
    log.info("after dedup", extra={"new": len(new_candidates), "updates": len(version_updates), "dupes": run.sources_fetched - len(new_candidates)})

    if not new_candidates:
        log.info("no new papers to process")
        run.finished_at = datetime.now()
        store.append_run(run)
        return run

    # --- Embedding pre-filter ---
    ref_set = ReferenceSet(config.data_path / "reference_set.jsonl")
    ref_embeddings = ref_set.embeddings
    has_references = len(ref_embeddings) > 0

    filtered_candidates: list[CandidatePaper] = []
    candidate_embeddings: dict[str, list[float]] = {}  # dedup_key -> embedding

    for candidate in new_candidates:
        arxiv_id = candidate.arxiv_id
        if not arxiv_id or not has_references:
            # No arXiv ID or no reference set → skip pre-filter
            filtered_candidates.append(candidate)
            continue

        embedding = fetch_embedding(arxiv_id)
        if embedding is None:
            # S2 API failure → skip pre-filter (graceful degradation)
            filtered_candidates.append(candidate)
            continue

        similarity = max_similarity(embedding, ref_embeddings)
        candidate_embeddings[candidate.dedup_key] = embedding

        if similarity < config.scoring.embedding_threshold:
            log.info(
                "embedding reject",
                extra={"source_id": candidate.source_id, "similarity": round(similarity, 3)},
            )
            run.embedding_rejected = (run.embedding_rejected or 0) + 1
            continue

        filtered_candidates.append(candidate)

    if run.embedding_rejected:
        log.info("embedding pre-filter", extra={"rejected": run.embedding_rejected, "passed": len(filtered_candidates)})

    # --- Score ---
    scored: list[ScoredPaper] = []
    for candidate in filtered_candidates:
        # Find similar papers for novelty scoring (using embedding if available)
        similar_papers = _find_similar_papers(candidate, candidate_embeddings, ref_set)

        paper = scorer.score_paper(candidate, similar_papers=similar_papers)

        # Flag version updates
        prev = version_updates.get(candidate.dedup_key)
        if prev is not None:
            paper.is_update = True
            paper.previous_version = prev

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

    # --- Generate topic index ---
    index_path = generate_index(kb_path)
    if index_path.exists():
        file_paths.append(index_path)

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


def _find_similar_papers(
    candidate: CandidatePaper,
    candidate_embeddings: dict[str, list[float]],
    ref_set: ReferenceSet,
) -> list[dict] | None:
    """Find most similar reference papers by embedding similarity for novelty scoring."""
    embedding = candidate_embeddings.get(candidate.dedup_key)
    if embedding is None:
        return None

    entries = ref_set.load()
    if not entries:
        return None

    from dork.scoring.embeddings import cosine_similarity

    scored_refs = []
    for entry in entries:
        ref_emb = entry.get("embedding")
        if not ref_emb:
            continue
        sim = cosine_similarity(embedding, ref_emb)
        scored_refs.append({
            "title": entry.get("title", ""),
            "arxiv_id": entry.get("arxiv_id", ""),
            "similarity": sim,
        })

    scored_refs.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_refs[:3]


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
            score_str = f"rel={p.relevance.score:.2f}"
            if p.novelty:
                score_str += f" nov={p.novelty.score:.2f} combined={p.combined_score:.2f}"
            update_tag = f" [UPDATED v{p.previous_version}→v{p.arxiv_version}]" if p.is_update else ""
            print(f"  [{score_str}] {p.title[:70]}{update_tag}")
            print(f"         topics: {', '.join(p.relevance.topics[:5])}")
            print(f"         reason: {p.relevance.reasoning}")

    if borderline:
        print(f"\nBORDERLINE ({len(borderline)}):")
        for p in borderline:
            score_str = f"rel={p.relevance.score:.2f}"
            if p.novelty:
                score_str += f" nov={p.novelty.score:.2f} combined={p.combined_score:.2f}"
            update_tag = f" [UPDATED v{p.previous_version}→v{p.arxiv_version}]" if p.is_update else ""
            print(f"  [{score_str}] {p.title[:70]}{update_tag}")
            print(f"         reason: {p.relevance.reasoning}")

    if rejected:
        print(f"\nREJECTED ({len(rejected)}):")
        for p in rejected[:10]:
            print(f"  [{p.relevance.score:.2f}] {p.title[:70]}")
            print(f"         reason: {p.relevance.reasoning}")
        if len(rejected) > 10:
            print(f"  ... and {len(rejected) - 10} more")

    print(f"\n{'='*60}")
    print(f"Total: {len(accepted)} accepted, {len(borderline)} borderline, {len(rejected)} rejected")
