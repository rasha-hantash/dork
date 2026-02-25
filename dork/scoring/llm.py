from __future__ import annotations

import logging

import anthropic

from dork.config import ScoringConfig
from dork.models import CandidatePaper, Decision, RelevanceScore, ScoredPaper

log = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = """\
You are an AI engineering research analyst. Your job is to assess whether a research paper \
is relevant to practitioners building AI-powered systems.

A paper is relevant if it relates to any of these topics:
{topics}

Be strict: a paper about pure theoretical math or biology that happens to mention "neural" \
is NOT relevant. Focus on papers that would change how someone builds, deploys, or evaluates \
AI systems."""


class LLMScorer:
    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self.client = anthropic.Anthropic()

    def score_relevance(self, paper: CandidatePaper) -> RelevanceScore:
        topics_str = ", ".join(self.config.topics.include)
        system = RELEVANCE_SYSTEM_PROMPT.format(topics=topics_str)

        user_message = (
            f"Title: {paper.title}\n\n"
            f"Authors: {', '.join(paper.authors[:5])}\n\n"
            f"Abstract: {paper.abstract}\n\n"
            f"Categories: {', '.join(paper.categories)}"
        )

        log.debug("scoring relevance", extra={"source_id": paper.source_id, "title": paper.title})

        response = self.client.messages.parse(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            output_format=RelevanceScore,
        )

        if response.parsed_output is not None:
            return response.parsed_output

        log.warning("failed to parse relevance score", extra={"stop_reason": response.stop_reason})
        return RelevanceScore(score=0.0, topics=[], reasoning="structured output parse failure")

    def score_paper(self, paper: CandidatePaper) -> ScoredPaper:
        relevance = self.score_relevance(paper)

        if relevance.score >= self.config.relevance_threshold:
            decision = Decision.ACCEPT
        elif relevance.score >= self.config.borderline_threshold:
            decision = Decision.BORDERLINE
        else:
            decision = Decision.REJECT

        log.info(
            "scored paper",
            extra={
                "source_id": paper.source_id,
                "relevance": relevance.score,
                "decision": decision.value,
            },
        )

        return ScoredPaper(
            source=paper.source,
            source_id=paper.source_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            url=paper.url,
            published=paper.published,
            categories=paper.categories,
            relevance=relevance,
            decision=decision,
        )
