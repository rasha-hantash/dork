from __future__ import annotations

import logging

import anthropic

from dork.config import ScoringConfig
from dork.models import BlogValueScore, CandidatePaper, ContentType, Decision, NoveltyScore, RelevanceScore, ScoredPaper

log = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = """\
You are an AI engineering research analyst. Your job is to assess whether a research paper \
is relevant to practitioners building AI-powered systems.

A paper is relevant if it relates to any of these topics:
{topics}

Be strict: a paper about pure theoretical math or biology that happens to mention "neural" \
is NOT relevant. Focus on papers that would change how someone builds, deploys, or evaluates \
AI systems."""

NOVELTY_SYSTEM_PROMPT = """\
You are an AI engineering research analyst assessing whether a paper offers genuinely new insights \
compared to existing work.

Given the candidate paper and the most similar papers already in the knowledge base, assess novelty \
on three dimensions:

1. **New technique or approach** — Does it introduce a method not covered by existing papers?
2. **Contradicts existing finding** — Does it present evidence against a previously accepted result?
3. **New empirical evidence at scale** — Does it validate/invalidate known ideas with significant new data?

Be strict: incremental improvements or minor variations on known techniques score low. \
Genuine paradigm shifts or surprising results score high."""


BLOG_RELEVANCE_SYSTEM_PROMPT = """\
You are an AI engineering content analyst. Your job is to assess whether a blog article \
or technical post is relevant to practitioners building AI-powered systems.

A post is relevant if it relates to any of these topics:
{topics}

Focus on practical value: tutorials, case studies, production lessons, tool comparisons, \
and engineering deep-dives. Marketing announcements, product launches without technical \
depth, and opinion pieces without evidence score low."""

BLOG_VALUE_SYSTEM_PROMPT = """\
You are an AI engineering content analyst assessing the practical value of a blog article \
for someone building AI systems.

Score on two dimensions:

1. **Actionability** (0–1) — Can the reader change their system or workflow based on this? \
   High: step-by-step guides, benchmarks with clear winners, production gotchas. \
   Low: high-level overviews, announcements, thought leadership.

2. **Depth** (0–1) — Does it go beyond surface-level? \
   High: shows code, shares metrics, explains trade-offs, covers failure modes. \
   Low: summarizes without adding insight, restates documentation.

Be strict: most blog posts are shallow marketing. Reserve high scores for genuinely \
useful technical content."""


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

    def score_novelty(
        self,
        paper: CandidatePaper,
        similar_papers: list[dict],
    ) -> NoveltyScore:
        """Score novelty by comparing against similar papers from the knowledge base."""
        similar_section = ""
        if similar_papers:
            parts = []
            for sp in similar_papers[:3]:
                parts.append(f"- {sp.get('title', 'Unknown')} (similarity: {sp.get('similarity', 0):.2f})")
            similar_section = "\n\nMost similar papers in knowledge base:\n" + "\n".join(parts)

        user_message = (
            f"Title: {paper.title}\n\n"
            f"Abstract: {paper.abstract}"
            f"{similar_section}"
        )

        log.debug("scoring novelty", extra={"source_id": paper.source_id})

        response = self.client.messages.parse(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=NOVELTY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            output_format=NoveltyScore,
        )

        if response.parsed_output is not None:
            return response.parsed_output

        log.warning("failed to parse novelty score", extra={"stop_reason": response.stop_reason})
        return NoveltyScore(score=0.5, contradiction=False, reasoning="structured output parse failure")

    def score_blog_relevance(self, paper: CandidatePaper) -> RelevanceScore:
        """Score relevance for blog articles using blog-specific prompt."""
        topics_str = ", ".join(self.config.topics.include)
        system = BLOG_RELEVANCE_SYSTEM_PROMPT.format(topics=topics_str)

        user_message = (
            f"Title: {paper.title}\n\n"
            f"Content: {paper.abstract}"
        )

        log.debug("scoring blog relevance", extra={"source_id": paper.source_id})

        response = self.client.messages.parse(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            output_format=RelevanceScore,
        )

        if response.parsed_output is not None:
            return response.parsed_output

        log.warning("failed to parse blog relevance", extra={"stop_reason": response.stop_reason})
        return RelevanceScore(score=0.0, topics=[], reasoning="structured output parse failure")

    def score_blog_value(self, paper: CandidatePaper) -> BlogValueScore:
        """Score actionability and depth for blog articles (replaces novelty)."""
        user_message = (
            f"Title: {paper.title}\n\n"
            f"Content: {paper.abstract}"
        )

        log.debug("scoring blog value", extra={"source_id": paper.source_id})

        response = self.client.messages.parse(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=BLOG_VALUE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            output_format=BlogValueScore,
        )

        if response.parsed_output is not None:
            return response.parsed_output

        log.warning("failed to parse blog value", extra={"stop_reason": response.stop_reason})
        return BlogValueScore(score=0.0, actionability=0.0, depth=0.0, reasoning="structured output parse failure")

    def score_paper(
        self,
        paper: CandidatePaper,
        similar_papers: list[dict] | None = None,
    ) -> ScoredPaper:
        # Dispatch scoring by content type
        if paper.content_type == ContentType.BLOG:
            return self._score_blog(paper)
        return self._score_paper(paper, similar_papers)

    def _score_blog(self, paper: CandidatePaper) -> ScoredPaper:
        """Score a blog article: blog relevance + blog value (no embedding/novelty)."""
        relevance = self.score_blog_relevance(paper)

        # Score value if relevance passes threshold
        blog_value = None
        if relevance.score >= self.config.borderline_threshold:
            blog_value = self.score_blog_value(paper)

        # Combined score uses value instead of novelty for blogs
        if blog_value is not None:
            combined = 0.6 * relevance.score + 0.4 * blog_value.score
        else:
            combined = relevance.score

        if combined >= self.config.relevance_threshold:
            decision = Decision.ACCEPT
        elif combined >= self.config.borderline_threshold:
            decision = Decision.BORDERLINE
        else:
            decision = Decision.REJECT

        log.info(
            "scored blog",
            extra={
                "source_id": paper.source_id,
                "relevance": relevance.score,
                "value": blog_value.score if blog_value else None,
                "combined": round(combined, 3),
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
            content_type=paper.content_type,
            relevance=relevance,
            decision=decision,
        )

    def _score_paper(
        self,
        paper: CandidatePaper,
        similar_papers: list[dict] | None = None,
    ) -> ScoredPaper:
        """Score a research paper: relevance + novelty."""
        relevance = self.score_relevance(paper)

        # Only run novelty scoring if paper passes relevance borderline threshold
        novelty = None
        if relevance.score >= self.config.borderline_threshold and similar_papers is not None:
            novelty = self.score_novelty(paper, similar_papers)

        # Compute decision using combined score when novelty is available
        if novelty is not None:
            combined = 0.6 * relevance.score + 0.4 * novelty.score
        else:
            combined = relevance.score

        if combined >= self.config.relevance_threshold:
            decision = Decision.ACCEPT
        elif combined >= self.config.borderline_threshold:
            decision = Decision.BORDERLINE
        else:
            decision = Decision.REJECT

        log.info(
            "scored paper",
            extra={
                "source_id": paper.source_id,
                "relevance": relevance.score,
                "novelty": novelty.score if novelty else None,
                "combined": round(combined, 3),
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
            content_type=paper.content_type,
            relevance=relevance,
            novelty=novelty,
            decision=decision,
        )
