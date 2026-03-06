# Add Blog/Medium RSS Support to Dork (via FreshRSS)

## Context

Dork currently scrapes research papers from arXiv, HuggingFace Daily Papers, and 5 institutional AI blog RSS feeds (Anthropic, OpenAI, Google Research, Meta AI, BAIR). The RSS source uses `feedparser` directly — no feed management, caching, or full-content extraction.

The user wants to:

1. Expand Dork to discover and curate **blog articles** (Medium, Substack, etc.) — practical, tutorial-oriented content
2. Use **FreshRSS** as the unified feed backend for ALL RSS sources (institutional blogs + Medium/blog feeds)
3. Leverage FreshRSS's full-content extraction to bypass Medium's paywall (user has a subscription)

## Approach: FreshRSS as Feed Backend + Content-Type Discriminator

**Architecture:**

```
FreshRSS (feed mgmt + full content + polling)
    ↓ GReader API
Dork FreshRssSource adapter
    ↓ CandidatePaper (with content_type)
Scoring (dual-track: paper prompts vs blog prompts)
    ↓ ScoredPaper
Output → brain-os PR (articles/ + papers/ dirs, same PR)
```

**FreshRSS replaces Dork's `feedparser`-based RSS source entirely.** ArXiv and HuggingFace sources stay as-is (they use dedicated APIs, not RSS). All RSS feeds — institutional blogs, Medium tags, author feeds — are managed through FreshRSS's web UI and queried via its GReader API.

### Key design decisions

- **FreshRSS for all RSS** — move existing institutional feeds + new blog feeds into FreshRSS. Remove `feedparser` dependency. Manage subscriptions via FreshRSS UI, not `dork.toml`
- **FreshRSS categories → content_type** — use FreshRSS categories to map feeds: "research" category → `ContentType.PAPER`, "blogs" category → `ContentType.BLOG`. The adapter reads the category from the API response
- **Full content from FreshRSS** — enable FreshRSS's full-content extension for Medium feeds. The GReader API returns full article HTML, which Dork converts to plain text for scoring and summary generation
- **ContentType discriminator** — `ContentType` enum (`paper` | `blog`) flows through the pipeline. Each scoring/output stage adapts behavior based on content type
- **Reuse models** — add `content_type` field to `CandidatePaper`/`ScoredPaper` with `PAPER` default (backward compatible)
- **Dual-track scoring** — blog relevance prompt values actionable content; blog "novelty" becomes "practical value" scoring (actionability/depth/uniqueness, no reference set)
- **Same PR, separate sections** — papers and articles in one daily PR
- **Separate output directory** — blogs go to `articles/` in brain-os, papers stay in `papers/`
- **Feed strategy** — tag feeds (broad discovery) + author feeds (reliable writers) + publication feeds. `blog_topics` in scoring config acts as "search tags" for LLM scoring regardless of how articles are tagged on Medium

### FreshRSS feeds to configure

**Research category feeds** (existing, moved from dork.toml):

- `https://www.anthropic.com/feed.xml`
- `https://openai.com/blog/rss.xml`
- `https://blog.research.google/feeds/posts/default?alt=rss`
- `https://ai.meta.com/blog/rss/`
- `https://bair.berkeley.edu/blog/feed.xml`

**Blogs category feeds** (new):

- Tag feeds: `medium.com/feed/tag/{machine-learning,vector-database,llm,rag,data-engineering,observability,kubernetes,postgresql,nextjs,ai-agents,langchain,tanstack}`
- Author feeds: `medium.com/feed/@adlumal`, `medium.com/feed/@guichenchen`
- Publication feeds: `https://blog.langchain.dev/rss/`

## Files to Modify

### Diff 1: FreshRSS setup + Model/Config layer

**New: `docker-compose.yml`** (project root)

- FreshRSS container with volume for data persistence
- Expose on localhost port (e.g., 8080)
- Environment variables for initial admin setup

**`dork/models.py`**

- Add `ContentType(str, Enum)` with `PAPER = "paper"`, `BLOG = "blog"`
- Add `content_type: ContentType = ContentType.PAPER` to `CandidatePaper` and `ScoredPaper`
- Add `blog_candidates: int = 0`, `blog_accepted: int = 0` to `PipelineRun`

**`dork/config.py`**

- Add `FreshRssSourceConfig` with `url: str`, `api_password: str`, `research_category: str = "research"`, `blogs_category: str = "blogs"`
- Replace `rss: RssSourceConfig` with `freshrss: FreshRssSourceConfig` in `SourcesConfig`
- Add `blog_relevance_threshold: float = 0.5` and `blog_topics: ScoringTopicsConfig` to `ScoringConfig`

**`dork.toml`**

```toml
[sources.freshrss]
enabled = true
url = "http://localhost:8080"
api_password = ""  # set via DORK_FRESHRSS_API_PASSWORD env var
research_category = "research"
blogs_category = "blogs"

[sources.rss]
enabled = false  # deprecated, feeds moved to FreshRSS

[scoring.blog_topics]
include = [
    # All existing scoring.topics (prompting, rag, fine-tuning, etc.) plus:
    "vector search", "vector databases",
    "claude code", "language server protocol", "LSP",
    "change data capture", "real-time pipelines", "event streaming",
    "developer tools", "IDE integration", "code assistants",
    "production ML", "system design", "architecture patterns",
    "observability", "tracing", "monitoring", "OpenTelemetry",
    "Kubernetes", "k8s", "scalability", "distributed systems",
    "Postgres", "PostgreSQL", "database optimization",
    "agents", "multi-agent", "agent orchestration", "LangChain", "LangGraph",
    "LLM evals", "evaluation frameworks", "benchmarks",
    "Next.js", "TanStack", "React Server Components", "frontend architecture",
    "TanStack Router", "TanStack Query", "design systems",
]
```

### Diff 2: FreshRSS source adapter

**New: `dork/sources/freshrss.py`**

- `FreshRssSource` class implementing `SourceAdapter` protocol
- Authenticates via GReader API (`/api/greader.php/accounts/ClientLogin`)
- Fetches unread articles from specified categories via `/api/greader.php/reader/api/0/stream/contents/...`
- Maps FreshRSS category → `ContentType` (research → PAPER, blogs → BLOG)
- Extracts article content (HTML → plain text via `html.parser` stdlib, no extra dependency)
- For blog articles: uses full content from FreshRSS (which has full-content extension enabled)
- For research articles: extracts arXiv links from content (same as current RSS source)
- Marks fetched articles as read in FreshRSS after processing

**`dork/sources/rss.py`**

- Keep file but mark as deprecated (still works if `[sources.rss].enabled = true` for fallback)

**`dork/pipeline.py`**

- Add FreshRSS source fetching block (replacing/alongside RSS block)

### Diff 3: Blog scoring track

**`dork/scoring/llm.py`**

- Add `BLOG_RELEVANCE_SYSTEM_PROMPT` — values actionable insights, working code, production lessons over academic rigor
- Add `BLOG_VALUE_SYSTEM_PROMPT` — replaces novelty scoring for blogs, assesses actionability/depth/uniqueness (no reference set comparison)
- Split `score_paper` to dispatch: `_score_research_paper` (existing logic) vs `_score_blog` (blog prompts, `blog_topics`, `blog_relevance_threshold`, no `similar_papers`)

### Diff 4: Output integration

**`dork/output/markdown.py`**

- Add `BLOG_SUMMARY_SYSTEM_PROMPT` — uses "Key Points" / "Applicable When" / "Watch Out For" sections instead of "Key Findings" / "Practical Implications" / "Limitations"
- Branch `generate_markdown` on `content_type`
- Branch `paper_path` to route blogs to `articles/{YYYY}/{MM}/` instead of `papers/`
- Add `content_type` to frontmatter

**`dork/output/index.py`**

- Scan both `papers/` and `articles/` directories

**`dork/output/pr.py`**

- Same PR, but separate papers and articles into distinct sections in PR body
- Articles get their own checklist in the rejected section (for feedback loop)

## Verification

1. Start FreshRSS via `docker compose up -d`, configure feeds via UI
2. `dork run --dry-run` — verify FreshRSS articles are fetched and scored with correct content types
3. Check that blog articles get `content_type: "blog"` in `papers.jsonl`
4. Verify blog markdown goes to `articles/` not `papers/` in brain-os
5. Verify existing paper pipeline (arXiv, HF) is unchanged
6. Run `/ci` for lint + tests

## Progress

- [ ] Diff 1: FreshRSS setup + Model/Config layer
- [ ] Diff 2: FreshRSS source adapter
- [ ] Diff 3: Blog scoring track
- [ ] Diff 4: Output integration
- [ ] Verification: end-to-end dry-run test
