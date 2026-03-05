# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is dork

AI engineering research paper discovery pipeline. Fetches papers from arXiv, HuggingFace Daily Papers, and RSS feeds, scores them for relevance/novelty using Claude, and publishes accepted papers to a knowledge base repo via GitHub PRs.

## Commands

```bash
uv sync                              # Install dependencies
uv run dork run --dry-run             # Fetch + score without creating PR
uv run dork run                       # Full pipeline (creates PR to knowledge base)
uv run dork feedback <pr_number>      # Accept checked papers from a PR's rejected section
uv run dork seed-references           # Initialize reference set embeddings (required before first real run)
```

No test suite exists yet. No linter is configured.

## Architecture

### Pipeline flow (`pipeline.py`)

```
Fetch → Cross-source dedup → Store dedup (version-aware) → Embedding pre-filter → Score → Generate markdown → Create PR
```

1. **Fetch** — each enabled source (`sources/`) returns `CandidatePaper` instances
2. **Dedup** — first cross-source (by `dedup_key`, which normalizes to arXiv ID when possible), then against the JSONL store (with arXiv version tracking: v1→v2 treated as new)
3. **Embedding pre-filter** — fetches SPECTER v2 embeddings from Semantic Scholar, rejects candidates below `embedding_threshold` cosine similarity to reference set. Gracefully skips on API failure.
4. **Scoring** — `LLMScorer` uses Anthropic structured outputs (`messages.parse()` with Pydantic models):
   - Relevance scored for all papers against configured topics
   - Novelty scored only if relevance ≥ `borderline_threshold` AND embeddings available
   - Combined score: `0.6 * relevance + 0.4 * novelty` (relevance-only fallback)
   - Decision thresholds: ACCEPT ≥ 0.6, BORDERLINE ≥ 0.4, REJECT < 0.4
5. **Output** — generates markdown with YAML frontmatter, topic index, then creates a PR via `gh` CLI

### Key modules

| Module                     | Role                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `models.py`                | Pydantic models: `CandidatePaper`, `ScoredPaper`, `RelevanceScore`, `NoveltyScore`, `PipelineRun`, `Decision` enum |
| `config.py`                | TOML config loader → `DorkConfig` (Pydantic). Config file: `dork.toml`                                             |
| `store.py`                 | JSONL persistence (`papers.jsonl`, `runs.jsonl`). Tracks seen versions for dedup.                                  |
| `sources/base.py`          | `SourceAdapter` protocol: `name` property + `fetch(since)` method                                                  |
| `scoring/llm.py`           | `LLMScorer` — relevance + novelty scoring via Anthropic API structured outputs                                     |
| `scoring/embeddings.py`    | SPECTER v2 embedding fetching from Semantic Scholar + cosine similarity                                            |
| `scoring/reference_set.py` | 64 hardcoded landmark papers for novelty comparison. Stored in `data/reference_set.jsonl`                          |
| `output/pr.py`             | Branch creation, commit, push, `gh pr create`                                                                      |
| `output/markdown.py`       | LLM-generated paper summaries with YAML frontmatter                                                                |
| `output/index.py`          | Topic-grouped index at `papers/index.md` in the knowledge base                                                     |
| `feedback.py`              | Parses checked boxes from PR body, generates summaries for rescued papers                                          |

### Data files (in `data/`)

- `papers.jsonl` — all scored papers (one `ScoredPaper` JSON per line)
- `runs.jsonl` — pipeline run metadata (determines `last_run_date()` for incremental fetches)
- `reference_set.jsonl` — seeded reference papers with 768-dim SPECTER v2 embeddings

### External dependencies

- **Anthropic API** — `ANTHROPIC_API_KEY` env var required. Uses `messages.parse()` for structured outputs.
- **Semantic Scholar API** — SPECTER v2 embeddings (timeout: 15s, no API key needed)
- **GitHub CLI (`gh`)** — must be authenticated. Used for PR creation and feedback parsing.
- **Knowledge base repo** — configured in `dork.toml` as `general.knowledge_base_repo`. Must be a git repo with `origin` remote.

## Conventions

- Python 3.14+, managed with `uv`
- All data models are Pydantic `BaseModel` with strict field constraints (scores: `ge=0, le=1`)
- JSONL for all persistence (append-only)
- Dedup keys normalize to `arxiv:{id}` when an arXiv ID is detectable, otherwise `{source}:{source_id}`
- API failures in non-critical paths (embeddings, novelty) degrade gracefully — the pipeline continues without that signal
- CLI uses Click with a `@click.group()` entrypoint

## Adding a new source

1. Create `dork/sources/newsource.py` implementing the `SourceAdapter` protocol (see `base.py`)
2. Add a config class in `config.py` and wire it into `SourcesConfig`
3. Register the source in `pipeline.py`'s fetch section
