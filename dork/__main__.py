from __future__ import annotations

import json
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import click

from dork.config import load_config
from dork.feedback import run_feedback
from dork.pipeline import fetch_candidates, run_pipeline


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


_SINCE_RE = re.compile(r"^(?P<n>\d+)(?P<unit>[hdw])$")


def parse_since(value: str | None) -> date | None:
    """Parse a --since value into a date cutoff.

    Accepts:
      - None or empty → None (callers fall back to PaperStore.last_run_date())
      - Relative durations: "24h", "7d", "2w" (hours collapse to a day boundary
        since CandidatePaper.published is a date, not datetime)
      - ISO date: "2025-12-01"
    """
    if not value:
        return None
    today = date.today()
    match = _SINCE_RE.match(value)
    if match:
        n = int(match.group("n"))
        unit = match.group("unit")
        if unit == "h":
            # Round up to a day (we have date precision, not datetime).
            days = max(1, (n + 23) // 24)
        elif unit == "d":
            days = n
        else:  # 'w'
            days = n * 7
        return today - timedelta(days=days)
    # Try ISO date
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise click.BadParameter(
            f"--since must be a duration like '24h'/'7d'/'2w' or ISO date 'YYYY-MM-DD' (got {value!r})"
        ) from e


@click.group()
def cli() -> None:
    """dork — AI engineering research paper discovery pipeline."""


@cli.command()
@click.option("--dry-run", is_flag=True, help="Fetch and score without creating a PR.")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
def run(dry_run: bool, config_path: Path | None) -> None:
    """Run the paper discovery pipeline."""
    config = load_config(config_path)
    _setup_logging(config.general.log_level)

    result = run_pipeline(config, dry_run=dry_run)

    click.echo(f"\nRun {result.run_id} complete:")
    click.echo(f"  Fetched:           {result.sources_fetched}")
    click.echo(f"  New:               {result.candidates_after_dedup}")
    if result.embedding_rejected:
        click.echo(f"  Embedding reject:  {result.embedding_rejected}")
    click.echo(f"  Accepted:          {result.accepted}")
    click.echo(f"  Borderline:        {result.borderline}")
    click.echo(f"  Rejected:          {result.rejected}")
    if result.pr_number:
        click.echo(f"  PR:                #{result.pr_number}")


@cli.command()
@click.option(
    "--since",
    "since_str",
    default=None,
    help="Cutoff for published date. Examples: '24h', '7d', '2w', '2026-05-01'. Defaults to last run date.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write JSON to this file instead of stdout.",
)
@click.option(
    "--max",
    "max_count",
    type=int,
    default=None,
    help="Cap the number of candidates emitted (default: no cap).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit JSON (the only supported format right now; accepted for forward-compat).",
)
def fetch(
    since_str: str | None,
    config_path: Path | None,
    output_path: Path | None,
    max_count: int | None,
    as_json: bool,
) -> None:
    """Fetch candidate papers from all configured sources and emit JSON.

    This is the pure-fetcher entry point: no LLM scoring, no markdown
    generation, no PR creation. Use it from a cron job or scheduled-task
    skill that owns the scoring layer separately.

    Example:

        python -m dork fetch --since 24h --json
    """
    config = load_config(config_path)
    _setup_logging(config.general.log_level)

    # Route logs to stderr so stdout stays clean JSON.
    for h in logging.getLogger().handlers:
        h.stream = sys.stderr

    since = parse_since(since_str)
    result = fetch_candidates(config, since=since)

    candidates = result.candidates
    if max_count is not None:
        candidates = candidates[:max_count]

    payload = [c.to_json_dict() for c in candidates]

    # `as_json` is currently the only supported format. Accepted as a flag for
    # forward-compat with potential `--format ndjson` etc.
    _ = as_json
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
        click.echo(
            f"Wrote {len(payload)} candidates to {output_path} "
            f"(sources_fetched={result.sources_fetched}, "
            f"embedding_rejected={result.embedding_rejected})",
            err=True,
        )
    else:
        click.echo(rendered)


@cli.command()
@click.argument("pr_number", type=int)
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
def feedback(pr_number: int, config_path: Path | None) -> None:
    """Accept checked papers from a PR. Check boxes in the rejected section, then run this."""
    config = load_config(config_path)
    _setup_logging(config.general.log_level)

    papers = run_feedback(config, pr_number)

    if papers:
        click.echo(f"\nAccepted {len(papers)} papers:")
        for p in papers:
            click.echo(f"  - {p.title[:70]}")
        click.echo(f"\nPushed to PR #{pr_number}")
    else:
        click.echo("No checked papers found.")


@cli.command("seed-references")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
def seed_references(config_path: Path | None) -> None:
    """Seed the reference set with embeddings from Semantic Scholar."""
    config = load_config(config_path)
    _setup_logging(config.general.log_level)

    from dork.scoring.reference_set import ReferenceSet

    ref_set = ReferenceSet(config.data_path / "reference_set.jsonl")
    ref_set.seed()

    entries = ref_set.load()
    click.echo(f"Reference set: {len(entries)} papers with embeddings")


if __name__ == "__main__":
    cli()
