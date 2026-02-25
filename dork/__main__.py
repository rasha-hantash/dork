from __future__ import annotations

import logging
from pathlib import Path

import click

from dork.config import load_config
from dork.feedback import run_feedback
from dork.pipeline import run_pipeline


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
