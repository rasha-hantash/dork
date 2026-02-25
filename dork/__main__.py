from __future__ import annotations

import logging
from pathlib import Path

import click

from dork.config import load_config
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
    click.echo(f"  Fetched:    {result.sources_fetched}")
    click.echo(f"  New:        {result.candidates_after_dedup}")
    click.echo(f"  Accepted:   {result.accepted}")
    click.echo(f"  Borderline: {result.borderline}")
    click.echo(f"  Rejected:   {result.rejected}")
    if result.pr_number:
        click.echo(f"  PR:         #{result.pr_number}")


if __name__ == "__main__":
    cli()
