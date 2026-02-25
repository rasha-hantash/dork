from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from git import Repo

from dork.config import DorkConfig
from dork.models import ScoredPaper
from dork.output.index import generate_index
from dork.output.markdown import generate_markdown, paper_path
from dork.store import PaperStore

log = logging.getLogger(__name__)

CHECKED_PATTERN = re.compile(r"^- \[x\] `([^`]+)`")


def run_feedback(config: DorkConfig, pr_number: int) -> list[ScoredPaper]:
    kb_path = config.knowledge_base_path
    repo_slug = _get_remote_repo(Repo(str(kb_path)))
    store = PaperStore(config.data_path)

    # Fetch PR body and branch name via gh
    pr_data = _fetch_pr(repo_slug, pr_number)
    if not pr_data:
        log.error("could not fetch PR", extra={"pr_number": pr_number})
        return []

    body = pr_data["body"]
    branch = pr_data["headRefName"]

    # Parse checked source_ids from the PR body
    checked_ids = _parse_checked_ids(body)
    if not checked_ids:
        log.info("no checked papers found in PR body")
        return []

    log.info("found checked papers", extra={"count": len(checked_ids), "ids": checked_ids})

    # Look up papers in the store
    papers: list[ScoredPaper] = []
    for source_id in checked_ids:
        paper = store.get_paper(source_id)
        if paper:
            papers.append(paper)
        else:
            log.warning("paper not found in store", extra={"source_id": source_id})

    if not papers:
        log.info("no matching papers found in store")
        return []

    # Checkout the PR branch, generate summaries, commit, push
    repo = Repo(str(kb_path))
    repo.git.fetch("origin", branch)
    repo.git.checkout(branch)

    file_paths: list[Path] = []
    for paper in papers:
        md_content = generate_markdown(paper, config)
        fp = paper_path(paper, kb_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(md_content)
        file_paths.append(fp)
        log.info("wrote paper", extra={"path": str(fp), "source_id": paper.source_id})

    # Regenerate topic index
    index_path = generate_index(kb_path)
    if index_path.exists():
        file_paths.append(index_path)

    # Stage and commit
    for fp in file_paths:
        rel = fp.relative_to(kb_path)
        repo.index.add([str(rel)])

    repo.index.commit(f"dork: accept {len(papers)} papers via feedback")
    repo.git.push("origin", branch)

    log.info("pushed feedback papers", extra={"count": len(papers), "branch": branch})
    return papers


def _fetch_pr(repo_slug: str, pr_number: int) -> dict | None:
    result = subprocess.run(
        ["gh", "api", f"repos/{repo_slug}/pulls/{pr_number}",
         "--jq", "{body: .body, headRefName: .head.ref}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("gh api failed", extra={"stderr": result.stderr})
        return None
    return json.loads(result.stdout)


def _parse_checked_ids(body: str) -> list[str]:
    ids: list[str] = []
    for line in body.splitlines():
        match = CHECKED_PATTERN.match(line.strip())
        if match:
            ids.append(match.group(1))
    return ids


def _get_remote_repo(repo: Repo) -> str:
    for remote in repo.remotes:
        if remote.name == "origin":
            url = remote.url
            if url.startswith("git@"):
                path = url.split(":")[-1]
            elif "github.com" in url:
                path = "/".join(url.split("github.com/")[-1].split("/")[:2])
            else:
                return url
            return path.removesuffix(".git")
    return ""
