from __future__ import annotations

import json
from pathlib import Path

from dork.models import PipelineRun, ScoredPaper


class PaperStore:
    def __init__(self, data_dir: Path) -> None:
        self.papers_path = data_dir / "papers.jsonl"
        self.runs_path = data_dir / "runs.jsonl"
        self.papers_path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_keys: set[str] | None = None

    def _load_seen_keys(self) -> set[str]:
        keys: set[str] = set()
        if not self.papers_path.exists():
            return keys
        with open(self.papers_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                keys.add(f"{record['source']}:{record['source_id']}")
        return keys

    @property
    def seen_keys(self) -> set[str]:
        if self._seen_keys is None:
            self._seen_keys = self._load_seen_keys()
        return self._seen_keys

    def is_seen(self, dedup_key: str) -> bool:
        return dedup_key in self.seen_keys

    def append_paper(self, paper: ScoredPaper) -> None:
        with open(self.papers_path, "a") as f:
            f.write(paper.model_dump_json() + "\n")
        self.seen_keys.add(paper.dedup_key)

    def append_run(self, run: PipelineRun) -> None:
        with open(self.runs_path, "a") as f:
            f.write(run.model_dump_json() + "\n")
