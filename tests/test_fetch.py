"""Smoke tests for `fetch_candidates()` extraction.

These verify the deterministic-fetch half of the pipeline returns a usable
list of Candidate dicts, without invoking any LLM scoring or output writes.
We monkey-patch the source adapters so the test is offline and fast.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from dork import pipeline
from dork.__main__ import parse_since
from dork.config import (
    AlphaXivSourceConfig,
    ArxivSourceConfig,
    DorkConfig,
    GeneralConfig,
    HuggingFaceSourceConfig,
    RssSourceConfig,
    ScoringConfig,
    SourcesConfig,
)
from dork.models import CandidatePaper


def _fake_config(data_dir: Path) -> DorkConfig:
    return DorkConfig(
        general=GeneralConfig(
            knowledge_base_repo=str(data_dir / "kb"),
            data_dir=str(data_dir),
            log_level="warning",
        ),
        sources=SourcesConfig(
            arxiv=ArxivSourceConfig(enabled=True, categories=["cs.CL"], max_results=2),
            huggingface=HuggingFaceSourceConfig(enabled=True),
            rss=RssSourceConfig(enabled=False),
            alphaxiv=AlphaXivSourceConfig(enabled=False),
        ),
        scoring=ScoringConfig(),
    )


def _arxiv_candidate(idx: int) -> CandidatePaper:
    return CandidatePaper(
        source="arxiv",
        source_id=f"2401.{10000 + idx}v1",
        title=f"Fake arXiv Paper {idx}",
        authors=["Alice", "Bob"],
        abstract="An abstract about prompting and evals.",
        url=f"https://arxiv.org/abs/2401.{10000 + idx}",
        published=date.today(),
        categories=["cs.CL"],
    )


def _hf_candidate(idx: int) -> CandidatePaper:
    return CandidatePaper(
        source="huggingface",
        source_id=f"2402.{20000 + idx}",
        title=f"Fake HF Daily Paper {idx}",
        authors=["Carol"],
        abstract="Abstract for HF daily paper.",
        url=f"https://arxiv.org/abs/2402.{20000 + idx}",
        published=date.today(),
    )


class _StubArxiv:
    def __init__(self, config):
        self.config = config

    @property
    def name(self) -> str:
        return "arxiv"

    def fetch(self, since=None):
        return [_arxiv_candidate(1), _arxiv_candidate(2)]


class _StubHF:
    def __init__(self, config):
        self.config = config

    @property
    def name(self) -> str:
        return "huggingface"

    def fetch(self, since=None):
        # Includes one duplicate with arxiv (same arxiv_id pattern → dedup_key)
        # to verify cross-source dedup.
        return [_hf_candidate(99)]


def test_fetch_candidates_returns_nonempty(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ArxivSource", _StubArxiv)
    monkeypatch.setattr(pipeline, "HuggingFaceSource", _StubHF)

    config = _fake_config(tmp_path)
    result = pipeline.fetch_candidates(config, since=date.today() - timedelta(days=1))

    assert result.sources_fetched == 3
    # No reference set on disk → embedding prefilter is a no-op.
    assert len(result.candidates) == 3
    assert result.embedding_rejected == 0


def test_fetch_candidates_cross_source_dedup(tmp_path, monkeypatch):
    # Both stubs emit the same arxiv id → dedup_key collides.
    same_id = "2401.10001"

    class _StubArxivDup(_StubArxiv):
        def fetch(self, since=None):
            return [
                CandidatePaper(
                    source="arxiv",
                    source_id=f"{same_id}v1",
                    title="dup",
                    authors=[],
                    abstract="",
                    url=f"https://arxiv.org/abs/{same_id}",
                    published=date.today(),
                )
            ]

    class _StubHFDup(_StubHF):
        def fetch(self, since=None):
            return [
                CandidatePaper(
                    source="huggingface",
                    source_id=same_id,
                    title="dup",
                    authors=[],
                    abstract="",
                    url=f"https://arxiv.org/abs/{same_id}",
                    published=date.today(),
                )
            ]

    monkeypatch.setattr(pipeline, "ArxivSource", _StubArxivDup)
    monkeypatch.setattr(pipeline, "HuggingFaceSource", _StubHFDup)

    config = _fake_config(tmp_path)
    result = pipeline.fetch_candidates(config, since=date.today() - timedelta(days=1))

    assert result.sources_fetched == 2
    # Cross-source dedup collapses both arxiv:2401.10001 entries.
    assert len(result.candidates) == 1


def test_to_json_dict_includes_derived_fields():
    c = _arxiv_candidate(7)
    d = c.to_json_dict()
    # Core fields
    assert d["source"] == "arxiv"
    assert d["title"] == "Fake arXiv Paper 7"
    assert d["authors"] == ["Alice", "Bob"]
    assert d["published"] == date.today().isoformat()
    # Derived fields surfaced for downstream scorers
    assert d["arxiv_id"] == "2401.10007"
    assert d["arxiv_version"] == 1
    assert d["dedup_key"] == "arxiv:2401.10007"


def test_parse_since_durations():
    today = date.today()
    assert parse_since(None) is None
    assert parse_since("") is None
    assert parse_since("24h") == today - timedelta(days=1)
    assert parse_since("48h") == today - timedelta(days=2)
    assert parse_since("7d") == today - timedelta(days=7)
    assert parse_since("2w") == today - timedelta(days=14)
    assert parse_since("2026-05-01") == date(2026, 5, 1)


def test_parse_since_rejects_garbage():
    with pytest.raises(Exception):
        parse_since("not-a-date")
