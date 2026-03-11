from __future__ import annotations

from pathlib import Path

import tomli
from pydantic import BaseModel, Field


class ArxivSourceConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = Field(
        default_factory=lambda: ["cs.CL", "cs.AI", "cs.LG", "stat.ML", "cs.IR"]
    )
    max_results: int = 200
    days_back: int = 1


class HuggingFaceSourceConfig(BaseModel):
    enabled: bool = False


class RssSourceConfig(BaseModel):
    enabled: bool = False
    feeds: list[str] = Field(default_factory=list)


class AlphaXivSourceConfig(BaseModel):
    enabled: bool = False
    sorts: list[str] = Field(default_factory=lambda: ["Hot"])
    max_results: int = 100
    days_back: int = 7


class SourcesConfig(BaseModel):
    arxiv: ArxivSourceConfig = Field(default_factory=ArxivSourceConfig)
    huggingface: HuggingFaceSourceConfig = Field(default_factory=HuggingFaceSourceConfig)
    rss: RssSourceConfig = Field(default_factory=RssSourceConfig)
    alphaxiv: AlphaXivSourceConfig = Field(default_factory=AlphaXivSourceConfig)


class ScoringTopicsConfig(BaseModel):
    include: list[str] = Field(default_factory=list)


class ScoringConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    relevance_threshold: float = 0.6
    borderline_threshold: float = 0.4
    max_tokens: int = 1024
    embedding_threshold: float = 0.3
    novelty_weight: float = 0.4
    topics: ScoringTopicsConfig = Field(default_factory=ScoringTopicsConfig)


class OutputConfig(BaseModel):
    pr_batch: bool = True
    branch_prefix: str = "dork/daily"


class GeneralConfig(BaseModel):
    knowledge_base_repo: str
    data_dir: str = "data"
    log_level: str = "info"


class DorkConfig(BaseModel):
    general: GeneralConfig
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def knowledge_base_path(self) -> Path:
        return Path(self.general.knowledge_base_repo)

    @property
    def data_path(self) -> Path:
        return Path(self.general.data_dir)


def load_config(config_path: Path | None = None) -> DorkConfig:
    if config_path is None:
        config_path = Path("dork.toml")
    with open(config_path, "rb") as f:
        raw = tomli.load(f)
    return DorkConfig.model_validate(raw)
