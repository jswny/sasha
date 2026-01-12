from __future__ import annotations

from typing import List, Optional, Tuple
import tomllib
from pydantic import BaseModel, ConfigDict, Field


class OpenRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_ms: int = 60000
    http_referer: Optional[str] = None
    x_title: Optional[str] = None


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matchup_name: str
    hand_count: int
    run_seed: int
    blinds: Tuple[int, int]
    starting_stack_bb: int = 100
    db_path: str = "tmp/arena.db"
    code_version: str = "dev"
    lease_batch_size: int = 1000
    run_id: Optional[str] = None


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model: str
    use_memory: bool = False


class MatchupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    a: str
    b: str


class ArenaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    agents: List[AgentConfig]
    matchups: List[MatchupConfig] = Field(default_factory=list)


def load_config(path: str) -> ArenaConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return ArenaConfig.model_validate(data)
