"""Pydantic-settings loader for all bot8 configuration.

Loads from environment variables (and `.env` if present). A single `Settings`
instance is the canonical source for every module — do NOT read env vars directly
anywhere else.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Treat empty env vars as not-set. Without this, a shell-exported
        # ANTHROPIC_API_KEY="" silently overrides the value in .env — which we
        # hit when Claude Code inherits an empty var across subprocesses.
        env_ignore_empty=True,
    )

    # --- Alpaca ---
    alpaca_api_key: SecretStr = Field(default=SecretStr(""))
    alpaca_secret_key: SecretStr = Field(default=SecretStr(""))
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # --- Paths ---
    data_dir: Path = Path("./data")
    # Optional override for the logs directory — useful when mounting a
    # separate Railway volume at e.g. `/app/logs`. When unset, logs go to
    # `{data_dir}/logs/` (the default pattern).
    logs_dir_override: Path | None = Field(default=None, alias="LOGS_DIR")

    # --- Logging ---
    log_level: str = "INFO"

    # --- Dashboard ---
    dashboard_port: int = 8000

    # --- NLP ---
    finbert_model: str = "ProsusAI/finbert"

    # --- Claude API (optional — only for --submit path) ---
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))

    # --- Dashboard auth (optional — empty = auth disabled for local dev) ---
    dashboard_api_key: SecretStr = Field(default=SecretStr(""))

    # --- Schedule (America/New_York local time) ---
    rebalance_time: str = "09:35"
    premarket_fetch_time: str = "07:00"
    premarket_score_time: str = "07:30"

    # --- Portfolio ---
    target_gross_exposure: float = 1.0
    max_net_exposure: float = 0.10
    long_decile: float = 0.1
    short_decile: float = 0.1
    max_position_weight: float = 0.02

    # --- Risk ---
    min_regime_exposure: float = 0.4
    max_drawdown_stop: float = 0.15

    @field_validator("data_dir")
    @classmethod
    def _ensure_data_dir(cls, v: Path) -> Path:
        v = Path(v).expanduser().resolve()
        v.mkdir(parents=True, exist_ok=True)
        (v / "db").mkdir(exist_ok=True)
        (v / "cache").mkdir(exist_ok=True)
        (v / "models").mkdir(exist_ok=True)
        (v / "logs").mkdir(exist_ok=True)
        (v / "fnspid").mkdir(exist_ok=True)
        return v

    # --- Convenience paths ---
    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "bot8.duckdb"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        """Where to write rotating log files.

        Precedence:
          1. `LOGS_DIR` env var (for separate Railway log volume)
          2. `{data_dir}/logs/` default
        """
        if self.logs_dir_override is not None:
            p = Path(self.logs_dir_override).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p
        return self.data_dir / "logs"

    @property
    def fnspid_dir(self) -> Path:
        return self.data_dir / "fnspid"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Import this, don't instantiate Settings directly."""
    return Settings()
