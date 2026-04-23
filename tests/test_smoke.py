"""Smoke tests — the pipeline imports and the schema loads."""

from __future__ import annotations


def test_imports() -> None:
    import bot8
    import bot8.config  # noqa: F401
    import bot8.data  # noqa: F401
    import bot8.data.news.fnspid_loader  # noqa: F401

    assert bot8.__version__


def test_settings_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)

    from bot8.config.settings import Settings

    s = Settings()
    assert s.data_dir == tmp_path.resolve()
    assert s.db_path.parent.exists()
    assert s.fnspid_dir.exists()
    assert s.target_gross_exposure == 1.0
    assert s.long_decile == 0.1


def test_db_init(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    # Force settings cache refresh
    from bot8.config import settings as s_mod
    s_mod.get_settings.cache_clear()

    from bot8.data.db import init_schema, session

    init_schema()
    with session(read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert {"news_raw", "news_scored", "bars_daily", "universe", "ingest_log"} <= tables
