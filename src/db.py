"""Database connection and schema management.

Uses a DB_URL environment variable so the same code runs against SQLite
(zero-setup development) and PostgreSQL (target deployment) without change.
The schema uses no dialect-specific features, so the swap is a config change.

    export DB_URL=sqlite:///blotter.db                      # default
    export DB_URL=postgresql+psycopg2://user:pw@host/blotter
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_URL = f"sqlite:///{PROJECT_ROOT / 'blotter.db'}"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"


def get_engine(db_url: str | None = None) -> Engine:
    """Return a SQLAlchemy engine for the configured database."""
    url = db_url or os.getenv("DB_URL", DEFAULT_DB_URL)
    engine = create_engine(url, future=True)
    if url.startswith("sqlite"):
        # Foreign keys are off by default in SQLite; the schema relies on them.
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
    return engine


def create_schema(engine: Engine) -> None:
    """Execute schema.sql against the target database.

    Destructive: the DDL drops existing tables first. Intended for rebuilding
    a development blotter, not for migrating a populated one.
    """
    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    # Strip line comments before splitting: a ';' inside a comment would
    # otherwise truncate the statement it sits in.
    stripped = "\n".join(line.split("--")[0] for line in raw.splitlines())
    statements = [s.strip() for s in stripped.split(";") if s.strip()]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def table_counts(engine: Engine) -> dict[str, int]:
    """Row count per table — used to verify a generation run."""
    tables = [
        "instruments", "workflow_runs", "workflow_stages", "portfolio_snapshots",
        "pair_evaluations", "positions", "position_updates", "orders",
        "order_allocations", "fills", "stop_orders", "risk_checks", "system_events",
    ]
    counts = {}
    with engine.connect() as conn:
        for table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            counts[table] = result.scalar_one()
    return counts
