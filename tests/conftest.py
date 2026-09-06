"""Shared fixtures.

The one that matters is `blotter_url`: it decides which database the
integration fixtures build their blotter in.

Every test module that generates a blotter used to hard-code
`sqlite:///{tmp_path}`. That meant the PostgreSQL CI job stood up a real
Postgres, generated a blotter in it, and then ran a test suite that quietly
built its own SQLite files and never touched Postgres. The job was green; the
claim it was supposed to verify was untested. External review caught it.

Now: if DB_URL is set (as it is in the Postgres CI job), integration fixtures
use it. Each module-scoped fixture calls create_schema, which drops and
recreates every table, so modules run in sequence share one database cleanly.
Without DB_URL, fixtures fall back to a per-module SQLite file as before.

Tests that construct their own in-memory SQLite for reasons unrelated to SQL
(the loop tests, the exporter tests) are left alone — they are not testing
queries.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def blotter_url(tmp_path_factory, request):
    """Database URL for a module's integration fixtures."""
    configured = os.getenv("DB_URL")
    if configured:
        return configured
    name = request.module.__name__.rsplit(".", 1)[-1]
    return f"sqlite:///{tmp_path_factory.mktemp(name) / 'blotter.db'}"


@pytest.fixture(scope="session", autouse=True)
def report_backend():
    """Say which backend the run used, so a CI log makes the claim verifiable."""
    url = os.getenv("DB_URL", "sqlite (per-module temp files)")
    backend = url.split("://")[0] if "://" in url else url
    print(f"\n[integration fixtures on: {backend}]")
    yield
