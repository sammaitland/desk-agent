"""Tests for the MCP server.

The server is intentionally thin, so these check that the wrapper preserves
what matters rather than re-testing the tools underneath: every tool is
exposed, the schemas the SDK derives from the type hints are correct, the
provenance envelope survives the transport, and nothing that writes is
reachable.

Built against mcp 2.x. Field names are snake_case (`input_schema`, not
`inputSchema`) — v1 code fails here.
"""

from __future__ import annotations

import json

import pytest

from src.db import create_schema, get_engine
from src.generate_blotter import Generator
from src.tools import TOOLS


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Point the server at a purpose-built blotter rather than the dev one."""
    from src.mcp_server import server as module

    db = tmp_path_factory.mktemp("mcp") / "blotter.db"
    engine = get_engine(f"sqlite:///{db}")
    create_schema(engine)
    gen = Generator(seed=42, days=90)
    gen.run()
    gen.write(engine)

    module._engine = engine
    yield module.mcp
    module._engine = None


async def call(server, name, arguments):
    """Invoke a tool and return the decoded envelope."""
    result = await server.call_tool(name, arguments)
    assert not result.is_error, f"{name} returned an error"
    return json.loads(result.content[0].text)


# --- registration ---------------------------------------------------------

@pytest.mark.asyncio
async def test_every_analytical_tool_is_exposed(server):
    """make_chart is deliberately absent: it writes PNGs to local disk, which
    is meaningless to a remote MCP host."""
    names = {t.name for t in await server.list_tools()}
    assert names == set(TOOLS) - {"make_chart"}
    assert "search_documentation" in names


@pytest.mark.asyncio
async def test_descriptions_carry_over(server):
    """The docstring is what an MCP host shows its model when choosing a tool."""
    for tool in await server.list_tools():
        assert tool.description and len(tool.description) > 100


@pytest.mark.asyncio
async def test_schemas_are_derived_from_type_hints(server):
    """No hand-written JSON Schema: the SDK generates it from the signature."""
    tools = {t.name: t for t in await server.list_tools()}

    position = tools["explain_position"].input_schema
    assert position["required"] == ["tag"]
    assert position["properties"]["tag"]["type"] == "string"

    quality = tools["execution_quality"].input_schema
    assert set(quality["properties"]) == {
        "start_date", "end_date", "ticker", "order_id", "group_by"}
    assert not quality.get("required")


# --- envelope preservation ------------------------------------------------

@pytest.mark.asyncio
async def test_envelope_survives_the_transport(server):
    """Provenance must reach the host, or it cannot report scope honestly."""
    payload = await call(server, "alpha_attribution", {"group_by": "idx"})
    assert set(payload) == {"summary", "provenance", "data"}
    assert payload["provenance"]["group_by"] == "idx"
    assert "alpha" in payload["provenance"]["measure"].lower()


@pytest.mark.asyncio
async def test_optional_arguments_may_be_omitted(server):
    payload = await call(server, "detect_anomalies", {"limit": 5})
    assert payload["provenance"]["window"]


@pytest.mark.asyncio
async def test_none_arguments_are_dropped_not_forwarded(server):
    """Explicit nulls must not override the tools' own defaults."""
    payload = await call(server, "query_blotter",
                         {"entity": "positions", "ticker": None, "limit": 3})
    assert len(payload["data"]) <= 3


@pytest.mark.asyncio
async def test_chained_lookup_works_over_mcp(server):
    """The same two-step investigation the agent runs in-process."""
    orders = await call(server, "query_blotter", {"entity": "orders", "limit": 1})
    order_id = orders["data"][0]["order_id"]
    detail = await call(server, "execution_quality", {"order_id": order_id})
    assert detail["data"]["order_id"] == order_id


@pytest.mark.asyncio
async def test_missing_record_returns_a_result_not_an_error(server):
    payload = await call(server, "explain_position", {"tag": "NOT_A_TAG"})
    assert payload["data"] == [] and "No position found" in payload["summary"]


# --- resources ------------------------------------------------------------

@pytest.mark.asyncio
async def test_coverage_resource_reports_the_window(server):
    from src.mcp_server.server import coverage

    text = coverage()
    assert "Blotter covers" in text and "Read-only" in text


# --- read-only ------------------------------------------------------------

def test_no_write_tools_are_exposed():
    """The read-only guarantee must hold across every transport, not just one."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "src" / "mcp_server" / "server.py").read_text().upper()
    for keyword in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER "):
        assert keyword not in source
