"""Tests for the Slack layer.

Covers everything that does not need a live Slack connection: the mrkdwn
conversion, message truncation, mention stripping, and chart extraction. The
formatting is where the bugs actually live — Slack's dialect differs from
Markdown in ways that fail silently, rendering literal asterisks and unreadable
pipe soup rather than raising anything.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.trace import Trace
from src.slack.format import to_mrkdwn, truncate


def make_trace(answer: str = "answer", tools=None) -> Trace:
    trace = Trace(question="q")
    for name, data in tools or []:
        result = type("R", (), {"provenance": {"rows": 1}, "summary": "s", "data": data})()
        trace.record_tool(name, {}, result, 1, 1)
    trace.finish(answer, "end_turn")
    return trace


# --- mrkdwn conversion ----------------------------------------------------

def test_bold_is_single_asterisk():
    """Slack renders ** literally; it wants *bold*."""
    assert to_mrkdwn("This is **important**.") == "This is *important*."


def test_headers_become_bold_lines():
    assert to_mrkdwn("### Execution detail") == "*Execution detail*"


def test_links_are_inverted():
    assert to_mrkdwn("[the docs](https://x.com)") == "<https://x.com|the docs>"


def test_horizontal_rules_removed():
    assert "---" not in to_mrkdwn("one\n\n---\n\ntwo")


def test_table_becomes_aligned_code_block():
    """Slack has no tables at all; monospace is the only way to align columns."""
    markdown = (
        "| Sector | Trades | Alpha |\n"
        "|--------|--------|-------|\n"
        "| VIS    | 56     | +34.57% |\n"
        "| VOX    | 49     | -8.58% |"
    )
    out = to_mrkdwn(markdown)
    assert out.startswith("```") and out.endswith("```")
    assert "|" not in out
    lines = out.strip("`").strip().split("\n")
    # Header, divider, two data rows — and columns aligned to equal width.
    assert len(lines) == 4
    assert lines[2].index("56") == lines[3].index("49")


def test_table_strips_emphasis_inside_code_block():
    """mrkdwn is not interpreted inside code blocks, so ** would show literally."""
    markdown = "| A | B |\n|---|---|\n| **x** | _y_ |"
    out = to_mrkdwn(markdown)
    assert "**" not in out and "_y_" not in out


def test_identifiers_keep_their_underscores():
    """Caught in the first live Slack run: VGT_AAPL_NVDA_L became VGTAAPLNVDAL.

    Stripping emphasis markers inside a code block was over-eager — underscores
    in position tags are data, not formatting.
    """
    markdown = ("| Date | Position |\n|---|---|\n"
                "| Aug 17 | **VGT_AAPL_NVDA_L_20260810_329** |")
    out = to_mrkdwn(markdown)
    assert "VGT_AAPL_NVDA_L_20260810_329" in out
    assert "**" not in out


def test_emphasis_inside_a_word_is_left_alone():
    from src.slack.format import _strip_emphasis

    assert _strip_emphasis("snake_case_name") == "snake_case_name"
    assert _strip_emphasis("*bold*") == "bold"


def test_prose_around_a_table_is_preserved():
    out = to_mrkdwn("Before.\n\n| A |\n|---|\n| 1 |\n\nAfter.")
    assert out.startswith("Before.") and out.endswith("After.")


def test_conversion_is_idempotent_on_plain_text():
    plain = "Nothing special here."
    assert to_mrkdwn(plain) == plain


# --- truncation -----------------------------------------------------------

def test_short_text_untouched():
    assert truncate("short") == "short"


def test_long_text_cut_and_flagged():
    """Slack rejects blocks over 3000 characters."""
    out = truncate("word " * 2000, limit=500)
    assert len(out) < 700 and "truncated" in out


def test_truncation_prefers_a_paragraph_boundary():
    text = "A" * 400 + "\n\n" + "B" * 400
    out = truncate(text, limit=500)
    assert out.startswith("A" * 400)
    assert "B" not in out.split("_(")[0]


# --- bot helpers ----------------------------------------------------------

def test_mention_is_stripped():
    from src.slack.bot import clean_question

    assert clean_question("<@U012AB3CD> why did the C order fill badly?") == \
        "why did the C order fill badly?"


def test_empty_after_mention_is_empty():
    from src.slack.bot import clean_question

    assert clean_question("<@U012AB3CD>") == ""


def test_footer_shows_provenance():
    """On a surface with no visible reasoning, the tool list is the audit trail."""
    from src.slack.bot import _footer

    trace = make_trace("a", [("detect_anomalies", {}), ("execution_quality", {})])
    footer = _footer(trace)
    assert "detect_anomalies, execution_quality" in footer
    assert trace.run_id in footer


def test_charts_extracted_only_when_present(tmp_path):
    from src.slack.bot import _charts

    real = tmp_path / "chart.png"
    real.write_bytes(b"x")
    trace = make_trace("a", [
        ("make_chart", {"chart_path": str(real)}),
        ("make_chart", {"chart_path": "/nonexistent/chart.png"}),
        ("alpha_attribution", {"breakdown": []}),
    ])
    assert _charts(trace) == [Path(real)]


def test_no_charts_when_none_rendered():
    from src.slack.bot import _charts

    assert _charts(make_trace("a", [("query_blotter", [])])) == []
