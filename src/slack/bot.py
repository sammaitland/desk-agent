"""Slack interface.

Socket Mode, so no public URL or inbound webhook is needed — the bot opens an
outbound WebSocket to Slack. That makes it runnable from a laptop, which is the
whole point for a desk tool that queries a local blotter.

Mention the bot in a channel and it investigates, replying in-thread with any
charts attached:

    @deskbot why did the C order fill badly on the 24th?
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from src import env
from src.agent.loop import run_agent
from src.db import get_engine
from src.slack.format import to_mrkdwn, truncate

log = logging.getLogger(__name__)

_MENTION = re.compile(r"<@[A-Z0-9]+>")


def clean_question(text: str) -> str:
    """Strip the bot mention Slack prepends to the message."""
    return _MENTION.sub("", text).strip()


def build_app(engine=None) -> App:
    engine = engine or get_engine()
    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    @app.event("app_mention")
    def handle_mention(event, client, say, logger):
        question = clean_question(event.get("text", ""))
        # Reply in a thread: keeps the channel readable and keeps each
        # investigation with the question that prompted it.
        thread_ts = event.get("thread_ts") or event["ts"]
        channel = event["channel"]

        if not question:
            say(text="Ask me something about the blotter.", thread_ts=thread_ts)
            return

        # A run takes 10-30 seconds. Post a placeholder immediately and edit it
        # in place, so the channel does not sit silent while the agent works.
        placeholder = client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text="_Looking into it…_"
        )

        try:
            with engine.connect() as conn:
                trace = run_agent(question, conn, save_trace=True)
        except Exception:
            logger.exception("agent run failed")
            client.chat_update(
                channel=channel, ts=placeholder["ts"],
                text="Something went wrong running that query. Check the logs.",
            )
            return

        answer = truncate(to_mrkdwn(trace.answer))
        footer = _footer(trace)

        client.chat_update(
            channel=channel, ts=placeholder["ts"],
            text=answer,
            blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": answer}},
                {"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]},
            ],
        )

        for path in _charts(trace):
            try:
                client.files_upload_v2(
                    channel=channel, thread_ts=thread_ts,
                    file=str(path), title=path.stem,
                )
            except Exception:
                logger.exception("chart upload failed for %s", path)

    @app.event("message")
    def ignore_other_messages(logger):
        """Bolt warns about unhandled message events; this silences the noise."""

    return app


def _footer(trace) -> str:
    """A one-line provenance strip under every answer.

    Showing which tools ran, on a surface where the reasoning is otherwise
    invisible, is what makes the answer auditable rather than oracular.
    """
    tools = ", ".join(trace.tool_sequence) or "no tools"
    seconds = trace.duration_ms / 1000
    return (f"`{tools}` · {trace.turns} turns · {seconds:.1f}s · "
            f"{trace.input_tokens:,} in / {trace.output_tokens:,} out · "
            f"run `{trace.run_id}`")


def _charts(trace) -> list[Path]:
    paths = []
    for call in trace.tool_calls:
        if call.name != "make_chart" or call.error:
            continue
        raw = call.raw_result or {}
        path = Path(raw["chart_path"]) if isinstance(raw, dict) and "chart_path" in raw else None
        if path and path.exists():
            paths.append(path)
    return paths


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env.require("anthropic", "slack")

    app = build_app()
    log.info("Desk agent listening on Slack (Socket Mode). Mention the bot to ask a question.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
