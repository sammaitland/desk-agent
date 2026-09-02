"""Environment loading.

Credentials live in a `.env` file at the project root, gitignored, and are
loaded on import by every entry point. This is not only convenience: when
Claude Desktop launches the MCP server it spawns a subprocess with a minimal
environment that does not inherit the shell's exports, so a server relying on
`export ANTHROPIC_API_KEY=...` would simply fail to authenticate.

Real environment variables always win over the file, so CI and containers can
inject secrets without a `.env` present.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

REQUIRED = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "slack": ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
}


def load() -> None:
    """Load .env if present. Existing environment variables take precedence."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional dependency; exports still work without it
        return
    load_dotenv(ENV_PATH, override=False)


def missing(*groups: str) -> list[str]:
    """Names of required variables that are still unset, for a clear error."""
    needed = [name for group in groups for name in REQUIRED.get(group, [])]
    return [name for name in needed if not os.getenv(name)]


def require(*groups: str) -> None:
    """Exit with a useful message rather than a traceback deep in an SDK."""
    absent = missing(*groups)
    if not absent:
        return
    raise SystemExit(
        f"Missing environment variable(s): {', '.join(absent)}.\n"
        f"Add them to {ENV_PATH} (copy .env.example) or export them."
    )


load()
