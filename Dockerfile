# Reproducible environment for the desk agent.
#
# The motivation is the same one behind calibration/live parity in the trading
# system: results are only meaningful if the environment that produced them is
# the environment that runs them. A container makes that a property of the
# artefact rather than a hope about the host.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, in their own layer: source changes then rebuild in
# seconds rather than re-resolving the dependency tree every time.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install "anthropic>=1.0" "sqlalchemy>=2.0" "matplotlib>=3.8" \
                "python-dotenv>=1.0" "slack-bolt>=1.18" "mcp>=2.0" \
                "psycopg2-binary>=2.9" \
                "streamlit>=1.30" "pandas>=2.0"

COPY src/ ./src/
COPY schema.sql cli.py run_evals.py ./

# Run as a non-root user: the container holds API credentials and reaches the
# network, so it should not also hold root in its own filesystem.
RUN useradd --create-home --uid 1000 desk && chown -R desk:desk /app
USER desk

# Generate a blotter on first run if none is mounted, so the image is usable
# out of the box rather than requiring a setup step nobody documents.
ENV DB_URL=sqlite:////app/data/blotter.db
VOLUME ["/app/data"]

ENTRYPOINT ["python"]
CMD ["cli.py", "--help"]
