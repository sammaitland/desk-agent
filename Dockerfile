# Reproducible environment for the desk agent.
#
# The motivation is the same one behind calibration/live parity in the trading
# system: results only mean something if the environment that produced them is
# the environment that runs them. A container makes that a property of the
# artefact rather than a hope about the host.
#
# Dependencies come from pyproject.toml, not a hand-copied list. An earlier
# version listed packages inline and omitted scikit-learn — so documentation
# retrieval failed inside the container while working on every developer
# machine. External review caught it. The single source of truth is the
# project definition; the Dockerfile installs it.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Project definition first, in its own layer: source edits then rebuild in
# seconds rather than re-resolving the dependency tree every time.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install "psycopg2-binary>=2.9" && \
    pip install .[slack,dashboard,observability]

# Source, schema, entry points, and the documentation corpus the retrieval
# tool reads from. docs/ was missing before; without it search_documentation
# returns nothing.
COPY src/ ./src/
COPY docs/ ./docs/
COPY schema.sql cli.py run_evals.py ./

# Reinstall in editable mode so `src` resolves from the copied tree.
RUN pip install --no-deps -e .

RUN useradd --create-home --uid 1000 desk && chown -R desk:desk /app
USER desk

ENV DB_URL=sqlite:////app/data/blotter.db
VOLUME ["/app/data"]

ENTRYPOINT ["python"]
CMD ["cli.py", "--help"]


# ---------------------------------------------------------------------------
# Test stage: the base image plus pytest and the tests. `make docker-test`
# targets this stage; the runtime image stays free of test tooling.
# ---------------------------------------------------------------------------
FROM base AS test
USER root
RUN pip install .[dev]
COPY tests/ ./tests/
RUN chown -R desk:desk /app/tests
USER desk
CMD ["-m", "pytest", "-q"]
