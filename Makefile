.PHONY: help install test blotter ask evals critic slack mcp dashboard adapter docker-test clean

help:
	@echo "install     install in editable mode with dev extras"
	@echo "test        run the test suite"
	@echo "blotter     regenerate the synthetic blotter"
	@echo "ask Q=...   ask the agent one question"
	@echo "evals       run the eval suite (costs API calls)"
	@echo "slack       start the Slack bot"
	@echo "mcp         start the MCP server (stdio)"
	@echo "dashboard   start the Streamlit dashboard"
	@echo "adapter     load archived V9.2C output (ARCHIVE=path)"
	@echo "fixtures    rebuild adapter test fixtures (DAY=archive/YYYY-MM-DD)"
	@echo "docker-test run the suite against Postgres in Docker"

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

blotter:
	python -m src.generate_blotter --days 120 --seed 42

ask:
	@python cli.py "$(Q)" --trace

evals:
	python run_evals.py

critic:
	python run_critic_benchmark.py

slack:
	python -m src.slack.bot

mcp:
	python -m src.mcp_server.server

dashboard:
	streamlit run src/dashboard/app.py

adapter:
	python run_adapter.py $(ARCHIVE)

fixtures:
	python make_fixtures.py $(DAY)

docker-test:
	docker compose run --rm test

clean:
	rm -rf .pytest_cache **/__pycache__ charts traces eval_results
