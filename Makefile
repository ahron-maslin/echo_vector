.PHONY: install lint typecheck test test-unit test-integration test-e2e test-perf coverage mutate docs serve-docs clean

install:
	uv sync --all-extras

lint:
	uv run ruff check echovector/ tests/
	uv run ruff format --check echovector/ tests/

lint-fix:
	uv run ruff check --fix echovector/ tests/
	uv run ruff format echovector/ tests/

typecheck:
	uv run mypy echovector/

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v -m integration

test-e2e:
	uv run pytest tests/e2e/ -v -m e2e

test-perf:
	uv run pytest tests/performance/ -v -m performance

coverage:
	uv run pytest tests/ --cov=echovector --cov-report=html --cov-report=term-missing

mutate:
	uv run mutmut run

docs:
	cd docs && uv run mkdocs build

serve-docs:
	cd docs && uv run mkdocs serve

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage site/ .mutmut-cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint typecheck test
