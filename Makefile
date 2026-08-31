.PHONY: install lint typecheck test test-unit test-integration test-e2e coverage mutate clean all

install:
	uv sync --all-extras

lint:
	uv run ruff check echovector/ tests/ benchmarks/
	uv run ruff format --check echovector/ tests/ benchmarks/

lint-fix:
	uv run ruff check --fix echovector/ tests/ benchmarks/
	uv run ruff format echovector/ tests/ benchmarks/

typecheck:
	uv run mypy echovector/ benchmarks/

benchmark:
	uv run python benchmarks/esc50.py --dataset $(ESC50) --backend clap --json-out results/esc50-clap.json

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v -m integration

test-e2e:
	uv run pytest tests/e2e/ -v

coverage:
	uv run pytest tests/ --cov=echovector --cov-report=html --cov-report=term-missing

mutate:
	uv run mutmut run

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage .mutmut-cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint typecheck test
