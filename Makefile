# ARIA — Project Makefile
# Run all targets from the aria/ root directory (same directory as this file).
# On Windows: winget install GnuWin32.Make  |  scoop install make  |  choco install make

PYTHON  := python
PIP     := pip
PYTEST  := pytest

.DEFAULT_GOAL := help

.PHONY: help install init-db dev ui test test-all test-integration lint clean verify-imports

help:
	@echo ""
	@echo "  ARIA — available make targets"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install          Install all dependencies + spaCy model"
	@echo "  make init-db          Initialise SQLite database with all tables"
	@echo "  make dev              Start FastAPI dev server (hot-reload, port 8000)"
	@echo "  make ui               Launch Streamlit chat interface (port 8501)"
	@echo "  make test             Run unit tests"
	@echo "  make test-all         Run full test suite (unit + integration)"
	@echo "  make test-integration Run integration tests only"
	@echo "  make lint             Run ruff + mypy"
	@echo "  make clean            Remove all __pycache__ directories"
	@echo "  make verify-imports   Verify all canonical imports resolve"
	@echo ""

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

init-db:
	$(PYTHON) -c "import asyncio; from db.database import init_db; asyncio.run(init_db())"

dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py

test:
	$(PYTEST) tests/unit/ -v --timeout=30

test-all:
	$(PYTEST) tests/ -v --timeout=60 -m "not integration"

test-integration:
	$(PYTEST) tests/integration/ -v --timeout=180 -m integration

lint:
	ruff check . && mypy . --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

verify-imports:
	@echo "Verifying all package imports..."
	$(PYTHON) -c "from shared import PlanStep, EpisodicMemory, IntentType; print('v shared')"
	$(PYTHON) -c "from shared.constants import TaskType; print('v shared.constants.TaskType')"
	$(PYTHON) -c "from config import get_config; print('v config')"
	$(PYTHON) -c "from db import init_db; print('v db')"
	$(PYTHON) -c "from db.repositories import SessionRepository, MemoryRepository; print('v db.repositories')"
	$(PYTHON) -c "from core.state import ARIAState; print('v core.state')"
	@echo "All imports verified."
