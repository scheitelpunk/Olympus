# GASM-Roboting Makefile
# Convenient commands for development and deployment

.PHONY: help clean install install-dev test lint format build upload docs

# Default target
help:
	@echo "GASM-Roboting Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  install       Install package in production mode"
	@echo "  install-dev   Install package in development mode with all extras"
	@echo "  install-min   Install minimal dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  test          Run test suite"
	@echo "  lint          Run linting (flake8, mypy)"
	@echo "  format        Format code (black, isort)"
	@echo "  clean         Clean build artifacts"
	@echo ""
	@echo "Building:"
	@echo "  build         Build distribution packages"
	@echo "  docs          Build documentation"
	@echo "  check         Check package before upload"
	@echo ""
	@echo "Publishing:"
	@echo "  upload-test   Upload to Test PyPI"
	@echo "  upload        Upload to PyPI"
	@echo ""
	@echo "Utilities:"
	@echo "  setup-env     Run environment setup script"
	@echo "  agents-2d     Run 2D spatial agent demo"
	@echo "  agents-3d     Run 3D physics agent demo"

# Installation targets
install:
	pip install .

install-dev:
	pip install -e ".[dev,all]"
	pre-commit install

install-min:
	pip install -e .

# Development targets
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -x -v

lint:
	flake8 src/ tests/ scripts/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building targets
build: clean
	python -m build

docs:
	cd docs && make html

check: build
	twine check dist/*

# Publishing targets
upload-test: build check
	twine upload --repository testpypi dist/*

upload: build check
	twine upload dist/*

# Utility targets
setup-env:
	python scripts/setup_environment.py --mode development --verbose

agents-2d:
	python -m spatial_agent.agent_loop_2d --demo

agents-3d:
	python -m spatial_agent.agent_loop_pybullet --demo

server:
	python -m api.main

# Development server with auto-reload
dev-server:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Docker targets (optional)
docker-build:
	docker build -t gasm-roboting .

docker-run:
	docker run -p 8000:8000 gasm-roboting

# CI/CD simulation
ci: clean install-dev lint test build check
	@echo "✅ All CI checks passed!"

# Quick development setup
dev-setup: clean install-dev
	@echo "🚀 Development environment ready!"
	@echo "Try: make agents-2d"

# Package info
info:
	python -c "import sys; sys.path.insert(0, 'src'); from gasm import quick_start, check_dependencies; quick_start(); print('\nDependencies:'); [print(f'  {k}: {v}') for k, v in check_dependencies().items()]"

# Environment check
check-env:
	python scripts/setup_environment.py --check-only --verbose