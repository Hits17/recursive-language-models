.PHONY: install test lint format clean run help

# Default target
help:
	@echo "RLM - Recursive Language Models"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run        - Run the main demo"
	@echo "  make demo       - Run enhancement demos"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Clean temporary files"
	@echo "  make quickstart - Quick test with Ollama"

# Install dependencies
install:
	pip install -r requirements.txt

# Run main demo
run:
	python main.py

# Run enhancement demos
demo:
	python demo_enhancements.py

# Run tests
test:
	pytest tests/ -v

# Lint code
lint:
	flake8 rlm/ --max-line-length=100 --ignore=E501,W503

# Format code
format:
	black rlm/ --line-length=100

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.jsonl 2>/dev/null || true
	@echo "Cleaned!"

# Quick test
quickstart:
	@echo "Testing RLM with Ollama..."
	python -c "from rlm import RLM, OllamaClient; \
		client = OllamaClient(); \
		print(f'Connected to: {client.model_name}'); \
		print('RLM package working!')"
