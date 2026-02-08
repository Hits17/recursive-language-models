# Contributing to Recursive Language Models

Thank you for your interest in contributing to RLM! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Ollama installed and running
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Hits17/recursive-language-models.git
cd recursive-language-models

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (including dev deps)
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Copy environment file
cp .env.example .env
```

## 📝 Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

```bash
# Format code
black rlm/

# Lint
flake8 rlm/

# Type check
mypy rlm/
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rlm

# Run specific test file
pytest tests/test_rlm_core.py
```

## 📁 Project Structure

```
rlm/
├── __init__.py        # Package exports
├── rlm_core.py        # Core RLM implementation
├── repl.py            # REPL environment
├── ollama_client.py   # Ollama API client
├── async_rlm.py       # Async/parallel features
├── streaming.py       # Streaming output
├── budget.py          # Budget controls
└── rag.py             # RAG integration
```

## 🔀 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### PR Guidelines

- Write clear, descriptive commit messages
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Keep PRs focused on a single feature/fix

## 🐛 Reporting Bugs

Please use the GitHub issue tracker with the following information:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, Ollama version, OS

## 💡 Feature Requests

We welcome feature requests! Please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How might it work?
3. **Alternatives**: Any alternatives you've considered?

## 🏛️ Architecture Decisions

### Core Principles

1. **Drop-in Replacement**: `rlm.completion()` should mirror standard LLM APIs
2. **Modularity**: Enhancements should be optional and composable
3. **Local First**: Optimize for local Ollama, cloud APIs secondary
4. **Safety**: REPL execution should be sandboxed by default

### Adding New Features

When adding enhancements:

1. Create a new module in `rlm/` directory
2. Export in `__init__.py`
3. Add usage example to `demo_enhancements.py`
4. Document in README

## 📜 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers feel welcome
- Assume good intentions

## 📧 Contact

- Open an issue for questions
- Tag maintainers for urgent matters

---

Thank you for contributing! 🎉
