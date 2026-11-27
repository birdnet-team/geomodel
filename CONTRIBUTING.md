# Contributing to BirdNET Geomodel

Contributions are welcome. This page covers how to report bugs, suggest features and open pull requests.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://github.com/birdnet-team/geomodel/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <ccb-birdnet@cornell.edu>.

## How to Contribute

### Reporting Bugs

- Use a clear and descriptive title.
- Describe the exact steps to reproduce the problem.
- Include the Python version, OS, and GPU/CUDA version if relevant.
- Attach error tracebacks and log output.

### Suggesting Features

- Open an issue describing the feature and its motivation.
- Explain how it fits the project's goals.
- If possible, outline a proposed implementation.

### Pull Requests

1. Fork the repo and create a branch from `main`.
2. If you've added code, add or update docstrings.
3. Make sure the code runs without errors and `python -m pytest tests` passes (under a minute on a CPU).
4. Keep commits small and focused; write clear commit messages.
5. Open a pull request with a description of what changed and why.

### AI-Assisted Contributions

Contributions made with AI coding assistants (Copilot, Cursor, etc.) are welcome. Keep each PR to a single, isolated change: one bug fix, one feature or one refactor. Large PRs that touch many unrelated parts of the codebase are hard to review and will be asked to be split up. In particular:

- Do not bundle AI-generated reformatting, renaming or stylistic changes with functional work.
- Check that AI-generated code is correct and tested, and that you understand what it does.
- Mention in the PR description if AI tools were used.

## Development Setup

```bash
git clone https://github.com/birdnet-team/geomodel.git
cd geomodel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
```

The documentation builds with `pip install -r requirements-docs.txt && mkdocs serve`.

## Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use type hints for function signatures.
- Write docstrings for all public classes, methods, and functions.
- Keep lines to 120 characters or fewer where practical.
- Use `snake_case` for functions and variables, `PascalCase` for classes.

## Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Fix bug" not "Fixes bug").
- Keep the first line under 72 characters.
- Reference issues with `#123` where applicable.
