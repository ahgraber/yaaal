# Contributing

Thank you for contributing! This section describes the typical steps in setting up a development environment.

## Setup a virtual environment with [`uv`](https://github.com/astral-sh/uv)

```sh
uv sync
source venv/bin/activate
```

## Install [pre-commit](https://pre-commit.com/)

This will install the `pre-commit` package and then install the pre-commit hooks for standardized formatting.

```sh
uv tool install pre-commit
pre-commit install
```

## Running the tests

From within your virtual environment:

```sh
pytest
```

## Contributing

Committing the change will run all necessary formatting, type checking, and
linting.
Prefer small PRs to make reviews easy to manage.
