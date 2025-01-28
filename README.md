# Yet Another AI Agent Library

[![Apache](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CommonsClause](https://img.shields.io/badge/License-CommonsClause-blue?color=blue)](https://commonsclause.com/)

<!-- markdownlint-disable -->
<div align="center">
  <img src="./assets/yaaal.webp" width="400"/>
</div>

<br>
<!-- markdownlint-enable -->

YAAAL (Yet Another AI Agent Library) is a highly composable and lightweight framework for building AI agents with minimal dependencies.
It aims to offer developers a simple yet flexible toolkit for creating autonomous systems that can interact with various environments, make decisions, and perform tasks in a modular manner ([more](docs/design_document.md)).

> [!WARNING]  
> `yaaal` is under active development and should not be considered for production use!

## Quickstart

Install, then check out how to use [Prompts](./examples/demo-prompt.ipynb) and [Callers](./examples/demo-caller.ipynb)

## Installation

> Requires Python 3.11+

Install with [`uv`](https://github.com/astral-sh/uv):

```sh
uv add "yaaal @ git+https://github.com/ahgraber/yaaal"
```

Install with pip:

```sh
pip install git+https://github.com/ahgraber/yaaal
```

## Configuration

`yaaal` depends on external services, which may define API keys.  We recommend saving these keys in a local `.env` file with a "YAAAL_" prefix so that they are not automatically detected/used unintentionally.

For instance, some libraries that use LangChain will automatically use / fall back to `OPENAI_API_KEY` if it exists in the environment, which may not be what you want.

_Sample `.env` file_:

```ini
# .env
YAAAL_OPENAI_API_KEY="..."
YAAAL_ANTHROPIC_API_KEY="..."
YAAAL_JINA_READER_KEY="..."
```
