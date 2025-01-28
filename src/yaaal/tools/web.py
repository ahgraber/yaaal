from functools import wraps
import inspect
import json
import logging
import os
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field, create_model

import requests

from ..core.tools import tool

logger = logging.getLogger(__name__)


class URLContent(BaseModel, extra="ignore"):
    """Text content from a webpage."""

    url: str = Field(description="The webpage url")
    title: str = Field(description="The page title")
    content: str = Field(description="The webpage's text content")


@tool
def get_page_content(url: str, timeout: int = 30) -> URLContent:
    """Use Jina Reader to extract content from url as markdown."""
    jina_pfx = "https://r.jina.ai"

    headers = {"Accept": "application/json"}
    if "YAAAL_JINA_READER_KEY" in os.environ:
        headers["Authorization"] = f"Bearer {os.environ['YAAAL_JINA_READER_KEY']}"

    try:
        logger.debug(f"Requesting page content from {url}...")
        with requests.get(f"{jina_pfx}/{url}", headers=headers, timeout=timeout) as response:
            _ = response.raise_for_status()
            content = json.loads(response.text)["data"]
        logger.debug("Retrieved content successfully!")
    except requests.exceptions.RequestException:
        logger.exception(f"Failed to fetch the original URL {url}")
        raise

    return URLContent(**content)
