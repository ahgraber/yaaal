from functools import wraps
import inspect
import json
import logging
import os

from pydantic import BaseModel, Field, create_model

import requests

from ..core.tool import tool

logger = logging.getLogger(__name__)


class URLContent(BaseModel, extra="ignore"):
    """Text content from a webpage."""

    url: str = Field(description="The webpage url")
    title: str = Field(description="The page title")
    content: str = Field(description="The webpage's text content")


@tool
def get_url_content(url: str, timeout: int = 30) -> URLContent:
    """Fetch content from a webpage using the Jina Reader service.

    Parameters
    ----------
        url (str): The URL of the webpage to fetch.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.

    Returns
    -------
        URLContent: A model containing the URL, page title, and text content.
    """
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


# TODO: jina deepsearch or exa search


class GitHubContent(BaseModel, extra="ignore"):
    """Text content from a GitHub repo."""

    url: str = Field(description="The github url")
    tree: str = Field(description="A tree-like structure of the files")
    content: str = Field(description="The repo's content as a single markdown-formatted string")


@tool
def get_github_content(
    url: str,
    max_file_size: int = 50 * 1024 * 1024,  # 50 MB
    include_patterns: str | set[str] | None = None,
    exclude_patterns: str | set[str] | None = None,
    branch: str | None = None,
) -> GitHubContent:
    """Clone and parse a GitHub repository to aggregate its file contents into a single markdown-formatted string.

    Parameters
    ----------
        url (str): A valid GitHub repository URL.
        max_file_size (int, optional): Maximum allowed file size for ingestion. Files larger than this size are ignored (default 50 MB).
        include_patterns (Union[str, Set[str]], optional): File patterns to include. If None, all files are included.
        exclude_patterns (Union[str, Set[str]], optional): File patterns to exclude. If None, no files are excluded.
        branch (str, optional): Specific branch to clone. If None, the repository's default branch is used.

    Returns
    -------
        GitHubContent: A model containing the repository URL, a tree-like file structure, and the aggregated markdown content.

    Raises
    ------
        ValueError: If the URL is not a github.com URL.
        TypeError: If the clone operation does not return the expected coroutine.
    """
    from gitingest import ingest

    if "github.com" not in url:
        raise ValueError("The url must be a github.com url")

    _summary, tree, content = ingest(
        url,
        max_file_size=max_file_size,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        branch=branch,
    )

    return GitHubContent(url=url, tree=tree, content=content)
