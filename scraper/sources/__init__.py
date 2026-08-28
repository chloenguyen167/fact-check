"""Registry mapping scraper_key -> scraper class."""

from __future__ import annotations

import json
from typing import Type

from scraper.base import BaseScraper
from scraper.sources.skds import SKDSScraper

SCRAPER_REGISTRY: dict[str, Type[BaseScraper]] = {
    "skds": SKDSScraper,
}


def build_scraper(source_row: dict) -> BaseScraper:
    key = source_row["scraper_key"]
    cls = SCRAPER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Chưa hỗ trợ scraper_key={key!r}. Có sẵn: {sorted(SCRAPER_REGISTRY)}"
        )
    config = source_row.get("config_json") or "{}"
    if isinstance(config, str):
        config = json.loads(config or "{}")
    return cls(
        base_url=source_row["base_url"],
        list_url=source_row["list_url"],
        config=config,
    )


def available_scraper_keys() -> list[str]:
    return sorted(SCRAPER_REGISTRY)
