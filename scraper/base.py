"""Abstract scraper interface and shared HTTP helpers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Iterator, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


@dataclass
class ArticleListing:
    title: str
    link: str
    publish_time: Optional[datetime] = None
    external_id: Optional[str] = None


@dataclass
class ArticleDetail:
    title: str
    link: str
    publish_time: Optional[datetime]
    content: str
    author: Optional[str] = None
    tags: Optional[str] = None
    related_one_news: Optional[str] = None
    external_id: Optional[str] = None
    source: str = ""


@dataclass
class ScrapeProgress:
    found: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    messages: list[str] = field(default_factory=list)


def day_range(target: date) -> tuple[datetime, datetime]:
    """Full calendar day [00:00:00, 23:59:59]."""
    return (
        datetime.combine(target, dtime.min),
        datetime.combine(target, dtime(23, 59, 59)),
    )


class BaseScraper(ABC):
    """Mỗi trang báo implement một subclass."""

    key: str = "base"
    source_code: str = "unknown"

    def __init__(
        self,
        *,
        base_url: str,
        list_url: str,
        config: Optional[dict] = None,
        request_delay: float = 0.6,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.list_url = list_url
        self.config = config or {}
        self.request_delay = request_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def absolute_url(self, href: str) -> str:
        return urljoin(self.base_url + "/", href)

    def fetch_html(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        if self.request_delay:
            time.sleep(self.request_delay)
        return resp.text

    def fetch_soup(self, url: str) -> BeautifulSoup:
        return BeautifulSoup(self.fetch_html(url), "html.parser")

    @abstractmethod
    def iter_listings_for_range(
        self, from_time: datetime, to_time: datetime
    ) -> Iterator[ArticleListing]:
        """Yield listings with publish_time in ``[from_time, to_time]``."""

    def iter_listings_for_date(self, target: date) -> Iterator[ArticleListing]:
        start, end = day_range(target)
        yield from self.iter_listings_for_range(start, end)

    @abstractmethod
    def fetch_article(self, listing: ArticleListing) -> ArticleDetail:
        """Fetch and parse full article page."""

    def scrape_range(
        self, from_time: datetime, to_time: datetime
    ) -> Iterator[tuple[str, ArticleDetail | ArticleListing, Optional[str]]]:
        """
        Yield tuples:
          ("ok", ArticleDetail, None)
          ("error", ArticleListing, error_message)
        """
        for listing in self.iter_listings_for_range(from_time, to_time):
            try:
                detail = self.fetch_article(listing)
                detail.source = self.source_code
                yield "ok", detail, None
            except Exception as exc:  # noqa: BLE001 - collect per-article failures
                yield "error", listing, str(exc)

    def scrape_date(
        self, target: date
    ) -> Iterator[tuple[str, ArticleDetail | ArticleListing, Optional[str]]]:
        start, end = day_range(target)
        yield from self.scrape_range(start, end)
