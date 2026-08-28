"""Scraper cho Báo Sức khỏe & Đời sống (suckhoedoisong.vn)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator, Optional
from urllib.parse import urljoin

from scraper.base import ArticleDetail, ArticleListing, BaseScraper


def _parse_vn_datetime(raw: str | None) -> Optional[datetime]:
    if not raw:
        return None
    text = raw.strip()
    text = text.replace("|", " ").strip()
    # ISO
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass

    # Ưu tiên DD/MM/YYYY (tiếng Việt)
    candidates = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_mdy_datetime(raw: str | None) -> Optional[datetime]:
    """Parse MM/DD/YYYY (SKDS timeline title attribute)."""
    if not raw:
        return None
    text = raw.strip().replace("|", " ").strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_publish_time(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S.000000")


class SKDSScraper(BaseScraper):
    key = "skds"
    source_code = "suckhoedoisong"

    def resolve_zone_id(self) -> int:
        cfg_zone = self.config.get("zone_id")
        if cfg_zone:
            return int(cfg_zone)
        soup = self.fetch_soup(self.list_url)
        el = soup.find("input", {"id": "hdZoneId"}) or soup.find("input", {"name": "hdZoneId"})
        if not el or not el.get("value"):
            raise RuntimeError(f"Không tìm thấy hdZoneId trên {self.list_url}")
        return int(el["value"])

    def _timeline_url(self, zone_id: int, page: int) -> str:
        return f"{self.base_url}/timelinelist/{zone_id}/{page}.htm"

    def _parse_listing_item(self, article) -> Optional[ArticleListing]:
        link_el = article.select_one("a[data-type='title'], a.box-category-link-title, a.box-home-focus-link-title")
        if not link_el:
            link_el = article.select_one("a[href$='.htm']")
        if not link_el or not link_el.get("href"):
            return None
        href = link_el["href"]
        if "javascript" in href or not href.endswith(".htm"):
            return None
        title = (link_el.get("title") or link_el.get_text(" ", strip=True) or "").strip()
        if not title:
            return None

        time_el = article.select_one(
            "span.box-category-time, span.box-home-focus-time, span.time-ago, time"
        )
        publish = None
        if time_el is not None:
            # Text hiển thị thường là DD/MM/YYYY; attr title lại là MM/DD/YYYY.
            publish = _parse_vn_datetime(time_el.get_text(" ", strip=True))
            if publish is None:
                publish = _parse_vn_datetime(time_el.get("datetime"))
            if publish is None:
                # title trên timeline SKDS là MM/DD/YYYY
                publish = _parse_mdy_datetime(time_el.get("title"))
            if publish is None:
                publish = _parse_vn_datetime(time_el.get("title"))

        external_id = (
            article.get("data-id")
            or article.get("data-newsid")
            or link_el.get("data-id")
        )
        if not external_id:
            m = re.search(r"-(\d{10,})\.htm$", href)
            external_id = m.group(1) if m else None

        return ArticleListing(
            title=title,
            link=self.absolute_url(href),
            publish_time=publish,
            external_id=str(external_id) if external_id else None,
        )

    def iter_listings_for_range(
        self, from_time: datetime, to_time: datetime
    ) -> Iterator[ArticleListing]:
        zone_id = self.resolve_zone_id()
        seen: set[str] = set()
        max_pages = int(self.config.get("max_pages", 40))
        stop_after_older = int(self.config.get("stop_after_older_pages", 2))
        older_pages = 0

        for page in range(1, max_pages + 1):
            soup = self.fetch_soup(self._timeline_url(zone_id, page))
            items = soup.select("article.box-category-item, article.item")
            if not items:
                break

            page_has_in_range = False
            page_all_older = True
            page_all_newer_or_unknown = True

            for item in items:
                listing = self._parse_listing_item(item)
                if not listing:
                    continue
                if listing.link in seen:
                    continue
                seen.add(listing.link)

                if listing.publish_time is None:
                    # Keep unknown times; detail page will confirm
                    page_all_older = False
                    yield listing
                    continue

                pt = listing.publish_time
                if from_time <= pt <= to_time:
                    page_has_in_range = True
                    page_all_older = False
                    page_all_newer_or_unknown = False
                    yield listing
                elif pt > to_time:
                    page_all_older = False
                else:
                    # older than from_time
                    page_all_newer_or_unknown = False

            if page_has_in_range:
                older_pages = 0
            elif page_all_older and not page_all_newer_or_unknown:
                older_pages += 1
                if older_pages >= stop_after_older:
                    break
            elif not items:
                break

    def fetch_article(self, listing: ArticleListing) -> ArticleDetail:
        soup = self.fetch_soup(listing.link)
        title_el = soup.select_one("h1.detail-title, h1[data-role='title']")
        title = (title_el.get_text(" ", strip=True) if title_el else listing.title).strip()

        time_el = soup.select_one("time.publish-date, time[data-role='publishdate'], meta[property='article:published_time']")
        publish = listing.publish_time
        if time_el is not None:
            if time_el.name == "meta":
                publish = _parse_vn_datetime(time_el.get("content")) or publish
            else:
                publish = (
                    _parse_vn_datetime(time_el.get("datetime"))
                    or _parse_vn_datetime(time_el.get("title"))
                    or _parse_vn_datetime(time_el.get_text(" ", strip=True))
                    or publish
                )

        author_el = soup.select_one(".detail-author, [data-role='author']")
        author = author_el.get_text(" ", strip=True) if author_el else None

        tags = [
            a.get_text(" ", strip=True)
            for a in soup.select(".detail-tag-list a, ul[data-role='tags'] a")
            if a.get_text(" ", strip=True)
        ]
        tags_str = ", ".join(tags) if tags else None

        related_el = soup.select_one("a.OneNewsTitle")
        related = None
        if related_el and related_el.get("href"):
            related = urljoin(self.base_url + "/", related_el["href"])

        sapo_el = soup.select_one(".detail-sapo, [data-role='sapo']")
        sapo = sapo_el.get_text(" ", strip=True) if sapo_el else ""

        content_el = soup.select_one(".detail-content, [data-role='content']")
        if content_el is None:
            raise RuntimeError("Không tìm thấy nội dung bài (.detail-content)")

        # Bỏ các block phụ trong nội dung (quảng cáo / tin liên quan nhúng)
        for junk in content_el.select(
            "script, style, .VCSortableInPreviewMode, .detail-related-list, .detail-tag-list"
        ):
            junk.decompose()

        body = content_el.get_text(" ", strip=True)
        content = f"{sapo} {body}".strip() if sapo and sapo not in body[: len(sapo) + 20] else body
        if not content:
            raise RuntimeError("Nội dung bài trống")

        external_id = listing.external_id
        if not external_id:
            m = re.search(r"-(\d{10,})\.htm$", listing.link)
            external_id = m.group(1) if m else None

        # Filter: caller may still receive off-date if listing date was wrong;
        # runner verifies against target date.
        return ArticleDetail(
            title=title,
            link=listing.link,
            publish_time=publish,
            content=content,
            author=author,
            tags=tags_str,
            related_one_news=related,
            external_id=external_id,
            source=self.source_code,
        )


def detail_to_record(detail: ArticleDetail) -> dict:
    return {
        "title": detail.title,
        "publish_time": _format_publish_time(detail.publish_time),
        "content": detail.content,
        "source": detail.source or "suckhoedoisong",
        "link": detail.link,
        "author": detail.author,
        "tags": detail.tags,
        "related_one_news": detail.related_one_news,
        "external_id": detail.external_id,
        "scrape_status": "success",
    }
