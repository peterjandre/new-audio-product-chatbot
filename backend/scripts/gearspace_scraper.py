import argparse
import html
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree

import requests

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - runtime guard if dependency missing
    genai = None

# Gearspace RSS feed URL for New Product Alert forum
GEARSPACE_RSS_URL = "https://gearspace.com/board/external.php?type=RSS2&f=9"


@dataclass
class FeedItem:
    """Normalized representation of a Gearspace RSS item."""

    id: str
    title: str
    link: str
    published: Optional[str]
    summary: Optional[str]
    summary_model: Optional[str]
    summary_timestamp: Optional[str]
    original_hash: str
    raw_excerpt: Optional[str]

    @classmethod
    def from_xml(cls, item_element: ElementTree.Element) -> "FeedItem":
        title = _get_text(item_element, "title")
        link = _get_text(item_element, "link")
        raw_html = _get_content_encoded(item_element)
        excerpt = _clean_html(raw_html)
        guid = _get_text(item_element, "guid") or link or title
        published_raw = _get_text(item_element, "pubDate")
        published_iso = _coerce_datetime_iso(published_raw)

        normalized_id = hashlib.sha256((guid or link or title or "").encode("utf-8")).hexdigest()
        original_hash = hashlib.sha256((raw_html or "").encode("utf-8")).hexdigest()

        return cls(
            id=normalized_id,
            title=title or "",
            link=link or "",
            published=published_iso,
            summary=None,
            summary_model=None,
            summary_timestamp=None,
            original_hash=original_hash,
            raw_excerpt=excerpt,
        )


def _get_text(element: ElementTree.Element, tag: str) -> Optional[str]:
    child = element.find(tag)
    if child is None:
        return None
    if child.text is None:
        return None
    return child.text.strip() or None


def _get_content_encoded(element: ElementTree.Element) -> Optional[str]:
    namespace_tag = "{http://purl.org/rss/1.0/modules/content/}encoded"
    child = element.find(namespace_tag)
    if child is None or child.text is None:
        return None
    return child.text.strip() or None


def _clean_html(raw_html: Optional[str]) -> Optional[str]:
    if not raw_html:
        return None
    # Normalize line breaks for readability.
    text = re.sub(r"(?i)<br\\s*/?>", "\n", raw_html)
    # Remove all other tags.
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    # Collapse whitespace.
    text = re.sub(r"[ \\t\\f\\v]+", " ", text)
    text = re.sub(r"\\s*\\n\\s*", "\n", text)
    return text.strip() or None


def _coerce_datetime_iso(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def fetch_rss(feed_url: str, timeout: float = 10.0) -> str:
    response = requests.get(feed_url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_rss(xml_payload: str) -> List[FeedItem]:
    root = ElementTree.fromstring(xml_payload)
    channel = root.find("channel")
    if channel is None:
        raise ValueError("RSS payload missing channel element")

    items: List[FeedItem] = []
    for item in channel.findall("item"):
        normalized = FeedItem.from_xml(item)
        items.append(normalized)
    return items


def load_existing_corpus(path: Path) -> Dict[str, FeedItem]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse existing corpus JSON at {path}") from exc

    corpus: Dict[str, FeedItem] = {}
    for record in payload:
        corpus[record["id"]] = FeedItem(
            id=record["id"],
            title=record.get("title", ""),
            link=record.get("link", ""),
            published=record.get("published"),
            summary=record.get("summary"),
            summary_model=record.get("summary_model"),
            summary_timestamp=record.get("summary_timestamp"),
            original_hash=record.get("original_hash", ""),
            raw_excerpt=record.get("raw_excerpt"),
        )
    return corpus


def configure_gemini(api_key: Optional[str]) -> None:
    if genai is None:
        raise RuntimeError(
            "google-generativeai package not available. Install it or adjust requirements."
        )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)


def summarize_item(
    item: FeedItem,
    model_name: str,
    temperature: float,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
) -> str:
    if genai is None:
        raise RuntimeError(
            "google-generativeai package not available. Install it or adjust requirements."
        )

    # Respect request rate limits by pacing outbound calls.
    time.sleep(5.0)

    model = genai.GenerativeModel(model_name)
    prompt = (
        "Summarize the following news item about audio production gear releases in 1-2 sentences. "
        "Focus on the product name, release timing, and notable features. "
        "Do not invent details."
    )
    content = [
        {"role": "user", "parts": [{"text": prompt}]},
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        f"Title: {item.title or 'N/A'}\n"
                        f"Link: {item.link or 'N/A'}\n"
                        f"Published: {item.published or 'N/A'}\n"
                        f"Excerpt: {item.raw_excerpt or 'N/A'}"
                    )
                }
            ],
        },
    ]

    attempt = 0
    while True:
        attempt += 1
        try:
            response = model.generate_content(content, generation_config={"temperature": temperature})
            summary_text = response.text.strip()
            if not summary_text:
                raise RuntimeError("Empty summary returned from Gemini.")
            return summary_text
        except Exception as exc:  # pylint: disable=broad-except
            if attempt >= max_retries:
                raise RuntimeError(f"Gemini summarization failed after {max_retries} attempts") from exc
            sleep_for = backoff_seconds * attempt
            time.sleep(sleep_for)


def merge_items(
    existing: Dict[str, FeedItem],
    new_items: Iterable[FeedItem],
) -> Dict[str, FeedItem]:
    merged = dict(existing)
    for item in new_items:
        current = merged.get(item.id)
        if current and current.original_hash == item.original_hash:
            # Preserve previous summary if the content hasn't changed.
            item.summary = current.summary
            item.summary_model = current.summary_model
            item.summary_timestamp = current.summary_timestamp
        merged[item.id] = item
    return merged


def write_corpus(path: Path, items: Iterable[FeedItem]) -> None:
    serialized: List[Dict[str, Optional[str]]] = []
    for item in sorted(items, key=lambda entry: entry.published or "", reverse=True):
        record = asdict(item)
        # raw text is only needed to refresh summaries and shouldn't be part of the exported corpus
        record.pop("raw_excerpt", None)
        serialized.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Gearspace RSS feed and summarize entries.")
    parser.add_argument(
        "--feed-url",
        default=GEARSPACE_RSS_URL,
        help="RSS feed URL to ingest.",
    )
    parser.add_argument(
        "--output",
        default="data/gearspace_corpus.json",
        help="Path to write the summarized corpus JSON.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model name to use for summarization.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature for Gemini summaries.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional limit on the number of newest items to summarize.",
    )
    parser.add_argument(
        "--max-items-skip-cached",
        action="store_true",
        help="When used with --max-items, only count items requiring new summaries toward the limit.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Ingest feed without calling Gemini (existing summaries preserved).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)

    xml_payload = fetch_rss(args.feed_url)
    parsed_items = parse_rss(xml_payload)
    if args.max_items is not None and not args.max_items_skip_cached:
        parsed_items = parsed_items[: args.max_items]

    existing_items = load_existing_corpus(output_path)
    merged_items = merge_items(existing_items, parsed_items)

    items_sequence = list(merged_items.values())

    if args.skip_summary:
        print("Skipping Gemini summaries (--skip-summary enabled).", flush=True)
        # Remove raw excerpts before persisting to avoid leaking content.
        for idx, item in enumerate(items_sequence, start=1):
            print(f"[{idx}/{len(items_sequence)}] {item.title or item.link or item.id} -> preserved existing summary", flush=True)
            item.raw_excerpt = None
        write_corpus(output_path, items_sequence)
        return 0

    configure_gemini(os.environ.get("GEMINI_API_KEY"))

    total_items = len(items_sequence)
    summaries_generated = 0
    for idx, item in enumerate(items_sequence, start=1):
        if (
            args.max_items is not None
            and args.max_items_skip_cached
            and summaries_generated >= args.max_items
        ):
            print(f"Reached maximum of {args.max_items} new summaries; stopping.", flush=True)
            break
        display_name = item.title or item.link or item.id
        print(f"[{idx}/{total_items}] Processing {display_name}", flush=True)
        if item.summary and item.summary_model == args.model and item.original_hash:
            print("  -> using cached summary (unchanged)", flush=True)
            continue
        if not item.raw_excerpt:
            print("  -> skipped (no excerpt available)", flush=True)
            continue
        summary = summarize_item(item, args.model, args.temperature)
        item.summary = summary
        item.summary_model = args.model
        item.summary_timestamp = datetime.now(timezone.utc).isoformat()
        # Strip raw excerpt once summary is generated.
        item.raw_excerpt = None
        summaries_generated += 1
        print("  -> summary refreshed", flush=True)

    write_corpus(output_path, items_sequence)
    return 0


if __name__ == "__main__":
    sys.exit(main())

