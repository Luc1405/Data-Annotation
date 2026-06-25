from __future__ import annotations

import argparse
import csv
import hashlib
import html.parser
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT_URL = "https://api.pdok.nl/"
DEFAULT_OUTPUT_CSV = Path("input_data/pdok_coreconcept_annotations.csv")
DEFAULT_DATASETS_DIR = Path("input_data/datasets")
DEFAULT_USER_AGENT = "pdok-coreconcept-annotation-fetcher/1.0"


class LinkExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class OgcService:
    landing_url: str
    title: str
    description: str
    keywords: list[str]
    collections_url: str


@dataclass(frozen=True)
class OgcCollection:
    service: OgcService
    collection_id: str
    title: str
    description: str
    keywords: list[str]
    items_url: str
    html_url: str


def request_url(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, timeout: int = 120) -> dict[str, Any]:
    body = request_url(url, timeout=timeout)
    return json.loads(body.decode("utf-8"))


def fetch_text(url: str, timeout: int = 120) -> str:
    return request_url(url, timeout=timeout).decode("utf-8", errors="replace")


def with_query_params(url: str, **params: str | int | None) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = str(value)
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def normalize_json_url(url: str) -> str:
    return with_query_params(url, f="json")


def link_href(payload: dict[str, Any], rel_values: set[str], type_contains: str | None = None) -> str | None:
    for link in payload.get("links", []):
        if not isinstance(link, dict):
            continue
        rel = str(link.get("rel", ""))
        href = link.get("href")
        link_type = str(link.get("type", ""))
        if rel in rel_values and isinstance(href, str):
            if type_contains is None or type_contains.lower() in link_type.lower():
                return href
    return None


def keyword_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords: list[str] = []
    for item in value:
        if isinstance(item, str):
            keywords.append(item)
        elif isinstance(item, dict) and item.get("keyword"):
            keywords.append(str(item["keyword"]))
    return keywords


def safe_slug(text: str, max_length: int = 120) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "pdok_layer"
    return text[:max_length].strip("_")


def stable_short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def discover_ogc_landing_urls(root_url: str, limit: int | None = None) -> list[str]:
    html = fetch_text(root_url)
    extractor = LinkExtractor()
    extractor.feed(html)

    urls: list[str] = []
    seen: set[str] = set()
    for href in extractor.links:
        absolute = urllib.parse.urljoin(root_url, href)
        parsed = urllib.parse.urlparse(absolute)
        if parsed.netloc != "api.pdok.nl":
            continue
        # PDOK OGC landing pages usually contain /ogc/ and end at a version path.
        if "/ogc/" not in parsed.path:
            continue
        normalized = urllib.parse.urlunparse(parsed._replace(query="", fragment=""))
        normalized = normalized.rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
        if limit is not None and len(urls) >= limit:
            break
    return urls


def load_service(landing_url: str) -> OgcService:
    landing_json_url = normalize_json_url(landing_url)
    payload = fetch_json(landing_json_url)
    collections_url = link_href(
        payload,
        {
            "data",
            "http://www.opengis.net/def/rel/ogc/1.0/data",
        },
        type_contains="json",
    )
    if not collections_url:
        collections_url = landing_url.rstrip("/") + "/collections"
    return OgcService(
        landing_url=landing_json_url,
        title=str(payload.get("title") or landing_url),
        description=str(payload.get("description") or ""),
        keywords=keyword_list(payload.get("keywords")),
        collections_url=normalize_json_url(collections_url),
    )


def load_collections(service: OgcService) -> list[OgcCollection]:
    payload = fetch_json(service.collections_url)
    collections = payload.get("collections", [])
    result: list[OgcCollection] = []

    for collection in collections:
        if not isinstance(collection, dict):
            continue
        collection_id = str(collection.get("id") or collection.get("name") or "").strip()
        if not collection_id:
            continue

        items_url = link_href(
            collection,
            {"items", "http://www.opengis.net/def/rel/ogc/1.0/items"},
            type_contains="json",
        )
        if not items_url:
            items_url = service.collections_url.split("/collections", 1)[0].rstrip("/")
            items_url = f"{items_url}/collections/{urllib.parse.quote(collection_id, safe='')}/items"

        html_url = link_href(collection, {"self", "alternate"}, type_contains="html")
        if not html_url:
            html_url = with_query_params(
                service.collections_url.split("/collections", 1)[0].rstrip("/")
                + f"/collections/{urllib.parse.quote(collection_id, safe='')}",
                f="html",
            )

        result.append(
            OgcCollection(
                service=service,
                collection_id=collection_id,
                title=str(collection.get("title") or collection_id),
                description=str(collection.get("description") or ""),
                keywords=keyword_list(collection.get("keywords")),
                items_url=normalize_json_url(items_url),
                html_url=html_url,
            )
        )
    return result


def next_url_from_links(payload: dict[str, Any]) -> str | None:
    for link in payload.get("links", []):
        if isinstance(link, dict) and link.get("rel") == "next" and isinstance(link.get("href"), str):
            return normalize_json_url(str(link["href"]))
    return None


def fetch_collection_geojson(collection: OgcCollection, limit: int, page_size: int, delay: float) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    first_url = with_query_params(collection.items_url, f="json", limit=min(page_size, limit))
    current_url: str | None = first_url
    number_matched: int | None = None
    crs: Any = None
    bbox: Any = None

    while current_url and len(features) < limit:
        payload = fetch_json(current_url)
        if number_matched is None and isinstance(payload.get("numberMatched"), int):
            number_matched = payload["numberMatched"]
        if crs is None and payload.get("crs") is not None:
            crs = payload.get("crs")
        if bbox is None and payload.get("bbox") is not None:
            bbox = payload.get("bbox")

        page_features = payload.get("features", [])
        if not isinstance(page_features, list):
            break
        remaining = limit - len(features)
        features.extend(page_features[:remaining])

        current_url = next_url_from_links(payload)
        if current_url and len(features) < limit and delay > 0:
            time.sleep(delay)

    geojson: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
        "pdok_source": {
            "service_title": collection.service.title,
            "service_description": collection.service.description,
            "service_keywords": collection.service.keywords,
            "service_landing_url": collection.service.landing_url,
            "collection_id": collection.collection_id,
            "collection_title": collection.title,
            "collection_description": collection.description,
            "collection_keywords": collection.keywords,
            "collection_url": collection.html_url,
            "items_url": first_url,
            "number_matched": number_matched,
            "number_fetched": len(features),
            "feature_limit_used": limit,
        },
    }
    if crs is not None:
        geojson["crs"] = crs
    if bbox is not None:
        geojson["bbox"] = bbox
    return geojson


def build_map_description(collection: OgcCollection) -> str:
    parts = []
    if collection.service.description:
        parts.append(f"Service description: {collection.service.description}")
    if collection.description:
        parts.append(f"Collection description: {collection.description}")
    keywords = sorted(set(collection.service.keywords + collection.keywords))
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))
    return "\n\n".join(parts)


def collection_matches(collection: OgcCollection, search_terms: list[str]) -> bool:
    if not search_terms:
        return True
    haystack = "\n".join(
        [
            collection.service.title,
            collection.service.description,
            " ".join(collection.service.keywords),
            collection.collection_id,
            collection.title,
            collection.description,
            " ".join(collection.keywords),
        ]
    ).lower()
    return all(term.lower() in haystack for term in search_terms)


def write_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Title",
        "EnglishTitle",
        "PageLink",
        "MapLink",
        "Kaartlaag",
        "Geometry",
        "Entity",
        "MapDescriptionTitle",
        "MapDescription",
        "MapDescriptionSource",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch PDOK OGC API Features collections and create an annotation CSV plus "
            "GeoJSON files that can be used by the existing LLM annotation script."
        )
    )
    parser.add_argument(
        "--root-url",
        default=DEFAULT_ROOT_URL,
        help="PDOK index page used when --discover-root is enabled. Default: https://api.pdok.nl/",
    )
    parser.add_argument(
        "--discover-root",
        action="store_true",
        help="Discover OGC API landing pages from the PDOK API index page.",
    )
    parser.add_argument(
        "--api-url",
        action="append",
        default=[],
        help=(
            "Specific OGC API landing page to fetch, for example "
            "https://api.pdok.nl/lv/bgt/ogc/v1. Can be used multiple times."
        ),
    )
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        help="Only keep collections whose service/collection title, description or keywords contain this term. Can be repeated.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--max-services", type=int, default=None)
    parser.add_argument("--max-collections", type=int, default=50)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between paginated API requests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset JSON files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.datasets_dir.mkdir(parents=True, exist_ok=True)

    landing_urls: list[str] = []
    if args.discover_root or not args.api_url:
        landing_urls.extend(discover_ogc_landing_urls(args.root_url, limit=args.max_services))
    landing_urls.extend(args.api_url)

    # Stable de-duplication while preserving order.
    seen_urls: set[str] = set()
    landing_urls = [url.rstrip("/") for url in landing_urls if not (url.rstrip("/") in seen_urls or seen_urls.add(url.rstrip("/")))]

    rows: list[dict[str, str]] = []
    processed_collections = 0

    for landing_url in landing_urls:
        if args.max_collections is not None and processed_collections >= args.max_collections:
            break
        try:
            service = load_service(landing_url)
            collections = load_collections(service)
        except Exception as exc:
            print(f"Skipping service {landing_url}: {type(exc).__name__}: {exc}")
            continue

        for collection in collections:
            if args.max_collections is not None and processed_collections >= args.max_collections:
                break
            if not collection_matches(collection, args.search):
                continue

            base_slug = safe_slug(f"pdok_{safe_slug(service.title, 50)}_{collection.collection_id}")
            kaartlaag = f"{base_slug}_{stable_short_hash(collection.items_url)}"
            json_path = args.datasets_dir / f"{kaartlaag}.json"

            if json_path.exists() and not args.overwrite:
                print(f"Keeping existing {json_path}")
            else:
                try:
                    geojson = fetch_collection_geojson(
                        collection=collection,
                        limit=args.max_features,
                        page_size=args.page_size,
                        delay=args.delay,
                    )
                    json_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    print(f"Skipping collection {service.title} / {collection.collection_id}: {type(exc).__name__}: {exc}")
                    continue

            rows.append(
                {
                    "Title": collection.service.title,
                    "EnglishTitle": collection.title,
                    "PageLink": collection.service.landing_url,
                    "MapLink": collection.html_url,
                    "Kaartlaag": kaartlaag,
                    # Leave gold labels empty. The annotation script requires these columns,
                    # but for new PDOK data there is normally no expert reference label yet.
                    "Geometry": "",
                    "Entity": "",
                    "MapDescriptionTitle": collection.title,
                    "MapDescription": build_map_description(collection),
                    "MapDescriptionSource": "PDOK OGC API Features landing page and collection metadata",
                }
            )
            processed_collections += 1
            print(f"Added {collection.service.title} / {collection.collection_id} -> {kaartlaag}")

    write_csv(rows, args.output_csv)
    print(f"\nWrote {len(rows)} CSV rows to {args.output_csv}")
    print(f"Wrote/used GeoJSON files in {args.datasets_dir}")
    if not rows:
        print("No rows were written. Try a specific --api-url or remove/adjust --search filters.")


if __name__ == "__main__":
    main()
