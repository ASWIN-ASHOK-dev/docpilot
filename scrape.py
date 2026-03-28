import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque


def _print_progress(label: str, current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current >= total else "\r"
    print(f"{label}: [{bar}] {current}/{total}", end=end, flush=True)


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="")
    normalized = cleaned.geturl()
    if normalized.endswith("/") and len(normalized) > len(f"{cleaned.scheme}://{cleaned.netloc}/"):
        normalized = normalized.rstrip("/")
    return normalized


def scrape_url(url: str, client: httpx.Client | None = None) -> str:
    if client is None:
        res = httpx.get(url, follow_redirects=True, timeout=30.0)
    else:
        res = client.get(url, follow_redirects=True, timeout=30.0)
    res.raise_for_status()
    return _extract_text(res.text)


def _fetch_html(url: str, client: httpx.Client) -> tuple[str, str]:
    res = client.get(url, follow_redirects=True, timeout=20.0)
    res.raise_for_status()
    return url, res.text


def _collect_sitemap_urls(
    sitemap_url: str,
    seen: set[str] | None = None,
    client: httpx.Client | None = None,
) -> list[str]:
    if seen is None:
        seen = set()
    if sitemap_url in seen:
        return []
    seen.add(sitemap_url)

    try:
        if client is None:
            res = httpx.get(sitemap_url, follow_redirects=True, timeout=30.0)
        else:
            res = client.get(sitemap_url, follow_redirects=True, timeout=30.0)
        res.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"Sitemap not found (404): {sitemap_url}")
        else:
            print(f"Failed to fetch sitemap (HTTP {e.response.status_code}): {sitemap_url}")
        return []
    except Exception as e:
        print(f"Error fetching sitemap: {sitemap_url} - {e}")
        return []

    soup = BeautifulSoup(res.text, "xml")
    if not soup.find():
        print(f"Invalid sitemap XML from: {sitemap_url}")
        return []

    urls: list[str] = []
    for loc in soup.find_all("loc"):
        if not loc.text:
            continue
        target = loc.text.strip()
        if not target:
            continue
        if target.endswith(".xml") or target.endswith(".xml.gz"):
            urls.extend(_collect_sitemap_urls(target, seen, client=client))
        else:
            urls.append(target)
    return urls


def scrape_sitemap(sitemap_url: str, max_workers: int = 16) -> list[str]:
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    with httpx.Client(limits=limits) as client:
        urls = _collect_sitemap_urls(sitemap_url, client=client)
    texts: list[str] = []
    total = len(urls)
    if total == 0:
        return texts

    workers = max(1, min(max_workers, total))
    with httpx.Client(limits=limits) as client:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(scrape_url, u, client) for u in urls]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    texts.append(future.result())
                except Exception:
                    pass
                _print_progress("Scraping sitemap pages", completed, total)
    return texts

def scrape_site(base_url: str, max_pages: int = 100, max_workers: int = 16) -> list[str]:
    base_url = _normalize_url(base_url)
    base_netloc = urlparse(base_url).netloc

    visited: set[str] = set()
    queued: set[str] = {base_url}
    to_visit: deque[str] = deque([base_url])
    texts = []
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    max_workers = max(1, max_workers)
    with httpx.Client(limits=limits) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while to_visit and len(visited) < max_pages:
                batch: list[str] = []
                while to_visit and len(batch) < max_workers and (len(visited) + len(batch)) < max_pages:
                    candidate = to_visit.popleft()
                    if candidate in visited:
                        continue
                    batch.append(candidate)

                futures = {executor.submit(_fetch_html, url, client): url for url in batch}
                for future in as_completed(futures):
                    url = futures[future]
                    visited.add(url)
                    _print_progress("Crawling pages", len(visited), max_pages)
                    try:
                        _, html = future.result()
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            print(f"  Page not found (404): {url}")
                        else:
                            print(f"  HTTP error {e.response.status_code}: {url}")
                        continue
                    except Exception as e:
                        print(f"  Error scraping {url}: {type(e).__name__}")
                        continue

                    texts.append(_extract_text(html))

                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        full = _normalize_url(urljoin(url, a["href"]))
                        parsed = urlparse(full)
                        if parsed.netloc != base_netloc:
                            continue
                        if full in visited or full in queued:
                            continue
                        queued.add(full)
                        to_visit.append(full)
    return texts