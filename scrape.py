import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import time


REQUEST_HEADERS = {
    "User-Agent": "docpilot/0.0.1 (+https://github.com/foss-hack/docpilot)"
}
MAX_RETRIES = 4
BACKOFF_SECONDS = 0.8


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
    return _extract_text_from_soup(soup)


def _extract_text_from_soup(soup: BeautifulSoup) -> str:
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


def _should_skip_url(url: str) -> bool:
    path = urlparse(url).path
    # Skip utility/search pages and language index variants that trigger many low-value requests.
    if "/title/Special:" in path:
        return True
    if "/title/Main_page_(" in path:
        return True
    return False


def _retry_delay_seconds(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.2, float(retry_after))
            except ValueError:
                pass
    return BACKOFF_SECONDS * (2 ** attempt)


def _get_with_retries(client: httpx.Client, url: str, timeout: float) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            res = client.get(url, follow_redirects=True, timeout=timeout)
            if res.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_retry_delay_seconds(res, attempt))
                    continue
            if 500 <= res.status_code < 600 and attempt < MAX_RETRIES - 1:
                time.sleep(_retry_delay_seconds(res, attempt))
                continue
            res.raise_for_status()
            return res
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(_retry_delay_seconds(None, attempt))
                continue
            raise
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(_retry_delay_seconds(e.response, attempt))
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Request failed for {url}")


def scrape_url(url: str, client: httpx.Client | None = None) -> str:
    if client is None:
        with httpx.Client(headers=REQUEST_HEADERS) as local_client:
            res = _get_with_retries(local_client, url, timeout=30.0)
    else:
        res = _get_with_retries(client, url, timeout=30.0)
    return _extract_text(res.text)


def _fetch_html(url: str, client: httpx.Client) -> tuple[str, str]:
    res = _get_with_retries(client, url, timeout=20.0)
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
            with httpx.Client(headers=REQUEST_HEADERS) as local_client:
                res = _get_with_retries(local_client, sitemap_url, timeout=30.0)
        else:
            res = _get_with_retries(client, sitemap_url, timeout=30.0)
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
    with httpx.Client(limits=limits, headers=REQUEST_HEADERS) as client:
        urls = _collect_sitemap_urls(sitemap_url, client=client)
    texts: list[str] = []
    total = len(urls)
    if total == 0:
        return texts

    workers = max(1, min(max_workers, total))
    with httpx.Client(limits=limits, headers=REQUEST_HEADERS) as client:
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
    effective_workers = max(1, max_workers)

    with httpx.Client(limits=limits, headers=REQUEST_HEADERS) as client:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            while to_visit and len(visited) < max_pages:
                batch: list[str] = []
                while to_visit and len(batch) < effective_workers and (len(visited) + len(batch)) < max_pages:
                    candidate = to_visit.popleft()
                    if candidate in visited:
                        continue
                    if _should_skip_url(candidate):
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

                    soup = BeautifulSoup(html, "html.parser")
                    texts.append(_extract_text_from_soup(soup))
                    for a in soup.find_all("a", href=True):
                        full = _normalize_url(urljoin(url, a["href"]))
                        parsed = urlparse(full)
                        if parsed.netloc != base_netloc:
                            continue
                        if _should_skip_url(full):
                            continue
                        if full in visited or full in queued:
                            continue
                        queued.add(full)
                        to_visit.append(full)
    return texts