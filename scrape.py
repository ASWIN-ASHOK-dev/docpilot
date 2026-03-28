import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser


def _print_progress(label: str, current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current >= total else "\r"
    print(f"{label}: [{bar}] {current}/{total}", end=end, flush=True)

def get_robot_parser(base_url: str) -> RobotFileParser:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    rp.read()
    return rp

def scrape_url(url: str) -> str:
    res = httpx.get(url, follow_redirects=True, timeout=30.0)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def _collect_sitemap_urls(sitemap_url: str, seen: set[str] | None = None) -> list[str]:
    if seen is None:
        seen = set()
    if sitemap_url in seen:
        return []
    seen.add(sitemap_url)

    try:
        res = httpx.get(sitemap_url, follow_redirects=True, timeout=30.0)
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
            urls.extend(_collect_sitemap_urls(target, seen))
        else:
            urls.append(target)
    return urls


def scrape_sitemap(sitemap_url: str) -> list[str]:
    urls = _collect_sitemap_urls(sitemap_url)
    texts: list[str] = []
    total = len(urls)
    for idx, u in enumerate(urls, start=1):
        try:
            texts.append(scrape_url(u))
        except Exception:
            pass
        _print_progress("Scraping sitemap pages", idx, total)
    return texts

def scrape_site(base_url: str, max_pages: int = 50) -> list[str]:
    rp = get_robot_parser(base_url)
    visited = set()
    to_visit = [base_url]
    texts = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        if not rp.can_fetch("*", url):
            print(f"Blocked by robots.txt: {url}")
            continue
        visited.add(url)
        _print_progress("Crawling pages", len(visited), max_pages)
        try:
            text = scrape_url(url)
            texts.append(text)
            res = httpx.get(url, follow_redirects=True)
            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.find_all("a", href=True):
                full = urljoin(base_url, a["href"])
                if full.startswith(base_url) and full not in visited:
                    to_visit.append(full)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"  Page not found (404): {url}")
            else:
                print(f"  HTTP error {e.response.status_code}: {url}")
        except Exception as e:
            print(f"  Error scraping {url}: {type(e).__name__}")
    return texts