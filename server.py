"""
Content Scraper MCP Server
Architecture: fire-and-forget async pattern (mirrors Gemini Deep Research MCP).
- scrape_site(domain)          → starts crawl, returns job_id immediately
- get_scrape_result(job_id)    → returns status or full JSON index when done
"""

import os
import re
import json
import uuid
import asyncio
import logging
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory job store  {job_id: {"status": ..., "result": ..., "error": ...}}
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Content Scraper",
    instructions=(
        "This server crawls websites and returns a structured JSON index.\n\n"
        "WORKFLOW:\n"
        "1. Call `scrape_site` with the domain — returns a job_id immediately.\n"
        "2. Wait 5–30 minutes depending on site size, then call `get_scrape_result`.\n"
        "3. If still in_progress, wait longer and call `get_scrape_result` again.\n\n"
        "The result is a JSON index with sections: main_site and help_center.\n"
        "Each page has: url, title, summary, full_content, data_points."
    ),
)


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _is_same_domain(url: str, base: str) -> bool:
    """Check if URL belongs to the base domain or its subdomains."""
    try:
        parsed = urlparse(url)
        base_parsed = urlparse(base if base.startswith("http") else f"https://{base}")
        base_host = base_parsed.netloc.lstrip("www.")
        url_host = parsed.netloc.lstrip("www.")
        return url_host == base_host or url_host.endswith(f".{base_host}")
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes for deduplication."""
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="").geturl()
    return normalized.rstrip("/")


def _should_skip(url: str) -> bool:
    """Skip non-content URLs."""
    skip_patterns = [
        r"\.(pdf|jpg|jpeg|png|gif|svg|webp|ico|css|js|woff|woff2|ttf|eot|mp4|mp3|zip)$",
        r"(mailto:|tel:|javascript:)",
        r"/(wp-admin|wp-login|wp-json|feed|rss|atom|sitemap\.xml)",
        r"\?.*=(download|print|export)",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def _fetch_static(url: str) -> "Response | None":
    """Fetch with plain HTTP (Webflow and static sites)."""
    try:
        from scrapling.fetchers import Fetcher
        response = Fetcher.get(url, timeout=30, retries=2)
        if response.status == 200:
            return response
    except Exception as e:
        logger.warning(f"Static fetch failed for {url}: {e}")
    return None


def _fetch_dynamic(url: str) -> "Response | None":
    """Fetch with headless Playwright (JS-rendered pages like Intercom)."""
    try:
        from scrapling.fetchers import DynamicFetcher
        response = DynamicFetcher.fetch(
            url,
            headless=True,
            network_idle=True,
            timeout=60,
            disable_resources=True,
        )
        if response.status == 200:
            return response
    except Exception as e:
        logger.warning(f"Dynamic fetch failed for {url}: {e}")
    return None


def _extract_page_data(page, url: str) -> dict:
    """Extract title, summary, full_content, and data_points from a fetched page."""
    try:
        # Title
        title_el = page.css("title").first
        title = title_el.text.strip() if title_el else ""

        # Meta description as summary seed
        meta_el = page.css('meta[name="description"]').first
        meta_desc = meta_el.attrib.get("content", "").strip() if meta_el else ""

        # Full text content from body
        body_el = page.css("body").first
        raw_text = body_el.get_all_text() if body_el else page.get_all_text()

        # Clean up whitespace
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        full_content = "\n".join(lines)

        # Summary: meta description if available, else first 400 chars of content
        summary = meta_desc if meta_desc else (full_content[:400] + "..." if len(full_content) > 400 else full_content)

        # Data points: extract lines that look like facts (numbers, %, $, rules)
        data_pattern = re.compile(
            r"(\$[\d,]+|[\d,]+%|[\d,]+ (traders|accounts|payouts|days|contracts|points)|"
            r"\d+:\d+|profit split|drawdown|payout|funded|evaluation|consistency rule|"
            r"max (loss|position|contracts)|trailing|EOD|end.of.day)",
            re.IGNORECASE,
        )
        data_lines = [ln for ln in lines if data_pattern.search(ln)]
        data_points = "\n".join(f"- {ln}" for ln in data_lines[:40])  # cap at 40 bullets

        return {
            "url": url,
            "title": title,
            "summary": summary,
            "full_content": full_content,
            "data_points": data_points,
        }
    except Exception as e:
        logger.warning(f"Extraction failed for {url}: {e}")
        return {
            "url": url,
            "title": "",
            "summary": "",
            "full_content": "",
            "data_points": "",
        }


def _get_sitemap_urls(domain: str) -> list[str]:
    """Try to fetch URLs from sitemap.xml."""
    from scrapling.fetchers import Fetcher

    urls = []
    sitemap_candidates = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://www.{domain}/sitemap.xml",
    ]
    for sitemap_url in sitemap_candidates:
        try:
            response = Fetcher.get(sitemap_url, timeout=15)
            if response.status != 200:
                continue
            # Parse XML
            root = ET.fromstring(response.body)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            # Check if it's a sitemap index
            sub_sitemaps = root.findall("sm:sitemap/sm:loc", ns)
            if sub_sitemaps:
                for s in sub_sitemaps:
                    sub_resp = Fetcher.get(s.text.strip(), timeout=15)
                    if sub_resp.status == 200:
                        sub_root = ET.fromstring(sub_resp.body)
                        for loc in sub_root.findall("sm:url/sm:loc", ns):
                            u = loc.text.strip()
                            if not _should_skip(u):
                                urls.append(u)
            else:
                for loc in root.findall("sm:url/sm:loc", ns):
                    u = loc.text.strip()
                    if not _should_skip(u):
                        urls.append(u)
            if urls:
                logger.info(f"Sitemap found at {sitemap_url}: {len(urls)} URLs")
                return urls
        except Exception as e:
            logger.debug(f"Sitemap {sitemap_url} failed: {e}")
    return []


def _spider_domain(start_url: str, max_pages: int = 500) -> list[str]:
    """BFS spider for domains without sitemap (e.g. Intercom help centers)."""
    from scrapling.fetchers import Fetcher

    visited = set()
    queue = [_normalize_url(start_url)]
    found = []
    base_domain = urlparse(start_url).netloc

    while queue and len(found) < max_pages:
        url = queue.pop(0)
        if url in visited or _should_skip(url):
            continue
        visited.add(url)

        try:
            page = Fetcher.get(url, timeout=20)
            if page.status != 200:
                continue
            found.append(url)

            # Collect internal links
            for href in page.xpath("//a/@href").getall():
                absolute = urljoin(url, href)
                normalized = _normalize_url(absolute)
                parsed = urlparse(normalized)
                if (
                    parsed.scheme in ("http", "https")
                    and parsed.netloc == base_domain
                    and normalized not in visited
                    and not _should_skip(normalized)
                    and normalized not in queue
                ):
                    queue.append(normalized)
        except Exception as e:
            logger.warning(f"Spider error on {url}: {e}")

    logger.info(f"Spider found {len(found)} pages on {base_domain}")
    return found


def _categorize_url(url: str, main_domain: str) -> str:
    """Assign a section category to a URL."""
    path = urlparse(url).path.lower()
    netloc = urlparse(url).netloc

    if "help." in netloc:
        return "help_articles"
    if path.rstrip("/") == "" or url.rstrip("/") == f"https://{main_domain}":
        return "homepage"
    if re.search(r"/(post|blog)/[^/]+", path):
        return "blog_posts"
    if path in ("/blog", "/blog/"):
        return "blog_index"
    if re.search(r"vs", path):
        return "comparison_pages"
    if re.search(r"/(privacy|terms|legal|affiliate-terms|funded-trader-agreement)", path):
        return "legal_pages"
    return "feature_pages"


def _run_scrape(job_id: str, domain: str):
    """Full scrape job — runs in a thread pool."""
    try:
        _jobs[job_id]["status"] = "in_progress"
        logger.info(f"[{job_id}] Starting scrape of {domain}")

        # Strip scheme and www properly
        main_domain = domain
        for prefix in ("https://", "http://", "www."):
            if main_domain.startswith(prefix):
                main_domain = main_domain[len(prefix):]
        main_domain = main_domain.split("/")[0]
        help_domain = f"help.{main_domain}"

        # ---------------------------------------------------------------
        # 1. Discover URLs
        # ---------------------------------------------------------------
        logger.info(f"[{job_id}] Discovering URLs...")

        # Main site — try sitemap first, fall back to spider
        main_urls = _get_sitemap_urls(main_domain)
        if not main_urls:
            logger.info(f"[{job_id}] No sitemap — spidering {main_domain}")
            main_urls = _spider_domain(f"https://{main_domain}")

        # Help subdomain — always spider (Intercom has no sitemap)
        help_urls = []
        try:
            from scrapling.fetchers import Fetcher
            test = Fetcher.get(f"https://{help_domain}", timeout=10)
            if test.status == 200:
                logger.info(f"[{job_id}] Help subdomain found — spidering {help_domain}")
                help_urls = _spider_domain(f"https://{help_domain}")
        except Exception:
            logger.info(f"[{job_id}] No help subdomain found at {help_domain}")

        all_urls = list(dict.fromkeys(main_urls + help_urls))  # dedup, preserve order
        logger.info(f"[{job_id}] Total URLs to fetch: {len(all_urls)}")

        # ---------------------------------------------------------------
        # 2. Fetch and extract content
        # ---------------------------------------------------------------
        index = {
            "site": main_domain,
            "scraped_at": datetime.utcnow().isoformat() + "Z",
            "total_pages": 0,
            "main_site": {
                "homepage": [],
                "feature_pages": [],
                "comparison_pages": [],
                "blog_index": [],
                "blog_posts": [],
                "legal_pages": [],
                "other_pages": [],
            },
            "help_center": {
                "help_articles": [],
            },
        }

        for i, url in enumerate(all_urls):
            logger.info(f"[{job_id}] Fetching {i+1}/{len(all_urls)}: {url}")
            is_help = "help." in urlparse(url).netloc

            # Try static first; fall back to dynamic for help pages
            page = _fetch_static(url)
            if page is None and is_help:
                page = _fetch_dynamic(url)

            if page is None:
                logger.warning(f"[{job_id}] Skipping (no content): {url}")
                continue

            entry = _extract_page_data(page, url)
            category = _categorize_url(url, main_domain)

            if category == "help_articles":
                index["help_center"]["help_articles"].append(entry)
            elif category in index["main_site"]:
                index["main_site"][category].append(entry)
            else:
                index["main_site"]["other_pages"].append(entry)

        # ---------------------------------------------------------------
        # 3. Sort and finalize
        # ---------------------------------------------------------------
        index["main_site"]["blog_posts"].sort(key=lambda x: x["title"])
        index["help_center"]["help_articles"].sort(key=lambda x: x["title"])

        total = sum(len(v) for v in index["main_site"].values()) + \
                sum(len(v) for v in index["help_center"].values())
        index["total_pages"] = total

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = index
        logger.info(f"[{job_id}] Scrape complete. {total} pages indexed.")

    except Exception as e:
        logger.error(f"[{job_id}] Scrape failed: {e}", exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool
def scrape_page(url: str) -> str:
    """
    Scrape a single URL and return its content immediately.

    Use this to test the scraper or fetch one specific page.
    Returns title, summary, full_content, and data_points synchronously.

    Args:
        url: The full URL to scrape, e.g. "https://tradeify.co"

    Returns:
        A JSON object with url, title, summary, full_content, and data_points.
    """
    page = _fetch_static(url)
    if page is None:
        page = _fetch_dynamic(url)
    if page is None:
        return f"Failed to fetch {url} — page returned no content."

    entry = _extract_page_data(page, url)
    return json.dumps(entry, indent=2, ensure_ascii=False)


@mcp.tool
def scrape_site(domain: str) -> str:
    """
    Start a website scrape and indexing job. Returns immediately with a job_id.

    Crawls the main domain (using sitemap.xml if available, or spidering if not)
    and the help subdomain if one exists. Each page is stored with:
    url, title, summary, full_content, and data_points.

    IMPORTANT: After calling this, wait 5–30 minutes depending on site size,
    then call `get_scrape_result` with the returned job_id.

    Args:
        domain: The domain to scrape, e.g. "tradeify.co" or "kixie.com".
                Do not include https:// or trailing slashes.

    Returns:
        A message containing the job_id to use with get_scrape_result.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    # Fire and forget
    _executor.submit(_run_scrape, job_id, domain)

    return (
        f"Scrape job started for {domain}.\n\n"
        f"job_id: {job_id}\n\n"
        f"The crawler is running in the background. Wait 5–30 minutes depending "
        f"on site size, then call `get_scrape_result` with job_id='{job_id}'."
    )


@mcp.tool
def get_scrape_result(job_id: str) -> str:
    """
    Retrieve the result of a scrape job by its job_id.

    Call this after starting a job with `scrape_site`. If still running,
    wait a few more minutes and call again.

    Args:
        job_id: The ID returned by a previous `scrape_site` call.

    Returns:
        The full JSON index as a string if completed, or a status message.
    """
    job = _jobs.get(job_id)
    if not job:
        return f"No job found with job_id '{job_id}'. Please check the ID."

    status = job["status"]

    if status == "completed":
        result = job["result"]
        total = result.get("total_pages", 0)
        domain = result.get("site", "")
        return (
            f"## Scrape Complete\n\n"
            f"Site: {domain}\n"
            f"Total pages indexed: {total}\n"
            f"Scraped at: {result.get('scraped_at', '')}\n\n"
            f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```"
        )

    elif status == "failed":
        return f"Scrape failed: {job.get('error', 'Unknown error')}"

    else:
        return (
            f"Status: {status} — still running. "
            f"Please wait a few more minutes and call `get_scrape_result` again "
            f"with job_id='{job_id}'."
        )


# ---------------------------------------------------------------------------
# ASGI app
# ---------------------------------------------------------------------------

async def healthz(request):
    return JSONResponse({"status": "ok"})


mcp_asgi = mcp.http_app()

app = Starlette(
    routes=[
        Route("/healthz", healthz),
        Mount("/", app=mcp_asgi),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["mcp-session-id"],
        ),
    ],
    lifespan=mcp_asgi.lifespan,
)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
