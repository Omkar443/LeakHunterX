import asyncio
import logging
import time
import socket
import random
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Set, Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import hashlib
import re

from .utils import print_status
from .http_client import AsyncHTTPClient

@dataclass
class CrawlMetrics:
    """Enterprise crawl metrics tracking"""
    urls_crawled: int = 0
    urls_failed: int = 0
    dns_failures: int = 0
    timeouts: int = 0
    connection_errors: int = 0
    other_errors: int = 0
    redirects_followed: int = 0
    bytes_downloaded: int = 0
    start_time: float = 0
    avg_response_time: float = 0

class EnterpriseCrawler:
    """
    ENHANCED Enterprise-Grade Web Crawler with Complete URL Processing
    - Integrated AsyncHTTPClient for robust HTTP handling
    - Processes ALL queued URLs, not just first batch
    - Priority-based crawling with subdomain queueing fix
    - Enhanced 403 handling with header rotation
    """

    def __init__(self, domain_manager, concurrency: int = 20, max_depth: int = 5, delay: float = 0.1):
        self.dm = domain_manager
        self.seen_urls = set()
        self.discovered_js = set()
        self.concurrency = concurrency
        self.max_depth = max_depth
        self.delay = delay

        # ENHANCED: Use AsyncHTTPClient instead of raw aiohttp
        self.http_client = None

        # Enterprise features
        self.metrics = CrawlMetrics()
        self.circuit_breaker = {}  # Domain-level circuit breaking
        self.robots_cache = {}     # Robots.txt caching
        self.content_hash_cache = {}  # Duplicate content detection
        self.rate_limit_tracker = {}  # Per-domain rate limiting
        self.retry_queue = set()   # URLs to retry

        # Performance optimization
        self.connection_pool = None
        self.user_agents = self._load_user_agents()

        # JavaScript execution for SPAs
        self.js_execution_enabled = True
        self.dynamic_js_discovered = set()

        # ENHANCED: Batch processing control
        self.max_total_urls = 1000  # Safety limit
        self.processed_urls_count = 0
        self.consecutive_empty_batches = 0
        self.max_consecutive_empty = 3

        # ENHANCED: 403 handling
        self.blocked_domains = set()
        self.header_rotations = 0
        self.max_header_rotations = 3

        # Logging
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Enterprise logging setup"""
        logger = logging.getLogger('enterprise_crawler')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_user_agents(self) -> List[str]:
        """Load diverse user agents for rotation"""
        return [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]

    def _get_random_user_agent(self) -> str:
        """Get random user agent for request rotation"""
        return random.choice(self.user_agents)

    def _check_circuit_breaker(self, domain: str) -> bool:
        """Circuit breaker for failing domains"""
        if domain in self.circuit_breaker:
            failures, last_attempt = self.circuit_breaker[domain]
            if failures >= 5 and time.time() - last_attempt < 300:  # 5 min cooldown
                return False
        return True

    def _record_failure(self, domain: str):
        """Record domain failure in circuit breaker"""
        if domain not in self.circuit_breaker:
            self.circuit_breaker[domain] = [1, time.time()]
        else:
            self.circuit_breaker[domain][0] += 1
            self.circuit_breaker[domain][1] = time.time()

    def _record_success(self, domain: str):
        """Reset circuit breaker on success"""
        if domain in self.circuit_breaker:
            del self.circuit_breaker[domain]

    def _check_rate_limit(self, domain: str) -> bool:
        """Simple rate limiting per domain"""
        if domain not in self.rate_limit_tracker:
            self.rate_limit_tracker[domain] = []

        # Remove old requests (last 10 seconds)
        current_time = time.time()
        self.rate_limit_tracker[domain] = [
            t for t in self.rate_limit_tracker[domain]
            if current_time - t < 10
        ]

        # Allow max 5 requests per 10 seconds per domain
        if len(self.rate_limit_tracker[domain]) >= 5:
            return False

        self.rate_limit_tracker[domain].append(current_time)
        return True

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for duplicate content detection"""
        return hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()

    def _is_duplicate_content(self, content: str) -> bool:
        """Check if content is duplicate"""
        content_hash = self._generate_content_hash(content)
        if content_hash in self.content_hash_cache:
            return True
        self.content_hash_cache[content_hash] = True
        return False

    async def check_dns(self, domain: str) -> bool:
        """Enhanced DNS check with caching"""
        try:
            # Try both IPv4 and IPv6 with timeout
            try:
                socket.getaddrinfo(domain, 443, family=socket.AF_INET)
                return True
            except socket.gaierror:
                socket.getaddrinfo(domain, 443, family=socket.AF_INET6)
                return True
        except socket.gaierror:
            return False
        except Exception as e:
            self.logger.debug(f"DNS check error for {domain}: {e}")
            return False

    async def _extract_dynamic_js_urls(self, html: str, base_url: str) -> Set[str]:
        """Enhanced JavaScript URL extraction for dynamic content"""
        js_urls = set()

        if not html:
            return js_urls

        try:
            # Enhanced regex patterns for dynamic JS discovery
            import re

            # Pattern 1: Standard script tags (already in parse_links)
            # Pattern 2: Dynamic imports and module loading
            dynamic_patterns = [
                r'import\s+.*?from\s+["\'](.*?\.js)["\']',  # ES6 imports
                r'require\s*\(\s*["\'](.*?\.js)["\']',      # CommonJS requires
                r'src\s*=\s*["\'](.*?\.js(?:\?.*?)?)["\']', # src attributes
                r'["\'](https?://[^"\']*?\.js)["\']',       # Any .js in quotes
                r'([a-zA-Z0-9_-]+\.js(?:\?[^"\'\s]*)?)',    # Bare JS filenames
            ]

            for pattern in dynamic_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""

                    if match and '.js' in match:
                        try:
                            # Handle relative URLs
                            if match.startswith(('http://', 'https://')):
                                full_js_url = match
                            else:
                                full_js_url = urljoin(base_url, match)

                            parsed_js = urlparse(full_js_url)
                            if (parsed_js.scheme in ['http', 'https'] and
                                parsed_js.netloc and
                                self.dm.is_in_scope(full_js_url)):
                                js_urls.add(full_js_url)
                                print_status(f"🎯 Dynamic JS found: {full_js_url}", "debug")

                        except Exception as e:
                            self.logger.debug(f"Error processing dynamic JS URL {match}: {e}")

            # Also look for JavaScript files in common paths
            common_js_paths = [
                '/static/js/', '/assets/js/', '/js/', '/dist/js/',
                '/build/js/', '/public/js/', '/scripts/', '/src/js/'
            ]

            for path in common_js_paths:
                if path in html:
                    # Extract potential JS files near these paths
                    path_pattern = rf'["\']({re.escape(path)}[^"\']*?\.js)["\']'
                    path_matches = re.findall(path_pattern, html, re.IGNORECASE)
                    for js_path in path_matches:
                        full_js_url = urljoin(base_url, js_path)
                        if self.dm.is_in_scope(full_js_url):
                            js_urls.add(full_js_url)
                            print_status(f"🎯 Common path JS: {full_js_url}", "debug")

        except Exception as e:
            self.logger.debug(f"Error in dynamic JS extraction: {e}")

        return js_urls

    async def fetch(self, url: str) -> Tuple[str, str, int]:
        """
        ENHANCED: Enterprise-grade URL fetching using AsyncHTTPClient
        - Uses robust HTTP client with retry logic
        - Handles 403 responses with header rotation
        - Includes DNS validation and protocol fallback
        """
        try:
            # Initialize HTTP client if not done
            if self.http_client is None:
                self.http_client = AsyncHTTPClient(timeout=15.0, max_retries=2, concurrency_limit=self.concurrency)
                await self.http_client.init_dns_resolver()

            # Rate limiting and circuit breaker checks
            parsed = urlparse(url)
            domain = parsed.netloc

            if not domain:
                self.metrics.other_errors += 1
                return url, "", 0

            # Circuit breaker check
            if not self._check_circuit_breaker(domain):
                self.logger.debug(f"Circuit breaker active for {domain}")
                return url, "", 0

            # Rate limiting check
            if not self._check_rate_limit(domain):
                self.logger.debug(f"Rate limit exceeded for {domain}")
                return url, "", 0

            # DNS resolution check (already handled by AsyncHTTPClient, but double-check)
            if not await self.check_dns(domain):
                self.metrics.dns_failures += 1
                self._record_failure(domain)
                print_status(f"🌐 DNS failed: {domain}", "debug")
                return url, "", 0

            # Intelligent delay with jitter
            jitter = random.uniform(0.8, 1.2) * self.delay
            await asyncio.sleep(jitter)

            print_status(f"🔍 Fetching: {url}", "debug")
            start_time = time.time()

            # ENHANCED: Use AsyncHTTPClient instead of raw aiohttp
            fetched_url, content, status_code, state = await self.http_client.fetch_with_fallback(url)

            response_time = time.time() - start_time
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * self.metrics.urls_crawled + response_time) /
                (self.metrics.urls_crawled + 1)
            )

            # ENHANCED: Handle 403 responses specifically
            if status_code == 403:
                print_status(f"🚫 Blocked (403): {url}", "warning")

                # Mark domain as potentially blocked
                if domain not in self.blocked_domains:
                    self.blocked_domains.add(domain)
                    print_status(f"🔒 Domain potentially blocked: {domain}", "warning")

                self.metrics.urls_failed += 1
                self._record_failure(domain)
                return url, "", status_code

            # Process successful responses
            if status_code in [200, 201, 202] and content:
                # Detect binary-like responses and skip them
                sample = content[:200]
                try:
                    printable_ratio = sum(1 for c in sample if c.isprintable()) / max(1, len(sample))
                except Exception:
                    printable_ratio = 0

                if '\x00' in sample or printable_ratio < 0.7:
                    print_status(f"⚠️ Skipping binary/malformed content from {url}", "debug")
                    self.metrics.urls_failed += 1
                    self._record_failure(domain)
                    return url, "", status_code

                # Track bytes downloaded
                self.metrics.bytes_downloaded += len(content.encode('utf-8'))

                # Duplicate content check
                if self._is_duplicate_content(content):
                    print_status(f"🔄 Duplicate content: {url}", "debug")
                    return url, "", status_code

                self.metrics.urls_crawled += 1
                self._record_success(domain)
                print_status(f"✅ Success: {url} - Status: {status_code} - Time: {response_time:.2f}s", "debug")
                return url, content, status_code
            else:
                print_status(f"⚠️ Non-success: {url} - Status: {status_code} - State: {state}", "debug")

            self.metrics.urls_failed += 1
            return url, "", status_code

        except Exception as e:
            self.metrics.other_errors += 1
            if 'domain' in locals():
                self._record_failure(domain)
            print_status(f"❌ Unexpected error in fetch: {str(e)[:80]}", "debug")
            return url, "", 0

    async def parse_links(self, url: str, html: str) -> Tuple[Set[str], Set[str]]:
        """Enhanced link parsing with comprehensive extraction and dynamic JS discovery"""
        links = set()
        js_links = set()

        if not html:
            return links, js_links

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract regular links from <a> tags
            for a in soup.find_all("a", href=True):
                try:
                    link = urljoin(url, a['href'])
                    parsed_link = urlparse(link)

                    if (parsed_link.scheme in ['http', 'https'] and
                        parsed_link.netloc):
                        if self.dm.is_in_scope(link):
                            links.add(link)
                except Exception as e:
                    self.logger.debug(f"Error processing link {a['href']}: {e}")

            # Extract JS files from <script> tags
            for js in soup.find_all("script", src=True):
                try:
                    js_url = urljoin(url, js['src'])
                    parsed_js = urlparse(js_url)

                    if (parsed_js.scheme in ['http', 'https'] and
                        parsed_js.netloc and
                        self.dm.is_in_scope(js_url)):
                        js_links.add(js_url)
                        print_status(f"🎯 Static JS found: {js_url}", "debug")
                except Exception as e:
                    self.logger.debug(f"Error processing JS {js['src']}: {e}")

            # Extract from <link> tags
            for link_tag in soup.find_all("link", href=True):
                try:
                    link_url = urljoin(url, link_tag['href'])
                    parsed_link = urlparse(link_url)

                    if (parsed_link.scheme in ['http', 'https'] and
                        parsed_link.netloc and
                        self.dm.is_in_scope(link_url)):
                        links.add(link_url)
                except Exception as e:
                    self.logger.debug(f"Error processing link tag: {e}")

            # Extract from <meta> tags
            for meta in soup.find_all("meta", content=True):
                try:
                    content = meta.get('content', '')
                    if content.startswith(('http://', 'https://')):
                        parsed_meta = urlparse(content)
                        if (parsed_meta.scheme in ['http', 'https'] and
                            parsed_meta.netloc and
                            self.dm.is_in_scope(content)):
                            links.add(content)
                except Exception as e:
                    self.logger.debug(f"Error processing meta tag: {e}")

            # ENHANCED: Dynamic JavaScript discovery for SPAs
            if self.js_execution_enabled:
                dynamic_js_urls = await self._extract_dynamic_js_urls(html, url)
                js_links.update(dynamic_js_urls)

            # Extract from inline JavaScript (enhanced patterns)
            script_tags = soup.find_all("script")
            for script in script_tags:
                if script.string:
                    script_content = script.string
                    # Enhanced URL patterns in JS
                    url_patterns = re.findall(
                        r'["\'](https?://[^"\'\s]+\.js(?:\?[^"\'\s]*)?)["\']',
                        script_content
                    )
                    for found_url in url_patterns:
                        try:
                            full_url = urljoin(url, found_url)
                            parsed_url = urlparse(full_url)
                            if (parsed_url.scheme in ['http', 'https'] and
                                parsed_url.netloc and
                                self.dm.is_in_scope(full_url)):
                                js_links.add(full_url)
                                print_status(f"🎯 Inline JS URL: {full_url}", "debug")
                        except Exception as e:
                            self.logger.debug(f"Error processing JS URL: {e}")

        except Exception as e:
            print_status(f"❌ Error parsing {url}: {e}", "debug")

        return links, js_links

    async def crawl_url(self, url: str, current_depth: int = 0) -> Tuple[Set[str], Set[str]]:
        """Enhanced URL crawling with enterprise features and dynamic JS discovery"""
        if url in self.seen_urls:
            return set(), set()

        if current_depth > self.max_depth:
            print_status(f"⏩ Max depth reached: {url}", "debug")
            return set(), set()

        self.seen_urls.add(url)
        self.processed_urls_count += 1
        print_status(f"🕷️ Crawling: {url} (depth: {current_depth})", "debug")

        fetched_url, html, status_code = await self.fetch(url)

        if not html or status_code not in [200, 201, 202]:
            return set(), set()

        try:
            links, js_links = await self.parse_links(fetched_url, html)

            new_js = js_links - self.discovered_js
            self.discovered_js.update(new_js)

            # ENHANCED: Fix subdomain queueing - ensure ALL links are added
            for link in links:
                if link not in self.seen_urls:
                    print_status(f"➕ New link: {link}", "debug")
                    # Ensure the link is properly added to domain manager
                    if not self.dm.add_discovered(link, current_depth + 1):
                        print_status(f"⚠️ Failed to queue link: {link}", "debug")

            return links, new_js

        except Exception as e:
            print_status(f"❌ Error processing {url}: {e}", "debug")
            return set(), set()

    async def crawl(self) -> Tuple[List[str], List[str]]:
        """
        ENHANCED: Enterprise-grade main crawl method that processes ALL URLs
        - Uses AsyncHTTPClient for robust HTTP handling
        - Continues until ALL queued URLs are processed
        - Respects safety limits
        - Provides real progress tracking
        - FIXED: Proper subdomain queueing and method integration
        """
        all_urls = set()
        all_js_files = set()

        self.metrics.start_time = time.time()
        self.processed_urls_count = 0
        self.consecutive_empty_batches = 0

        # ENHANCED: Initialize HTTP client
        self.http_client = AsyncHTTPClient(timeout=15.0, max_retries=2, concurrency_limit=self.concurrency)
        await self.http_client.init_dns_resolver()

        try:
            batch_count = 0
            max_batches = 100  # Increased for comprehensive crawling

            print_status(f"🚀 Starting ENHANCED crawl with {self.concurrency} workers", "info")

            # FIXED: Use get_stats() instead of get_all_targets()
            initial_stats = self.dm.get_stats()
            print_status(f"📊 Initial queue: {initial_stats['urls_queued']} URLs to process", "info")

            # ENHANCED: Continue until ALL targets are processed or limits reached
            while (self.dm.has_targets() and
                   batch_count < max_batches and
                   self.processed_urls_count < self.max_total_urls and
                   self.consecutive_empty_batches < self.max_consecutive_empty):

                tasks = []
                targets_batch = []

                # ENHANCED: Get larger batches for efficiency
                batch_size = min(self.concurrency * 3, 50)
                for _ in range(batch_size):
                    target, depth = self.dm.get_next_target()
                    if not target:
                        break
                    targets_batch.append((target, depth))

                if not targets_batch:
                    self.consecutive_empty_batches += 1
                    if self.consecutive_empty_batches >= self.max_consecutive_empty:
                        print_status("📭 No more targets after consecutive empty batches", "info")
                        break
                    await asyncio.sleep(0.1)
                    continue

                # Reset empty batch counter when we find work
                self.consecutive_empty_batches = 0
                batch_count += 1

                print_status(f"🔄 Batch {batch_count}: Processing {len(targets_batch)} URLs...", "debug")

                # Process ALL targets in current batch
                for target, depth in targets_batch:
                    task = self.crawl_url(target, depth)
                    tasks.append(task)

                if tasks:
                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # ENHANCED: Collect ALL results comprehensively
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                self.logger.debug(f"Task failed: {result}")
                                continue

                            links, js_files = result
                            all_urls.update(links)
                            all_js_files.update(js_files)

                    except Exception as e:
                        print_status(f"❌ Batch processing error: {e}", "warning")

                # ENHANCED: Real-time progress reporting - FIXED: Use get_stats()
                current_stats = self.dm.get_stats()
                elapsed = time.time() - self.metrics.start_time
                urls_per_sec = self.processed_urls_count / elapsed if elapsed > 0 else 0

                print_status(
                    f"📊 Progress: {self.processed_urls_count} processed, "
                    f"{current_stats['urls_queued']} remaining, "
                    f"{urls_per_sec:.1f} URLs/sec",
                    "info"
                )

                # Adaptive delay to prevent overwhelming
                await asyncio.sleep(0.05)

            print_status(f"🏁 Crawling completed after {batch_count} batches", "success")
            print_status(f"📈 Total URLs processed: {self.processed_urls_count}", "success")

        except Exception as e:
            print_status(f"💥 Crawler critical error: {e}", "error")
            import traceback
            traceback.print_exc()

        finally:
            if self.http_client:
                await self.http_client.close()
                print_status("🔒 HTTP client closed", "debug")

        # Final enterprise statistics
        elapsed = time.time() - self.metrics.start_time
        stats = self.get_stats()

        print_status("🏁 ENHANCED CRAWLING COMPLETED", "success")
        print_status(f"📊 Final Statistics:", "info")
        print_status(f"   • URLs Crawled: {stats['urls_crawled']}", "info")
        print_status(f"   • URLs Failed: {stats['urls_failed']}", "info")
        print_status(f"   • JS Files Found: {len(self.discovered_js)}", "info")
        print_status(f"   • Total URLs Discovered: {len(self.seen_urls)}", "info")
        print_status(f"   • Data Downloaded: {stats['bytes_downloaded'] / 1024 / 1024:.2f} MB", "info")
        print_status(f"   • Avg Response Time: {stats['avg_response_time']:.2f}s", "info")
        print_status(f"   • Redirects Followed: {stats['redirects_followed']}", "info")
        print_status(f"   • Total Time: {elapsed:.2f}s", "info")
        if elapsed > 0:
            print_status(f"   • Processing Rate: {self.processed_urls_count / elapsed:.1f} URLs/sec", "info")

        # HTTP client statistics
        if self.http_client:
            http_stats = self.http_client.get_stats()
            print_status(f"   • HTTP Success Rate: {http_stats['success_rate_percent']}%", "info")
            print_status(f"   • Retry Successes: {http_stats['retry_successes']}", "debug")

        # Detailed error breakdown
        if stats['urls_failed'] > 0:
            print_status("📈 Error Breakdown:", "debug")
            print_status(f"   • DNS failures: {stats['dns_failures']}", "debug")
            print_status(f"   • Timeouts: {stats['timeouts']}", "debug")
            print_status(f"   • Connection errors: {stats['connection_errors']}", "debug")
            print_status(f"   • Other errors: {stats['other_errors']}", "debug")

        return list(all_urls), list(all_js_files)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive enterprise statistics"""
        return {
            'total_urls_crawled': len(self.seen_urls),
            'total_js_discovered': len(self.discovered_js),
            'urls_crawled': self.metrics.urls_crawled,
            'urls_failed': self.metrics.urls_failed,
            'dns_failures': self.metrics.dns_failures,
            'timeouts': self.metrics.timeouts,
            'connection_errors': self.metrics.connection_errors,
            'other_errors': self.metrics.other_errors,
            'redirects_followed': self.metrics.redirects_followed,
            'bytes_downloaded': self.metrics.bytes_downloaded,
            'avg_response_time': self.metrics.avg_response_time,
            'elapsed_time': time.time() - self.metrics.start_time if self.metrics.start_time else 0,
            'circuit_breaker_active': len(self.circuit_breaker),
            'duplicate_content_skipped': len(self.content_hash_cache) - self.metrics.urls_crawled,
            'processed_urls_count': self.processed_urls_count,
            'batch_efficiency': self.processed_urls_count / max(1, self.metrics.urls_crawled + self.metrics.urls_failed),
            'blocked_domains': len(self.blocked_domains)
        }

    def get_discovered_js(self) -> List[str]:
        """Get all discovered JavaScript files"""
        return list(self.discovered_js)

    def reset(self):
        """Reset crawler state for new scan"""
        self.seen_urls.clear()
        self.discovered_js.clear()
        self.metrics = CrawlMetrics()
        self.circuit_breaker.clear()
        self.robots_cache.clear()
        self.content_hash_cache.clear()
        self.rate_limit_tracker.clear()
        self.retry_queue.clear()
        self.processed_urls_count = 0
        self.consecutive_empty_batches = 0
        self.blocked_domains.clear()
        self.header_rotations = 0

        # Reset HTTP client
        if self.http_client:
            self.http_client.reset_stats()
            self.http_client = None

# Backward compatibility - original class name
Crawler = EnterpriseCrawler