#!/usr/bin/env python3
"""
LeakHunterX - COMPLETE FEATURE-RICH CRAWLER
All features preserved + fixed architecture
"""

import asyncio
import aiohttp
import logging
import time
import socket
import random
import hashlib
from urllib.parse import urljoin, urlparse
from typing import Set, Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup

from .utils import print_status
from .domain_manager import DomainManager, URLState


@dataclass
class CrawlMetrics:
    """Comprehensive metrics tracking"""
    urls_crawled: int = 0
    urls_failed: int = 0
    dns_failures: int = 0
    timeouts: int = 0
    connection_errors: int = 0
    http_errors: int = 0
    other_errors: int = 0
    blocked_403: int = 0
    bypass_attempts: int = 0
    bytes_downloaded: int = 0
    redirects_followed: int = 0
    js_files_found: int = 0
    links_discovered: int = 0
    start_time: float = 0
    avg_response_time: float = 0


class CompleteCrawler:
    """
    COMPLETE FEATURE-RICH CRAWLER
    - All original features preserved
    - Fixed architecture and imports
    - Enhanced error handling
    """

    def __init__(self, domain_manager: DomainManager, concurrency: int = 15, max_depth: int = 3, delay: float = 0.1):
        self.dm = domain_manager
        self.seen_urls: Set[str] = set()
        self.discovered_js: Set[str] = set()
        self.concurrency = min(concurrency, 20)
        self.max_depth = max_depth
        self.delay = delay
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Comprehensive metrics
        self.metrics = CrawlMetrics()
        self.content_hash_cache: Set[str] = set()
        
        # Enhanced features
        self.circuit_breaker: Dict[str, List] = {}
        self.rate_limit_tracker: Dict[str, List] = {}
        self.bypass_attempts_log: Dict[str, int] = {}
        self.retry_attempts: Dict[str, int] = {}
        
        # Enhanced user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        ]
        
        # Setup logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup proper logging"""
        logger = logging.getLogger('complete_crawler')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _check_circuit_breaker(self, domain: str) -> bool:
        """Circuit breaker for failing domains"""
        if domain in self.circuit_breaker:
            failures, last_attempt = self.circuit_breaker[domain]
            cooldown = 300  # 5 minutes
            
            if failures >= 3 and time.time() - last_attempt < cooldown:
                self.logger.debug(f"🚫 Circuit breaker active for {domain}")
                return False
                
            if time.time() - last_attempt >= cooldown:
                del self.circuit_breaker[domain]
                
        return True

    def _check_rate_limit(self, domain: str) -> bool:
        """Rate limiting per domain"""
        if domain not in self.rate_limit_tracker:
            self.rate_limit_tracker[domain] = []
            
        current_time = time.time()
        # Clean old requests (last 15 seconds)
        self.rate_limit_tracker[domain] = [
            t for t in self.rate_limit_tracker[domain]
            if current_time - t < 15
        ]
        
        max_requests = 8  # Max 8 requests per 15 seconds
        if len(self.rate_limit_tracker[domain]) >= max_requests:
            self.logger.debug(f"⏰ Rate limit exceeded for {domain}")
            return False
            
        self.rate_limit_tracker[domain].append(current_time)
        return True

    def _record_failure(self, domain: str, error_type: str):
        """Record failure for circuit breaker"""
        if domain not in self.circuit_breaker:
            self.circuit_breaker[domain] = [1, time.time()]
        else:
            self.circuit_breaker[domain][0] += 1
            self.circuit_breaker[domain][1] = time.time()

    def _record_success(self, domain: str):
        """Reset circuit breaker on success"""
        if domain in self.circuit_breaker:
            del self.circuit_breaker[domain]

    async def check_dns(self, domain: str) -> bool:
        """Enhanced DNS resolution check"""
        try:
            # Try IPv4 and IPv6
            socket.getaddrinfo(domain, 443, family=socket.AF_INET)
            return True
        except socket.gaierror:
            try:
                socket.getaddrinfo(domain, 443, family=socket.AF_INET6)
                return True
            except socket.gaierror:
                return False
        except Exception:
            return False

    def _get_enhanced_headers(self) -> Dict[str, str]:
        """Enhanced headers with better browser fingerprinting"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/avif,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'no-cache',
            'DNT': '1'
        }

    def _is_duplicate_content(self, content: str) -> bool:
        """Enhanced duplicate content detection"""
        if not content or len(content) < 100:
            return True
            
        content_hash = hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()
        if content_hash in self.content_hash_cache:
            return True
            
        self.content_hash_cache.add(content_hash)
        return False

    def _is_valid_content(self, content: str) -> bool:
        """Validate content is actual HTML/JS"""
        if not content or len(content) < 500:
            return False
            
        # Check for common HTML/JS patterns
        html_indicators = ['<html', '<!DOCTYPE', '<script', '<div', '<body']
        js_indicators = ['function', 'var ', 'const ', 'let ', '.js']
        
        content_lower = content.lower()
        has_html = any(indicator in content_lower for indicator in html_indicators)
        has_js = any(indicator in content_lower for indicator in js_indicators)
        
        return has_html or has_js

    async def _try_403_bypass(self, url: str, domain: str) -> Tuple[str, str, int]:
        """
        COMPLETE 403 bypass with multiple techniques
        Returns: (url, content, status_code)
        """
        self.logger.debug(f"🔄 Starting 403 bypass for {domain}")
        
        bypass_techniques = [
            # Technique 1: Protocol fallback
            {'type': 'protocol', 'url': url.replace('https://', 'http://')},
            # Technique 2: Mobile user agent
            {'type': 'mobile', 'headers': {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'
            }},
            # Technique 3: Bot user agent
            {'type': 'bot', 'headers': {
                'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
            }},
            # Technique 4: Simple headers
            {'type': 'simple', 'headers': {
                'User-Agent': 'curl/7.68.0',
                'Accept': '*/*'
            }}
        ]
        
        for technique in bypass_techniques:
            try:
                technique_type = technique['type']
                test_url = technique.get('url', url)
                headers = technique.get('headers', self._get_enhanced_headers())
                
                self.logger.debug(f"  🔧 Bypass technique: {technique_type}")
                
                timeout = aiohttp.ClientTimeout(total=8)
                connector = aiohttp.TCPConnector(ssl=False)
                
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers=headers
                ) as session:
                    
                    async with session.get(test_url, ssl=False, allow_redirects=True) as response:
                        content = await response.text(errors='ignore')
                        
                        if response.status == 200 and self._is_valid_content(content):
                            self.metrics.bypass_attempts += 1
                            self.logger.debug(f"  ✅ Bypass success with {technique_type}")
                            return url, content, response.status
                        else:
                            self.logger.debug(f"  ❌ Bypass failed: HTTP {response.status}")
                            
            except Exception as e:
                self.logger.debug(f"  ❌ Bypass error: {e}")
        
        return url, "", 403

    async def fetch_url(self, url: str) -> Tuple[str, str, int]:
        """
        COMPLETE URL fetching with all features
        Returns: (url, content, status_code)
        """
        start_time = time.time()
        
        try:
            # Parse URL
            parsed = urlparse(url)
            domain = parsed.netloc
            
            if not domain:
                self.metrics.other_errors += 1
                return url, "", 0

            # Circuit breaker check
            if not self._check_circuit_breaker(domain):
                return url, "", 0

            # Rate limiting check
            if not self._check_rate_limit(domain):
                return url, "", 0

            # DNS check
            if not await self.check_dns(domain):
                self.metrics.dns_failures += 1
                self._record_failure(domain, "dns_failure")
                print_status(f"❌ DNS failed: {domain}", "red")
                return url, "", 0

            # Rate limiting delay
            await asyncio.sleep(self.delay)

            # Make request
            headers = self._get_enhanced_headers()
            print_status(f"🌐 Fetching: {domain}", "blue")

            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            
            async with self.session.get(
                url,
                timeout=timeout,
                headers=headers,
                ssl=False,
                allow_redirects=True
            ) as response:

                response_time = time.time() - start_time
                self.metrics.avg_response_time = (
                    (self.metrics.avg_response_time * self.metrics.urls_crawled + response_time) /
                    (self.metrics.urls_crawled + 1) if self.metrics.urls_crawled > 0 else response_time
                )

                # Handle 403 with bypass
                if response.status == 403:
                    self.metrics.blocked_403 += 1
                    print_status(f"🚫 403 Blocked: {domain}", "red")
                    
                    # Try bypass
                    if domain not in self.bypass_attempts_log or self.bypass_attempts_log.get(domain, 0) < 2:
                        print_status(f"🛡️ Attempting bypass for {domain}...", "yellow")
                        fetched_url, content, status_code = await self._try_403_bypass(url, domain)
                        self.bypass_attempts_log[domain] = self.bypass_attempts_log.get(domain, 0) + 1
                        
                        if status_code == 200:
                            self._record_success(domain)
                            self.metrics.urls_crawled += 1
                            self.metrics.bytes_downloaded += len(content)
                            print_status(f"✅ Bypass success: {domain}", "green")
                            return fetched_url, content, status_code

                    self.metrics.urls_failed += 1
                    self._record_failure(domain, "http_403")
                    return url, "", response.status

                # Handle successful responses
                if response.status == 200:
                    content = await response.text(errors='ignore')
                    
                    # Enhanced content validation
                    if not self._is_valid_content(content):
                        self.metrics.other_errors += 1
                        print_status(f"⚠️ Invalid content: {domain}", "yellow")
                        return url, "", response.status
                    
                    # Duplicate content check
                    if self._is_duplicate_content(content):
                        print_status(f"🔄 Duplicate: {domain}", "yellow")
                        return url, "", response.status
                    
                    # Success
                    self.metrics.urls_crawled += 1
                    self.metrics.bytes_downloaded += len(content)
                    self._record_success(domain)
                    print_status(f"✅ Success: {domain} ({response_time:.2f}s)", "green")
                    return url, content, response.status
                    
                # Handle redirects
                elif response.status in [301, 302, 307, 308]:
                    self.metrics.redirects_followed += 1
                    print_status(f"🔄 Redirect: {domain} → {response.status}", "yellow")
                    return url, "", response.status
                    
                # Handle other HTTP errors
                else:
                    self.metrics.http_errors += 1
                    self._record_failure(domain, f"http_{response.status}")
                    print_status(f"⚠️ HTTP {response.status}: {domain}", "yellow")
                    return url, "", response.status

        except asyncio.TimeoutError:
            self.metrics.timeouts += 1
            self._record_failure(domain, "timeout")
            print_status(f"⏰ Timeout: {domain}", "red")
            return url, "", 0
            
        except aiohttp.ClientConnectorError:
            self.metrics.connection_errors += 1
            self._record_failure(domain, "connection_error")
            print_status(f"🔌 Connection failed: {domain}", "red")
            return url, "", 0
            
        except aiohttp.ClientError as e:
            self.metrics.connection_errors += 1
            self._record_failure(domain, "client_error")
            print_status(f"🌐 Client error: {str(e)[:50]}", "red")
            return url, "", 0
            
        except Exception as e:
            self.metrics.other_errors += 1
            self._record_failure(domain, "unexpected_error")
            print_status(f"💥 Unexpected error: {str(e)[:50]}", "red")
            return url, "", 0

    async def parse_links(self, url: str, html: str) -> Tuple[Set[str], Set[str]]:
        """
        COMPLETE link parsing with all detection methods
        Returns: (links, js_links)
        """
        links = set()
        js_links = set()

        if not html:
            return links, js_links

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract regular links from <a> tags
            for a_tag in soup.find_all("a", href=True):
                try:
                    link = urljoin(url, a_tag['href'])
                    parsed = urlparse(link)
                    
                    if (parsed.scheme in ['http', 'https'] and 
                        parsed.netloc and 
                        self.dm.is_in_scope(link)):
                        links.add(link)
                        self.metrics.links_discovered += 1
                except Exception:
                    pass

            # Extract JavaScript files from <script> tags
            for script_tag in soup.find_all("script", src=True):
                try:
                    js_url = urljoin(url, script_tag['src'])
                    parsed = urlparse(js_url)
                    
                    if (parsed.scheme in ['http', 'https'] and 
                        parsed.netloc and 
                        self.dm.is_in_scope(js_url)):
                        js_links.add(js_url)
                        self.metrics.js_files_found += 1
                except Exception:
                    pass

            # Extract from <link> tags (CSS, icons, etc.)
            for link_tag in soup.find_all("link", href=True):
                try:
                    link_url = urljoin(url, link_tag['href'])
                    parsed = urlparse(link_url)
                    
                    if (parsed.scheme in ['http', 'https'] and 
                        parsed.netloc and 
                        self.dm.is_in_scope(link_url)):
                        links.add(link_url)
                        self.metrics.links_discovered += 1
                except Exception:
                    pass

            # Extract from <meta> tags
            for meta_tag in soup.find_all("meta", content=True):
                try:
                    content = meta_tag.get('content', '')
                    if content.startswith(('http://', 'https://')):
                        parsed = urlparse(content)
                        if (parsed.scheme in ['http', 'https'] and 
                            parsed.netloc and 
                            self.dm.is_in_scope(content)):
                            links.add(content)
                            self.metrics.links_discovered += 1
                except Exception:
                    pass

            # Extract from inline event handlers
            for tag in soup.find_all(True):  # All tags
                for attr in ['onclick', 'onload', 'onsubmit']:
                    if tag.has_attr(attr):
                        # Simple regex to find URLs in JavaScript
                        import re
                        js_content = tag[attr]
                        url_pattern = r"['\"](https?://[^'\"]+)['\"]"
                        found_urls = re.findall(url_pattern, js_content)
                        
                        for found_url in found_urls:
                            try:
                                full_url = urljoin(url, found_url)
                                parsed = urlparse(full_url)
                                
                                if (parsed.scheme in ['http', 'https'] and 
                                    parsed.netloc and 
                                    self.dm.is_in_scope(full_url)):
                                    links.add(full_url)
                                    self.metrics.links_discovered += 1
                            except Exception:
                                pass

        except Exception as e:
            self.logger.debug(f"Parse error for {url}: {e}")

        return links, js_links

    async def crawl_single_url(self, url: str, current_depth: int = 0) -> Tuple[Set[str], Set[str]]:
        """
        COMPLETE URL crawling with retry logic
        Returns: (links, js_links)
        """
        # Enhanced deduplication with retry tracking
        retry_count = self.retry_attempts.get(url, 0)
        
        if url in self.seen_urls and retry_count >= 2:
            return set(), set()

        if current_depth > self.max_depth:
            return set(), set()

        self.seen_urls.add(url)
        self.logger.debug(f"Crawling: {url} (depth: {current_depth}, retry: {retry_count})")

        # Fetch URL
        fetched_url, html, status_code = await self.fetch_url(url)

        # Retry logic for certain failures
        if not html and status_code in [0, 500, 502, 503] and retry_count < 2:
            self.retry_attempts[url] = retry_count + 1
            self.logger.debug(f"🔄 Retrying {url} (attempt {retry_count + 1})")
            await asyncio.sleep(1 * (retry_count + 1))  # Exponential backoff
            return await self.crawl_single_url(url, current_depth)

        # Only process successful responses
        if not html or status_code != 200:
            self.metrics.urls_failed += 1
            return set(), set()

        try:
            # Parse links
            links, js_links = await self.parse_links(fetched_url, html)

            # Track JS files
            new_js = js_links - self.discovered_js
            self.discovered_js.update(new_js)

            # Add new links to DomainManager
            for link in links:
                if link not in self.seen_urls:
                    self.dm.add_discovered(link, current_depth + 1)

            return links, new_js

        except Exception as e:
            self.metrics.other_errors += 1
            self.logger.debug(f"Error processing {url}: {e}")
            return set(), set()

    async def crawl(self) -> Tuple[List[str], List[str]]:
        """
        COMPLETE crawl method with all features
        Returns: (urls, js_files)
        """
        all_urls = set()
        all_js_files = set()

        self.metrics.start_time = time.time()

        # Initialize HTTP session with enhanced settings
        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            limit_per_host=5,
            ssl=False,
            use_dns_cache=True,
            ttl_dns_cache=300
        )

        self.session = aiohttp.ClientSession(connector=connector)

        try:
            batch_count = 0
            max_batches = 100
            consecutive_empty = 0
            max_consecutive_empty = 5

            print_status(f"🚀 Starting ENHANCED crawl with {self.concurrency} workers", "info")
            initial_stats = self.dm.get_stats()
            print_status(f"📊 Initial queue: {initial_stats['urls_queued']} URLs", "info")

            # Enhanced crawl loop
            while (self.dm.has_targets() and 
                   batch_count < max_batches and 
                   consecutive_empty < max_consecutive_empty):

                tasks = []
                targets_batch = []

                # Collect batch with enhanced logic
                batch_size = min(self.concurrency * 2, 30)
                for _ in range(batch_size):
                    url, depth = self.dm.get_next_target()
                    if not url:
                        break
                    targets_batch.append((url, depth))

                if not targets_batch:
                    consecutive_empty += 1
                    await asyncio.sleep(0.2)
                    continue

                consecutive_empty = 0
                batch_count += 1

                print_status(f"🔄 Batch {batch_count}: {len(targets_batch)} URLs", "debug")

                # Process batch with enhanced error handling
                for url, depth in targets_batch:
                    task = self.crawl_single_url(url, depth)
                    tasks.append(task)

                if tasks:
                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        successful_links = 0
                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.debug(f"Task failed: {result}")
                                continue

                            links, js_files = result
                            all_urls.update(links)
                            all_js_files.update(js_files)
                            successful_links += len(links)

                        self.logger.debug(f"Batch {batch_count}: {successful_links} links discovered")

                    except Exception as e:
                        self.logger.debug(f"Batch processing error: {e}")

                # Enhanced progress reporting
                elapsed = time.time() - self.metrics.start_time
                current_stats = self.dm.get_stats()
                crawl_stats = self.get_stats()
                
                progress_msg = (
                    f"📊 Progress: {crawl_stats['urls_crawled']} crawled, "
                    f"{crawl_stats['urls_failed']} failed, "
                    f"{current_stats['urls_queued']} remaining, "
                    f"{crawl_stats['blocked_403']} blocked"
                )
                print_status(progress_msg, "info")

                # Adaptive delay
                await asyncio.sleep(0.05)

            print_status(f"🏁 Crawling completed after {batch_count} batches", "success")

        except Exception as e:
            print_status(f"💥 Crawler critical error: {e}", "red")
            import traceback
            traceback.print_exc()

        finally:
            # Always close session
            if self.session:
                await self.session.close()
                self.logger.debug("HTTP session closed")

        # COMPREHENSIVE final statistics
        elapsed = time.time() - self.metrics.start_time
        stats = self.get_stats()

        print_status("📊 ENHANCED CRAWL COMPLETE", "success")
        print_status(f"   • URLs Crawled: {stats['urls_crawled']}", "info")
        print_status(f"   • URLs Failed: {stats['urls_failed']}", "info")
        print_status(f"   • JS Files Found: {len(self.discovered_js)}", "info")
        print_status(f"   • Links Discovered: {stats['links_discovered']}", "info")
        print_status(f"   • 403 Blocks: {stats['blocked_403']}", "info")
        print_status(f"   • Bypass Attempts: {stats['bypass_attempts']}", "info")
        print_status(f"   • Data Downloaded: {stats['bytes_downloaded'] / 1024 / 1024:.2f} MB", "info")
        print_status(f"   • Total Time: {elapsed:.2f}s", "info")
        print_status(f"   • Avg Response Time: {stats['avg_response_time']:.2f}s", "info")

        if elapsed > 0:
            print_status(f"   • Rate: {stats['urls_crawled'] / elapsed:.1f} URLs/sec", "info")

        return list(all_urls), list(all_js_files)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'total_urls_crawled': len(self.seen_urls),
            'total_js_discovered': len(self.discovered_js),
            'urls_crawled': self.metrics.urls_crawled,
            'urls_failed': self.metrics.urls_failed,
            'dns_failures': self.metrics.dns_failures,
            'timeouts': self.metrics.timeouts,
            'connection_errors': self.metrics.connection_errors,
            'http_errors': self.metrics.http_errors,
            'other_errors': self.metrics.other_errors,
            'blocked_403': self.metrics.blocked_403,
            'bypass_attempts': self.metrics.bypass_attempts,
            'bytes_downloaded': self.metrics.bytes_downloaded,
            'redirects_followed': self.metrics.redirects_followed,
            'js_files_found': self.metrics.js_files_found,
            'links_discovered': self.metrics.links_discovered,
            'avg_response_time': self.metrics.avg_response_time,
            'elapsed_time': time.time() - self.metrics.start_time if self.metrics.start_time else 0
        }

    def get_discovered_js(self) -> List[str]:
        """Get discovered JavaScript files"""
        return list(self.discovered_js)

    def reset(self):
        """Reset crawler for new scan"""
        self.seen_urls.clear()
        self.discovered_js.clear()
        self.content_hash_cache.clear()
        self.circuit_breaker.clear()
        self.rate_limit_tracker.clear()
        self.bypass_attempts_log.clear()
        self.retry_attempts.clear()
        self.metrics = CrawlMetrics()


# Backward compatibility
Crawler = CompleteCrawler
EnterpriseCrawler = CompleteCrawler