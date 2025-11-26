# modules/crawler.py
import asyncio
import aiohttp
import logging
import time
import socket
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Set, Optional, List, Tuple
from .utils import print_status


class Crawler:
    def __init__(self, domain_manager, concurrency: int = 20, max_depth: int = 5, delay: float = 0.1):
        self.dm = domain_manager
        self.seen_urls = set()
        self.discovered_js = set()
        self.concurrency = concurrency
        self.max_depth = max_depth
        self.delay = delay
        self.sem = asyncio.Semaphore(concurrency)
        self.session: Optional[aiohttp.ClientSession] = None
        self.crawl_stats = {
            'urls_crawled': 0,
            'urls_failed': 0,
            'dns_failures': 0,
            'timeouts': 0,
            'connection_errors': 0,
            'other_errors': 0,
            'start_time': None
        }

    async def check_dns(self, domain: str) -> bool:
        """Check if domain can be resolved via DNS"""
        try:
            # Try both IPv4 and IPv6
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

    async def fetch(self, session, url):
        """Fetch URL with comprehensive error handling"""
        async with self.sem:
            try:
                # Rate limiting
                await asyncio.sleep(self.delay)
                
                # Parse URL and check DNS first
                parsed = urlparse(url)
                domain = parsed.netloc
                
                # Skip if no domain
                if not domain:
                    self.crawl_stats['other_errors'] += 1
                    return url, "", 0
                
                # DNS resolution check
                if not await self.check_dns(domain):
                    self.crawl_stats['dns_failures'] += 1
                    print_status(f"🌐 DNS failed: {domain}", "debug")
                    return url, "", 0
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                print_status(f"🔍 Fetching: {url}", "debug")
                
                async with session.get(
                    url, 
                    timeout=aiohttp.ClientTimeout(total=10, connect=5),  # Shorter timeouts
                    headers=headers, 
                    ssl=False,
                    allow_redirects=True
                ) as resp:
                    
                    # Only process successful responses
                    if resp.status in [200, 201, 202]:
                        content_type = resp.headers.get('content-type', '').lower()
                        
                        # Only process HTML and JSON content
                        if any(ct in content_type for ct in ['text/html', 'application/json']):
                            text = await resp.text(errors='ignore')
                            self.crawl_stats['urls_crawled'] += 1
                            print_status(f"✅ Success: {url} - Status: {resp.status}", "debug")
                            return url, text, resp.status
                        else:
                            print_status(f"⚠️ Skipping non-HTML: {url} - {content_type[:50]}", "debug")
                    else:
                        print_status(f"⚠️ Non-200: {url} - Status: {resp.status}", "debug")
                    
                    self.crawl_stats['urls_failed'] += 1
                    return url, "", resp.status
                        
            except asyncio.TimeoutError:
                self.crawl_stats['timeouts'] += 1
                print_status(f"⏰ Timeout: {url}", "debug")
                return url, "", 0
                
            except aiohttp.ClientConnectorError as e:
                self.crawl_stats['connection_errors'] += 1
                # Don't log common connection errors to reduce noise
                if "Cannot connect to host" in str(e):
                    print_status(f"🌐 Connection failed: {parsed.netloc}", "debug")
                else:
                    print_status(f"🌐 Connection error: {str(e)[:80]}", "debug")
                return url, "", 0
                
            except aiohttp.ClientError as e:
                self.crawl_stats['connection_errors'] += 1
                print_status(f"🌐 Client error: {str(e)[:80]}", "debug")
                return url, "", 0
                
            except Exception as e:
                self.crawl_stats['other_errors'] += 1
                print_status(f"❌ Unexpected error: {str(e)[:80]}", "debug")
                return url, "", 0

    async def parse_links(self, url, html):
        """Parse links from HTML content with error handling"""
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
                    
                    # Filter valid HTTP/HTTPS links
                    if (parsed_link.scheme in ['http', 'https'] and 
                        parsed_link.netloc):
                        # Check if in scope using DomainManager
                        if self.dm.is_in_scope(link):
                            links.add(link)
                except Exception as e:
                    logging.debug(f"Error processing link {a['href']}: {e}")
            
            # Extract JS files from <script> tags
            for js in soup.find_all("script", src=True):
                try:
                    js_url = urljoin(url, js['src'])
                    parsed_js = urlparse(js_url)
                    
                    if (parsed_js.scheme in ['http', 'https'] and 
                        parsed_js.netloc and 
                        self.dm.is_in_scope(js_url)):
                        js_links.add(js_url)
                except Exception as e:
                    logging.debug(f"Error processing JS {js['src']}: {e}")
            
            # Extract from <link> tags (CSS, etc.)
            for link_tag in soup.find_all("link", href=True):
                try:
                    link_url = urljoin(url, link_tag['href'])
                    parsed_link = urlparse(link_url)
                    
                    if (parsed_link.scheme in ['http', 'https'] and 
                        parsed_link.netloc and 
                        self.dm.is_in_scope(link_url)):
                        links.add(link_url)
                except Exception as e:
                    logging.debug(f"Error processing link tag: {e}")
                    
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
                    logging.debug(f"Error processing meta tag: {e}")
                    
        except Exception as e:
            print_status(f"❌ Error parsing {url}: {e}", "debug")
            
        return links, js_links

    async def crawl_url(self, session, url, current_depth=0):
        """Crawl individual URL with comprehensive error handling"""
        # Skip if already seen or exceeded depth
        if url in self.seen_urls:
            return set(), set()
            
        if current_depth > self.max_depth:
            print_status(f"⏩ Max depth reached: {url}", "debug")
            return set(), set()
            
        self.seen_urls.add(url)
        print_status(f"🕷️ Crawling: {url} (depth: {current_depth})", "debug")
        
        fetched_url, html, status_code = await self.fetch(session, url)
        
        # Skip if fetch failed or no content
        if not html or status_code not in [200, 201, 202]:
            return set(), set()
            
        try:
            links, js_links = await self.parse_links(fetched_url, html)
            
            # Add new JS files to discovered set
            new_js = js_links - self.discovered_js
            self.discovered_js.update(new_js)
            
            # Add new links to DomainManager with incremented depth
            for link in links:
                if link not in self.seen_urls:
                    print_status(f"➕ New link: {link}", "debug")
                    self.dm.add_discovered(link, current_depth + 1)
            
            return links, new_js
            
        except Exception as e:
            print_status(f"❌ Error processing {url}: {e}", "debug")
            return set(), set()

    async def crawl(self):
        """Main crawl method with comprehensive error handling and progress tracking"""
        all_urls = set()
        all_js_files = set()
        
        self.crawl_stats['start_time'] = time.time()
        
        # Configure HTTP session with sensible limits
        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            limit_per_host=5,
            ttl_dns_cache=300,  # 5 minutes DNS cache
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=15)
        )
        
        try:
            batch_count = 0
            max_batches = 100  # Safety limit
            consecutive_empty_batches = 0
            max_consecutive_empty = 3  # Stop after 3 empty batches
            
            print_status(f"🚀 Starting crawl with {self.concurrency} concurrent workers", "info")
            
            while self.dm.has_targets() and batch_count < max_batches:
                tasks = []
                targets_batch = []
                
                # Collect batch of targets
                batch_size = min(self.concurrency, 20)  # Smaller batches for stability
                for _ in range(batch_size):
                    target, depth = self.dm.get_next_target()
                    if not target:
                        break
                    targets_batch.append((target, depth))
                
                # Check if we have any targets
                if not targets_batch:
                    consecutive_empty_batches += 1
                    if consecutive_empty_batches >= max_consecutive_empty:
                        print_status("📭 No more targets after consecutive empty batches", "info")
                        break
                    await asyncio.sleep(0.1)  # Small delay before checking again
                    continue
                
                # Reset consecutive empty counter since we found targets
                consecutive_empty_batches = 0
                
                batch_count += 1
                print_status(f"🔄 Batch {batch_count}: Processing {len(targets_batch)} URLs...", "debug")
                
                # Create tasks for batch
                for target, depth in targets_batch:
                    task = self.crawl_url(self.session, target, depth)
                    tasks.append(task)
                
                # Process batch with error handling
                if tasks:
                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                # Log the error but continue
                                logging.debug(f"Task failed: {result}")
                                continue
                                
                            links, js_files = result
                            all_urls.update(links)
                            all_js_files.update(js_files)
                            
                    except Exception as e:
                        print_status(f"❌ Batch processing error: {e}", "warning")
                
                # Progress reporting
                if batch_count % 10 == 0:
                    elapsed = time.time() - self.crawl_stats['start_time']
                    stats = self.get_stats()
                    print_status(
                        f"📊 Progress: {stats['urls_crawled']} succeeded, "
                        f"{stats['urls_failed']} failed in {elapsed:.1f}s", 
                        "info"
                    )
                
                # Small delay between batches to be respectful
                await asyncio.sleep(0.05)
                
            print_status(f"🏁 Crawling completed after {batch_count} batches", "success")
                
        except Exception as e:
            print_status(f"💥 Crawler critical error: {e}", "error")
            import traceback
            traceback.print_exc()
            
        finally:
            # Always close session
            if self.session:
                await self.session.close()
                print_status("🔒 HTTP session closed", "debug")
        
        # Final statistics
        elapsed = time.time() - self.crawl_stats['start_time']
        stats = self.get_stats()
        
        print_status(
            f"📊 Crawling completed: {stats['urls_crawled']} succeeded, "
            f"{stats['urls_failed']} failed in {elapsed:.2f}s", 
            "success"
        )
        
        # Detailed error breakdown
        if stats['urls_failed'] > 0:
            print_status("📈 Error breakdown:", "debug")
            print_status(f"   • DNS failures: {stats['dns_failures']}", "debug")
            print_status(f"   • Timeouts: {stats['timeouts']}", "debug")
            print_status(f"   • Connection errors: {stats['connection_errors']}", "debug")
            print_status(f"   • Other errors: {stats['other_errors']}", "debug")
        
        return all_urls, all_js_files

    def get_stats(self):
        """Get comprehensive crawling statistics"""
        return {
            'total_urls_crawled': len(self.seen_urls),
            'total_js_discovered': len(self.discovered_js),
            'urls_crawled': self.crawl_stats['urls_crawled'],
            'urls_failed': self.crawl_stats['urls_failed'],
            'dns_failures': self.crawl_stats['dns_failures'],
            'timeouts': self.crawl_stats['timeouts'],
            'connection_errors': self.crawl_stats['connection_errors'],
            'other_errors': self.crawl_stats['other_errors'],
            'elapsed_time': time.time() - self.crawl_stats['start_time'] if self.crawl_stats['start_time'] else 0
        }

    def get_discovered_js(self):
        """Get all discovered JavaScript files"""
        return list(self.discovered_js)

    def reset(self):
        """Reset crawler state for new scan"""
        self.seen_urls.clear()
        self.discovered_js.clear()
        self.crawl_stats = {
            'urls_crawled': 0,
            'urls_failed': 0,
            'dns_failures': 0,
            'timeouts': 0,
            'connection_errors': 0,
            'other_errors': 0,
            'start_time': None
        }
