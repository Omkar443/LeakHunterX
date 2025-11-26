"""
Enhanced HTTP Client with Industry-Grade Reliability
- Smart timeout management (4s connect / 10s read)
- Protocol fallback (HTTPS → HTTP)
- Targeted retry logic
- DNS validation
- Connection pooling
- Comprehensive error handling
"""

import aiohttp
import asyncio
import aiodns
import random
import time
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse
from .constants import USER_AGENTS
from .utils import print_status


class AsyncHTTPClient:
    """
    Enhanced HTTP client with intelligent error handling and reliability features
    """
    
    def __init__(self, timeout: float = 15.0, max_retries: int = 2, concurrency_limit: int = 20):
        # Smart timeouts: 4s connect, 10s read, 15s total
        self.timeout_config = aiohttp.ClientTimeout(
            total=timeout,
            connect=4.0,
            sock_read=10.0
        )
        self.max_retries = max_retries
        self.sem = asyncio.Semaphore(concurrency_limit)
        self.dns_resolver = None
        self.stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'dns_failures': 0,
            'timeouts': 0,
            'connection_errors': 0,
            'ssl_errors': 0,
            'protocol_fallbacks': 0,
            'retry_successes': 0
        }

    async def init_dns_resolver(self):
        """Initialize async DNS resolver"""
        try:
            self.dns_resolver = aiodns.DNSResolver()
        except Exception as e:
            print_status(f"⚠️ DNS resolver init failed: {e}, using fallback", "debug")
            self.dns_resolver = None

    async def validate_dns_async(self, domain: str) -> bool:
        """Async DNS validation with comprehensive error handling"""
        if not self.dns_resolver:
            # Fallback to socket-based DNS
            return await self._dns_fallback(domain)
            
        try:
            # Try both A and AAAA records
            await self.dns_resolver.query(domain, 'A')
            return True
        except aiodns.error.DNSError:
            try:
                await self.dns_resolver.query(domain, 'AAAA')
                return True
            except aiodns.error.DNSError:
                return False
        except Exception as e:
            print_status(f"⚠️ DNS query error for {domain}: {e}, using fallback", "debug")
            return await self._dns_fallback(domain)

    async def _dns_fallback(self, domain: str) -> bool:
        """Fallback DNS validation using socket"""
        try:
            loop = asyncio.get_event_loop()
            # Use thread pool executor to avoid blocking
            await loop.run_in_executor(None, self._socket_dns_lookup, domain)
            return True
        except Exception:
            return False

    def _socket_dns_lookup(self, domain: str):
        """Blocking DNS lookup for fallback"""
        import socket
        try:
            socket.getaddrinfo(domain, 443, socket.AF_INET)
            return True
        except socket.gaierror:
            try:
                socket.getaddrinfo(domain, 443, socket.AF_INET6)
                return True
            except socket.gaierror:
                return False
        except Exception:
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get randomized headers for requests"""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    async def _attempt_request(self, session: aiohttp.ClientSession, url: str) -> Tuple[Optional[str], int, str]:
        """
        Single request attempt with detailed error tracking
        Returns: (content, status_code, result_state)
        """
        try:
            async with session.get(
                url,
                timeout=self.timeout_config,
                ssl=False,
                allow_redirects=True
            ) as resp:
                
                content_type = resp.headers.get('content-type', '').lower()
                
                # Only return content for successful HTML/JSON responses
                if resp.status in [200, 201, 202]:
                    if any(ct in content_type for ct in ['text/html', 'application/json', 'text/plain']):
                        text = await resp.text(errors="ignore")
                        return text, resp.status, "success"
                    else:
                        return "", resp.status, "non_html_content"
                else:
                    return "", resp.status, f"http_error_{resp.status}"

        except asyncio.TimeoutError:
            return "", 0, "timeout"
        except aiohttp.ClientConnectorError as e:
            if "SSL" in str(e).upper():
                return "", 0, "ssl_error"
            return "", 0, "connection_error"
        except aiohttp.ServerDisconnectedError:
            return "", 0, "server_disconnected"
        except aiohttp.ClientOSError:
            return "", 0, "connection_reset"
        except aiohttp.TooManyRedirects:
            return "", 0, "too_many_redirects"
        except Exception as e:
            return "", 0, f"other_error:{str(e)[:50]}"

    async def fetch_with_fallback(self, url: str) -> Tuple[str, Optional[str], int, str]:
        """
        Enhanced fetch with protocol fallback and intelligent retries
        Returns: (original_url, content, status_code, final_state)
        """
        original_url = url
        self.stats['requests_made'] += 1

        # DNS validation first
        parsed = urlparse(url)
        if not await self.validate_dns_async(parsed.netloc):
            self.stats['dns_failures'] += 1
            return url, None, 0, "dns_failure"

        # Configure session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=3,
            verify_ssl=False,
            use_dns_cache=True
        )

        async with aiohttp.ClientSession(
            connector=connector,
            headers=self._get_headers()
        ) as session:

            # Main request attempt with retries
            for attempt in range(self.max_retries + 1):
                async with self.sem:  # Concurrency control
                    content, status, result_state = await self._attempt_request(session, url)

                    # Success case
                    if result_state == "success":
                        self.stats['successful_requests'] += 1
                        if attempt > 0:
                            self.stats['retry_successes'] += 1
                        return original_url, content, status, result_state

                    # Protocol fallback for SSL errors
                    if result_state == "ssl_error" and url.startswith('https://'):
                        http_url = url.replace('https://', 'http://')
                        print_status(f"🔁 SSL error, trying HTTP: {http_url}", "debug")
                        self.stats['protocol_fallbacks'] += 1
                        
                        # Try HTTP version
                        http_content, http_status, http_state = await self._attempt_request(session, http_url)
                        if http_state == "success":
                            self.stats['successful_requests'] += 1
                            return original_url, http_content, http_status, "success_fallback"
                        else:
                            # HTTP fallback also failed
                            break

                    # Retry logic for specific error types
                    retryable_errors = ["timeout", "connection_reset", "server_disconnected"]
                    if result_state in retryable_errors and attempt < self.max_retries:
                        print_status(f"🔄 Retry {attempt + 1} for {result_state}: {url}", "debug")
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue

                    # Non-retryable error or max retries reached
                    break

            # Update failure statistics
            self._update_error_stats(result_state)
            return original_url, None, status, result_state

    def _update_error_stats(self, result_state: str):
        """Update statistics based on error type"""
        if "timeout" in result_state:
            self.stats['timeouts'] += 1
        elif "connection" in result_state or "reset" in result_state or "disconnected" in result_state:
            self.stats['connection_errors'] += 1
        elif "ssl" in result_state:
            self.stats['ssl_errors'] += 1
        # Note: DNS failures are counted earlier

    async def fetch(self, url: str) -> Tuple[str, Optional[str], int]:
        """
        Main fetch method with backward compatibility
        Returns: (url, content, status_code)
        """
        if not hasattr(self, 'dns_resolver') or self.dns_resolver is None:
            await self.init_dns_resolver()

        original_url, content, status, state = await self.fetch_with_fallback(url)
        return original_url, content, status

    async def fetch_batch(self, urls: list) -> list:
        """
        Fetch multiple URLs concurrently with smart error handling
        """
        if not hasattr(self, 'dns_resolver') or self.dns_resolver is None:
            await self.init_dns_resolver()

        tasks = [self.fetch_with_fallback(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # Convert exceptions to error results
                processed_results.append(("", None, 0, f"exception:{str(result)[:50]}"))
            else:
                processed_results.append(result)
                
        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive request statistics"""
        success_rate = (self.stats['successful_requests'] / self.stats['requests_made'] * 100) if self.stats['requests_made'] > 0 else 0
        
        return {
            'requests_made': self.stats['requests_made'],
            'successful_requests': self.stats['successful_requests'],
            'success_rate_percent': round(success_rate, 1),
            'dns_failures': self.stats['dns_failures'],
            'timeouts': self.stats['timeouts'],
            'connection_errors': self.stats['connection_errors'],
            'ssl_errors': self.stats['ssl_errors'],
            'protocol_fallbacks': self.stats['protocol_fallbacks'],
            'retry_successes': self.stats['retry_successes']
        }

    def reset_stats(self):
        """Reset statistics for new scan"""
        self.stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'dns_failures': 0,
            'timeouts': 0,
            'connection_errors': 0,
            'ssl_errors': 0,
            'protocol_fallbacks': 0,
            'retry_successes': 0
        }

    async def close(self):
        """Cleanup resources"""
        if self.dns_resolver:
            # aiodns doesn't have explicit close in some versions
            pass


# Backward compatibility helper
async def create_http_client(timeout: float = 15.0, max_retries: int = 2) -> AsyncHTTPClient:
    """Factory function for creating HTTP client"""
    client = AsyncHTTPClient(timeout=timeout, max_retries=max_retries)
    await client.init_dns_resolver()
    return client
