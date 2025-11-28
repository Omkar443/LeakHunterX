"""
Enhanced HTTP Client with Industry-Grade Reliability
- FIXED DNS resolution compatibility
- Smart timeout management (4s connect / 10s read)
- Protocol fallback (HTTPS → HTTP)
- Targeted retry logic
- DNS validation with multiple fallbacks
- Connection pooling
- Comprehensive error handling
"""

import aiohttp
import asyncio
import aiodns
import random
import time
import socket
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse
from .constants import USER_AGENTS
from .utils import print_status


class AsyncHTTPClient:
    """
    COMPLETELY FIXED: Enhanced HTTP client with robust DNS resolution
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
            'retry_successes': 0,
            'bypass_attempts': 0
        }

    async def init_dns_resolver(self):
        """Initialize async DNS resolver with fallback"""
        try:
            self.dns_resolver = aiodns.DNSResolver()
            # Test the resolver works
            await self.dns_resolver.query('google.com', 'A')
            print_status("✅ Async DNS resolver initialized", "debug")
        except Exception as e:
            print_status(f"⚠️ Async DNS resolver failed: {e}, using socket fallback", "debug")
            self.dns_resolver = None

    async def validate_dns_async(self, domain: str) -> bool:
        """
        COMPLETELY FIXED: Robust DNS validation with multiple fallback methods
        """
        # Method 1: Try async DNS resolver first
        if self.dns_resolver:
            try:
                await self.dns_resolver.query(domain, 'A')
                return True
            except aiodns.error.DNSError:
                try:
                    await self.dns_resolver.query(domain, 'AAAA')
                    return True
                except aiodns.error.DNSError:
                    pass
            except Exception:
                pass

        # Method 2: Socket-based DNS with proper async handling
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._socket_dns_lookup, domain)
            return result
        except Exception:
            pass

        # Method 3: Direct aiohttp DNS (final fallback)
        try:
            connector = aiohttp.TCPConnector(use_dns_cache=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(f"http://{domain}", timeout=5, ssl=False) as response:
                    return response.status is not None
        except Exception:
            pass

        return False

    def _socket_dns_lookup(self, domain: str) -> bool:
        """
        FIXED: Socket DNS lookup without timeout parameter issues
        """
        try:
            # Try IPv4
            socket.getaddrinfo(domain, 80, socket.AF_INET)
            return True
        except socket.gaierror:
            try:
                # Try IPv6
                socket.getaddrinfo(domain, 80, socket.AF_INET6)
                return True
            except socket.gaierror:
                try:
                    # Try with HTTPS port
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

    def _get_enhanced_headers(self) -> Dict[str, str]:
        """Get randomized headers with enhanced browser fingerprinting"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0',
        ]
        
        accept_languages = [
            'en-US,en;q=0.9',
            'en-GB,en;q=0.8',
            'en-CA,en;q=0.7',
        ]
        
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": random.choice(accept_languages),
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }

    async def _attempt_request(self, session: aiohttp.ClientSession, url: str, custom_headers: Dict = None) -> Tuple[Optional[str], int, str]:
        """
        Single request attempt with comprehensive error tracking
        Returns: (content, status_code, result_state)
        """
        headers = custom_headers or self._get_enhanced_headers()
        
        try:
            async with session.get(
                url,
                timeout=self.timeout_config,
                ssl=False,
                allow_redirects=True,
                headers=headers
            ) as resp:

                content_type = resp.headers.get('content-type', '').lower()

                # Handle successful responses
                if resp.status in [200, 201, 202]:
                    if any(ct in content_type for ct in ['text/html', 'application/json', 'text/plain', 'application/javascript']):
                        try:
                            text = await resp.text(errors="ignore")
                            # Basic content validation
                            if len(text) < 10:  # Too small, probably error page
                                return "", resp.status, "empty_content"
                            return text, resp.status, "success"
                        except Exception as e:
                            return "", resp.status, f"content_error:{str(e)[:30]}"
                    else:
                        return "", resp.status, "non_text_content"
                else:
                    return "", resp.status, f"http_error_{resp.status}"

        except asyncio.TimeoutError:
            return "", 0, "timeout"
        except aiohttp.ClientConnectorError as e:
            if "SSL" in str(e).upper():
                return "", 0, "ssl_error"
            elif "Connection refused" in str(e):
                return "", 0, "connection_refused"
            else:
                return "", 0, "connection_error"
        except aiohttp.ServerDisconnectedError:
            return "", 0, "server_disconnected"
        except aiohttp.ClientOSError:
            return "", 0, "connection_reset"
        except aiohttp.TooManyRedirects:
            return "", 0, "too_many_redirects"
        except aiohttp.ClientResponseError as e:
            return "", e.status, f"client_response_error_{e.status}"
        except Exception as e:
            return "", 0, f"unexpected_error:{str(e)[:50]}"

    async def _try_bypass_403(self, url: str, domain: str) -> Tuple[str, Optional[str], int, str]:
        """
        Enhanced 403 bypass with multiple strategies
        """
        original_url = url
        parsed = urlparse(url)
        
        # Strategy 1: Protocol fallback (HTTPS → HTTP)
        if parsed.scheme == 'https':
            http_url = url.replace('https://', 'http://')
            print_status(f"🔄 403 Bypass: Trying HTTP for {domain}", "debug")
            
            connector = aiohttp.TCPConnector(verify_ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                content, status, state = await self._attempt_request(session, http_url)
                if status == 200:
                    self.stats['bypass_attempts'] += 1
                    return original_url, content, status, "bypass_http_success"

        # Strategy 2: Header rotation
        bypass_headers = [
            self._get_enhanced_headers(),
            {
                'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            },
            {
                'User-Agent': 'Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
        ]
        
        for headers in bypass_headers:
            print_status(f"🔄 403 Bypass: Trying header rotation for {domain}", "debug")
            
            connector = aiohttp.TCPConnector(verify_ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                content, status, state = await self._attempt_request(session, url, headers)
                if status == 200:
                    self.stats['bypass_attempts'] += 1
                    return original_url, content, status, "bypass_headers_success"

        # Strategy 3: Path variations (for root domain)
        if parsed.path in ['', '/']:
            test_paths = ['/index.html', '/home', '/main', '/default']
            for path in test_paths:
                test_url = f"{parsed.scheme}://{parsed.netloc}{path}"
                print_status(f"🔄 403 Bypass: Trying path {path} for {domain}", "debug")
                
                connector = aiohttp.TCPConnector(verify_ssl=False)
                async with aiohttp.ClientSession(connector=connector) as session:
                    content, status, state = await self._attempt_request(session, test_url)
                    if status == 200:
                        self.stats['bypass_attempts'] += 1
                        return original_url, content, status, "bypass_path_success"

        return original_url, None, 403, "bypass_failed"

    async def fetch_with_fallback(self, url: str) -> Tuple[str, Optional[str], int, str]:
        """
        COMPLETELY FIXED: Enhanced fetch with comprehensive error handling and bypass
        Returns: (original_url, content, status_code, final_state)
        """
        original_url = url
        self.stats['requests_made'] += 1

        # DNS validation with detailed logging
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if not domain:
            self.stats['dns_failures'] += 1
            return url, None, 0, "invalid_domain"

        dns_works = await self.validate_dns_async(domain)
        if not dns_works:
            self.stats['dns_failures'] += 1
            print_status(f"❌ DNS failed for {domain}", "debug")
            return url, None, 0, "dns_failure"

        # Configure session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=3,
            verify_ssl=False,
            use_dns_cache=True,
            ttl_dns_cache=300
        )

        async with aiohttp.ClientSession(
            connector=connector,
            headers=self._get_enhanced_headers()
        ) as session:

            # Main request attempt with intelligent retries
            last_status = 0
            last_state = "unknown"
            
            for attempt in range(self.max_retries + 1):
                async with self.sem:  # Concurrency control
                    content, status, result_state = await self._attempt_request(session, url)
                    last_status = status
                    last_state = result_state

                    # Success case
                    if result_state == "success":
                        self.stats['successful_requests'] += 1
                        if attempt > 0:
                            self.stats['retry_successes'] += 1
                        print_status(f"✅ HTTP {status}: {domain}", "debug")
                        return original_url, content, status, result_state

                    # Handle 403 with enhanced bypass
                    if status == 403 and attempt == 0:  # Only try bypass on first attempt
                        print_status(f"🚫 403 detected for {domain}, attempting bypass...", "debug")
                        bypass_url, bypass_content, bypass_status, bypass_state = await self._try_bypass_403(url, domain)
                        
                        if bypass_status == 200:
                            self.stats['successful_requests'] += 1
                            self.stats['bypass_attempts'] += 1
                            print_status(f"✅ 403 Bypass successful for {domain}", "debug")
                            return original_url, bypass_content, bypass_status, bypass_state
                        else:
                            # Continue with normal flow if bypass failed
                            content, status, result_state = bypass_content, bypass_status, bypass_state

                    # Protocol fallback for SSL errors
                    if "ssl_error" in result_state and url.startswith('https://'):
                        http_url = url.replace('https://', 'http://')
                        print_status(f"🔁 SSL error, trying HTTP: {domain}", "debug")
                        self.stats['protocol_fallbacks'] += 1

                        # Try HTTP version
                        http_content, http_status, http_state = await self._attempt_request(session, http_url)
                        if http_state == "success":
                            self.stats['successful_requests'] += 1
                            return original_url, http_content, http_status, "success_fallback"
                        else:
                            # HTTP fallback also failed, update status
                            status, result_state = http_status, http_state

                    # Retry logic for transient errors
                    retryable_errors = ["timeout", "connection_reset", "server_disconnected", "connection_refused"]
                    if result_state in retryable_errors and attempt < self.max_retries:
                        retry_delay = 1.0 * (attempt + 1)  # Exponential backoff
                        print_status(f"🔄 Retry {attempt + 1} for {result_state}: {domain} (delay: {retry_delay}s)", "debug")
                        await asyncio.sleep(retry_delay)
                        continue

                    # Non-retryable error or max retries reached
                    break

            # Final error handling and statistics
            self._update_error_stats(last_state)
            
            # Provide meaningful final state
            if last_status > 0:
                final_state = f"http_{last_status}"
            else:
                final_state = last_state
                
            print_status(f"❌ Final failure for {domain}: {final_state}", "debug")
            return original_url, None, last_status, final_state

    def _update_error_stats(self, result_state: str):
        """Update statistics based on detailed error type"""
        error_mapping = {
            'timeout': 'timeouts',
            'connection_error': 'connection_errors', 
            'connection_refused': 'connection_errors',
            'connection_reset': 'connection_errors',
            'server_disconnected': 'connection_errors',
            'ssl_error': 'ssl_errors',
            'dns_failure': 'dns_failures'
        }
        
        for error_pattern, stat_key in error_mapping.items():
            if error_pattern in result_state:
                self.stats[stat_key] += 1
                return
                
        # Default for unclassified errors
        if 'http_error' in result_state:
            self.stats['connection_errors'] += 1
        else:
            self.stats['connection_errors'] += 1

    async def fetch(self, url: str) -> Tuple[str, Optional[str], int]:
        """
        Backward compatibility wrapper
        Returns: (url, content, status_code)
        """
        if not hasattr(self, 'dns_resolver') or self.dns_resolver is None:
            await self.init_dns_resolver()

        original_url, content, status, state = await self.fetch_with_fallback(url)
        return original_url, content, status

    async def fetch_batch(self, urls: list) -> list:
        """
        Fetch multiple URLs concurrently with comprehensive error handling
        """
        if not hasattr(self, 'dns_resolver') or self.dns_resolver is None:
            await self.init_dns_resolver()

        tasks = [self.fetch_with_fallback(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # Convert exceptions to consistent error format
                processed_results.append(("", None, 0, f"exception:{str(result)[:50]}"))
            else:
                processed_results.append(result)

        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive request statistics with enhanced metrics"""
        total_requests = self.stats['requests_made']
        successful_requests = self.stats['successful_requests']
        
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        return {
            'requests_made': total_requests,
            'successful_requests': successful_requests,
            'success_rate_percent': round(success_rate, 1),
            'dns_failures': self.stats['dns_failures'],
            'timeouts': self.stats['timeouts'],
            'connection_errors': self.stats['connection_errors'],
            'ssl_errors': self.stats['ssl_errors'],
            'protocol_fallbacks': self.stats['protocol_fallbacks'],
            'retry_successes': self.stats['retry_successes'],
            'bypass_attempts': self.stats['bypass_attempts'],
            'error_breakdown': {
                'dns_failures': self.stats['dns_failures'],
                'timeouts': self.stats['timeouts'],
                'connection_errors': self.stats['connection_errors'],
                'ssl_errors': self.stats['ssl_errors'],
                'http_errors': self.stats['connection_errors']  # Approximate
            }
        }

    def reset_stats(self):
        """Reset all statistics for new scan"""
        self.stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'dns_failures': 0,
            'timeouts': 0,
            'connection_errors': 0,
            'ssl_errors': 0,
            'protocol_fallbacks': 0,
            'retry_successes': 0,
            'bypass_attempts': 0
        }

    async def close(self):
        """Cleanup resources"""
        # aiodns resolver doesn't need explicit close in most versions
        # Connection cleanup is handled by aiohttp session context managers
        pass


# Factory function for backward compatibility
async def create_http_client(timeout: float = 15.0, max_retries: int = 2) -> AsyncHTTPClient:
    """Factory function for creating HTTP client with proper initialization"""
    client = AsyncHTTPClient(timeout=timeout, max_retries=max_retries)
    await client.init_dns_resolver()
    return client