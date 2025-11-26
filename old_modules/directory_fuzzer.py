# modules/directory_fuzzer.py
import aiohttp
import asyncio
import re
import time
from urllib.parse import urljoin
from typing import List, Dict, Any, Optional, Set
from .utils import print_status

class DirectoryFuzzer:
    """
    Enhanced Directory Fuzzer with Industry-Grade Features
    Maintains backward compatibility while adding robust functionality
    """
    
    def __init__(self, domain, wordlist_path="wordlists/default.txt", threads=20, aggressive=False):
        self.domain = domain
        self.wordlist_path = wordlist_path
        self.threads = min(threads, 25)  # Cap threads for safety
        self.aggressive = aggressive
        self.found_paths = []
        self.session = None

        # Enhanced common directories for fuzzing
        self.common_dirs = [
            'admin', 'api', 'v1', 'v2', 'internal', 'private', 'secret',
            'backup', 'old', 'test', 'dev', 'staging', 'uploads', 'assets',
            'js', 'css', 'images', 'doc', 'docs', 'documentation',
            'wp-admin', 'wp-content', 'wp-includes', 'phpmyadmin',
            '.git', '.svn', '.env', 'config', 'database', 'login',
            'auth', 'dashboard', 'console', 'manager', 'webadmin'
        ]

        # Enhanced admin panels and sensitive files
        self.admin_panels = [
            '/admin/', '/administrator/', '/wp-admin/', '/manager/',
            '/login/', '/signin/', '/dashboard/', '/controlpanel/',
            '/webadmin/', '/admincp/', '/cp/', '/panel/', '/backend/',
            '/system/', '/root/', '/moderator/', '/operator/'
        ]

        # Statistics
        self.stats = {
            'requests_made': 0,
            'successful_responses': 0,
            'interesting_finds': 0,
            'errors': 0,
            'start_time': 0,
            'end_time': 0
        }

        # Visited URLs to avoid duplicates
        self.visited_urls = set()

    async def load_wordlist(self):
        """Load and clean wordlist from file with enhanced filtering"""
        try:
            with open(self.wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
                wordlist = set()
                for line in f:
                    path = self._clean_path(line.strip())
                    if path and self._is_valid_path(path):
                        wordlist.add(path)
                
                if wordlist:
                    print_status(f"Loaded {len(wordlist)} clean paths from wordlist", "success")
                    return list(wordlist)
                else:
                    print_status("Wordlist empty or invalid, using enhanced common directories", "warning")
                    return self._get_enhanced_wordlist()
                    
        except FileNotFoundError:
            print_status(f"Wordlist not found: {self.wordlist_path}, using enhanced common directories", "warning")
            return self._get_enhanced_wordlist()
        except Exception as e:
            print_status(f"Error loading wordlist: {e}, using enhanced common directories", "error")
            return self._get_enhanced_wordlist()

    def _clean_path(self, path: str) -> str:
        """Clean and normalize path entry"""
        if not path or path.strip() == "":
            return ""
        
        # Remove comments and whitespace
        path = path.split('#')[0].strip()
        
        # Skip comment indicators
        if any(path.startswith(indicator) for indicator in ['#', '//', '/*', '*/', '--']):
            return ""
        
        # Normalize path format
        if not path.startswith('/'):
            path = '/' + path
        
        # Remove multiple slashes and normalize
        path = '/' + '/'.join(segment for segment in path.split('/') if segment)
        
        return path

    def _is_valid_path(self, path: str) -> bool:
        """Validate if path is safe for fuzzing"""
        if not path or len(path) > 200:
            return False
        
        # Skip dangerous patterns
        dangerous_patterns = [
            '..', '//', '\\',  # Path traversal
            '<', '>', '"', "'", # Injection attempts
            '{', '}', '[', ']', # Code blocks
            ' ', '\t', '\n', '\r' # Whitespace
        ]
        
        if any(pattern in path for pattern in dangerous_patterns):
            return False
        
        return True

    def _get_enhanced_wordlist(self) -> List[str]:
        """Get comprehensive enhanced wordlist"""
        base_paths = set(self.common_dirs)
        
        # Add file extensions
        extensions = ['', '.php', '.html', '.asp', '.aspx', '.jsp', '.py', '.rb']
        enhanced_paths = set()
        
        for path in base_paths:
            for ext in extensions:
                enhanced_paths.add('/' + path + ext)
        
        # Add common file patterns
        file_patterns = [
            '/robots.txt', '/sitemap.xml', '/.htaccess', '/.gitignore',
            '/crossdomain.xml', '/clientaccesspolicy.xml',
            '/web.config', '/config.php', '/settings.py',
            '/package.json', '/composer.json'
        ]
        
        enhanced_paths.update(file_patterns)
        
        # Add aggressive paths if enabled
        if self.aggressive:
            aggressive_paths = self._get_aggressive_paths()
            enhanced_paths.update(aggressive_paths)
        
        return list(enhanced_paths)

    def _get_aggressive_paths(self) -> Set[str]:
        """Get aggressive scanning paths"""
        aggressive_paths = {
            '/.env', '/.env.local', '/.env.production',
            '/config/database.php', '/app/config.py',
            '/backup.sql', '/dump.sql', '/database.zip',
            '/error.log', '/access.log', '/debug.log',
            '/.bash_history', '/.ssh/config', '/.aws/credentials',
            '/adminer.php', '/phpinfo.php', '/test.php',
            '/.git/config', '/.svn/entries', '/.hg/store'
        }
        
        return aggressive_paths

    async def check_path(self, session, url):
        """Enhanced path checking with better response analysis"""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        self.stats['requests_made'] += 1
        
        try:
            async with session.get(url, timeout=10, ssl=False, allow_redirects=True) as response:
                content_length = response.headers.get('content-length', 0)
                content_type = response.headers.get('content-type', '').split(';')[0]
                
                # Determine if this is interesting
                is_interesting = self._is_interesting_response(
                    response.status, int(content_length) if content_length else 0, content_type
                )
                
                if is_interesting:
                    self.stats['interesting_finds'] += 1
                    result = {
                        'url': url,
                        'status': response.status,
                        'content_length': content_length,
                        'interesting': True
                    }
                    
                    print_status(f"Found: {url} - Status: {response.status}", "success")
                    return result
                
                self.stats['successful_responses'] += 1
                return None
                
        except asyncio.TimeoutError:
            self.stats['errors'] += 1
        except aiohttp.ClientError as e:
            self.stats['errors'] += 1
        except Exception as e:
            self.stats['errors'] += 1
            print_status(f"Error checking {url}: {e}", "debug")
        
        return None

    def _is_interesting_response(self, status_code: int, content_length: int, content_type: str) -> bool:
        """Determine if response is interesting for security testing"""
        # Success responses with content
        if 200 <= status_code < 300:
            # Skip common static files
            if any(static_type in content_type for static_type in ['image/', 'font/', 'video/', 'audio/']):
                return False
            
            # Skip very small responses (likely empty)
            if content_length < 100:
                return False
            
            # Interesting content types
            interesting_types = ['text/html', 'application/json', 'text/plain', 'application/xml']
            if any(interesting_type in content_type for interesting_type in interesting_types):
                return True
        
        # Redirects and client errors
        elif status_code in [301, 302, 303, 307, 308, 401, 403, 500, 502, 503]:
            return True
        
        return False

    async def fuzz_directories(self, base_url):
        """Enhanced directory fuzzing with better performance"""
        print_status(f"Starting directory fuzzing for {base_url}", "info")

        wordlist = await self.load_wordlist()
        self.stats['start_time'] = time.time()
        
        # Enhanced connector configuration
        connector = aiohttp.TCPConnector(
            limit=self.threads,
            limit_per_host=5,
            verify_ssl=False,
            use_dns_cache=True
        )

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            # Process in batches for better performance
            batch_size = min(50, self.threads * 3)
            
            for i in range(0, len(wordlist), batch_size):
                batch = wordlist[i:i + batch_size]
                
                # Show progress
                progress = f"Batch {i//batch_size + 1}/{(len(wordlist)-1)//batch_size + 1}"
                print_status(f"  {progress}: Testing {len(batch)} paths...", "debug")
                
                tasks = []
                for path in batch:
                    url = urljoin(base_url, path)
                    task = self.check_path(session, url)
                    tasks.append(task)
                
                # Process batch with limited concurrency
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process valid results
                for result in batch_results:
                    if isinstance(result, dict) and result:
                        self.found_paths.append(result)
                
                # Brief pause between batches
                await asyncio.sleep(0.1)

        self.stats['end_time'] = time.time()
        self._print_fuzzing_summary()
        
        print_status(f"Directory fuzzing completed. Found {len(self.found_paths)} interesting paths", "success")
        return self.found_paths

    async def fuzz_subdomains(self, domain):
        """Enhanced subdomain fuzzing (maintained for compatibility)"""
        print_status(f"Starting subdomain fuzzing for {domain}", "info")

        wordlist = await self.load_wordlist()
        found_subdomains = []
        
        connector = aiohttp.TCPConnector(limit=self.threads)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = []

            for sub in wordlist[:100]:  # Limit for subdomain fuzzing
                subdomain = f"{sub}.{domain}"
                url = f"https://{subdomain}"
                task = self.check_path(session, url)
                tasks.append(task)

                if len(tasks) >= self.threads:
                    results = await asyncio.gather(*tasks)
                    found_subdomains.extend([r for r in results if r])
                    tasks = []

            if tasks:
                results = await asyncio.gather(*tasks)
                found_subdomains.extend([r for r in results if r])

        print_status(f"Subdomain fuzzing completed. Found {len(found_subdomains)} subdomains", "success")
        return found_subdomains

    async def find_admin_panels(self, base_url):
        """Enhanced admin panel discovery"""
        print_status(f"Scanning for admin panels on {base_url}", "info")

        found_panels = []
        connector = aiohttp.TCPConnector(limit=10)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = []

            for panel in self.admin_panels:
                url = urljoin(base_url, panel)
                task = self.check_path(session, url)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            found_panels = [r for r in results if r]

        # Enhanced admin panel detection
        enhanced_panels = await self._enhanced_admin_detection(base_url, session)
        found_panels.extend(enhanced_panels)

        print_status(f"Admin panel scan completed. Found {len(found_panels)} panels", "success")
        return found_panels

    async def _enhanced_admin_detection(self, base_url, session):
        """Enhanced admin panel detection with content analysis"""
        additional_paths = [
            '/user', '/account', '/profile', '/settings',
            '/config', '/system', '/root', '/moderator',
            '/operator', '/webmaster', '/superuser'
        ]
        
        found_panels = []
        tasks = []
        
        for path in additional_paths:
            url = urljoin(base_url, path)
            task = self._check_admin_indicator(session, url)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        found_panels = [r for r in results if r]
        
        return found_panels

    async def _check_admin_indicator(self, session, url):
        """Check if URL shows admin panel indicators"""
        try:
            async with session.get(url, timeout=8, ssl=False) as response:
                if response.status in [200, 301, 302, 403]:
                    content = await response.text(errors='ignore')
                    
                    # Check for admin indicators in content
                    admin_indicators = ['login', 'password', 'admin', 'dashboard', 'username']
                    content_lower = content.lower()
                    
                    matches = sum(1 for indicator in admin_indicators if indicator in content_lower)
                    if matches >= 2:
                        return {
                            'url': url,
                            'status': response.status,
                            'content_length': len(content),
                            'admin_indicators': matches
                        }
        
        except Exception:
            pass
        
        return None

    def _print_fuzzing_summary(self):
        """Print comprehensive fuzzing summary"""
        duration = self.stats['end_time'] - self.stats['start_time']
        success_rate = (self.stats['successful_responses'] / self.stats['requests_made'] * 100) if self.stats['requests_made'] > 0 else 0
        
        print_status("📊 Directory Fuzzing Summary:", "info")
        print_status(f"  Total Requests: {self.stats['requests_made']}", "info")
        print_status(f"  Interesting Finds: {self.stats['interesting_finds']}", "success")
        print_status(f"  Success Rate: {success_rate:.1f}%", "info")
        print_status(f"  Errors: {self.stats['errors']}", "warning" if self.stats['errors'] > 0 else "info")
        print_status(f"  Duration: {duration:.2f}s", "info")

    def get_stats(self):
        """Enhanced statistics with more details"""
        duration = self.stats['end_time'] - self.stats['start_time'] if self.stats['end_time'] > 0 else 0
        
        return {
            'total_paths_found': len(self.found_paths),
            'admin_panels_found': len([p for p in self.found_paths if any(admin in p['url'] for admin in self.admin_panels)]),
            'total_requests': self.stats['requests_made'],
            'interesting_finds': self.stats['interesting_finds'],
            'success_rate': round((self.stats['successful_responses'] / self.stats['requests_made'] * 100), 1) if self.stats['requests_made'] > 0 else 0,
            'duration_seconds': round(duration, 2),
            'errors': self.stats['errors']
        }
