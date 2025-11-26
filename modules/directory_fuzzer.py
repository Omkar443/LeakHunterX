"""
Smart Directory Fuzzer with Scoring Engine & Adaptive Fuzzing
- Priority-based target selection
- Adaptive fuzzing strategies
- Progressive depth escalation
- Resource-aware allocation
"""

import aiohttp
import asyncio
import re
import time
import aiodns
from urllib.parse import urljoin
from typing import List, Dict, Any, Optional, Set, Tuple
from .utils import print_status


class DirectoryFuzzer:
    """
    Smart Directory Fuzzer with Industry-Grade Intelligence
    Implements scoring engine and adaptive fuzzing strategies
    """

    def __init__(self, domain, wordlist_path="wordlists/default.txt", threads=15, aggressive=False):
        self.domain = domain
        self.wordlist_path = wordlist_path
        self.threads = min(threads, 20)  # Conservative thread limit
        self.aggressive = aggressive
        self.found_paths = []
        self.session = None
        self.dns_resolver = None

        # Enhanced scoring keywords with weights
        self.priority_keywords = {
            'CRITICAL': ['api', 'admin', 'secure', 'auth', 'vpn', 'ssh', 'root'],
            'HIGH': ['app', 'portal', 'dashboard', 'console', 'manager', 'control', 'system'],
            'MEDIUM': ['dev', 'test', 'staging', 'uat', 'demo', 'backup', 'old'],
            'LOW': ['mail', 'ftp', 'cdn', 'img', 'static', 'assets', 'media']
        }

        # Enhanced common directories for fuzzing
        self.common_dirs = [
            'admin', 'api', 'v1', 'v2', 'v3', 'internal', 'private', 'secret',
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

        # Statistics with enhanced tracking
        self.stats = {
            'requests_made': 0,
            'successful_responses': 0,
            'interesting_finds': 0,
            'errors': 0,
            'dns_failures': 0,
            'timeouts': 0,
            'start_time': 0,
            'end_time': 0,
            'targets_fuzzed': 0,
            'priority_breakdown': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        }

        # Visited URLs to avoid duplicates
        self.visited_urls = set()

    async def init_dns_resolver(self):
        """Initialize async DNS resolver"""
        try:
            self.dns_resolver = aiodns.DNSResolver()
        except Exception:
            self.dns_resolver = None

    async def validate_dns_async(self, domain: str) -> bool:
        """Async DNS validation"""
        if not self.dns_resolver:
            return True  # Fallback to skip DNS check
            
        try:
            await self.dns_resolver.query(domain, 'A')
            return True
        except aiodns.error.DNSError:
            return False
        except Exception:
            return True  # Fallback on other errors

    def calculate_subdomain_score(self, subdomain: str) -> Tuple[int, str]:
        """
        Calculate attack score for subdomain prioritization
        Returns: (score, priority_level)
        """
        score = 0
        subdomain_lower = subdomain.lower()
        
        # Keyword-based scoring (highest weight)
        for priority_level, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword in subdomain_lower:
                    if priority_level == 'CRITICAL':
                        score += 100
                    elif priority_level == 'HIGH':
                        score += 50
                    elif priority_level == 'MEDIUM':
                        score += 25
                    else:
                        score += 10
        
        # Length-based scoring (shorter = often more important)
        if len(subdomain.split('.')[0]) <= 4:
            score += 30
        elif len(subdomain.split('.')[0]) <= 6:
            score += 15
            
        # Main domain bonus
        if subdomain == self.domain:
            score += 200
            
        # Determine priority level
        if score >= 100:
            return score, 'CRITICAL'
        elif score >= 50:
            return score, 'HIGH'
        elif score >= 25:
            return score, 'MEDIUM'
        else:
            return score, 'LOW'

    def select_priority_targets(self, all_subdomains: List[str], max_targets: int = 15) -> List[str]:
        """
        Select highest priority targets for fuzzing
        """
        scored_targets = []
        
        for subdomain in all_subdomains:
            score, priority = self.calculate_subdomain_score(subdomain)
            scored_targets.append((subdomain, score, priority))
        
        # Sort by score descending
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N targets
        selected = [target for target, score, priority in scored_targets[:max_targets]]
        
        # Update priority breakdown
        for target, score, priority in scored_targets[:max_targets]:
            self.stats['priority_breakdown'][priority] += 1
            
        return selected

    def get_adaptive_wordlist_strategy(self, target: str, priority: str, wordlist_size: int) -> Dict[str, Any]:
        """
        Determine optimal wordlist strategy based on target priority and wordlist size
        """
        base_strategy = {
            'CRITICAL': {'sample_size': 200, 'timeout': 30, 'description': 'Full aggressive fuzzing'},
            'HIGH': {'sample_size': 100, 'timeout': 20, 'description': 'Normal fuzzing'},
            'MEDIUM': {'sample_size': 50, 'timeout': 15, 'description': 'Light fuzzing'},
            'LOW': {'sample_size': 0, 'timeout': 0, 'description': 'Skip fuzzing'}
        }
        
        strategy = base_strategy[priority].copy()
        
        # Adjust for wordlist size
        if wordlist_size > 1000:
            # Large wordlist - use sampling
            strategy['sample_size'] = min(strategy['sample_size'], wordlist_size // 10)
            strategy['description'] += f" (sampled from {wordlist_size})"
        elif wordlist_size < 50:
            # Small wordlist - use fully
            strategy['sample_size'] = wordlist_size
            strategy['description'] += f" (full wordlist: {wordlist_size})"
            
        return strategy

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
        extensions = ['', '.php', '.html', '.asp', '.aspx', '.jsp', '.py', '.rb', '.json']
        enhanced_paths = set()

        for path in base_paths:
            for ext in extensions:
                enhanced_paths.add('/' + path + ext)

        # Add common file patterns
        file_patterns = [
            '/robots.txt', '/sitemap.xml', '/.htaccess', '/.gitignore',
            '/crossdomain.xml', '/clientaccesspolicy.xml',
            '/web.config', '/config.php', '/settings.py',
            '/package.json', '/composer.json', '/.env'
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
            '/.env', '/.env.local', '/.env.production', '/.env.development',
            '/config/database.php', '/app/config.py', '/config.json',
            '/backup.sql', '/dump.sql', '/database.zip', '/backup.zip',
            '/error.log', '/access.log', '/debug.log', '/logs/error.log',
            '/.bash_history', '/.ssh/config', '/.aws/credentials',
            '/adminer.php', '/phpinfo.php', '/test.php', '/info.php',
            '/.git/config', '/.svn/entries', '/.hg/store', '/CVS/Entries'
        }

        return aggressive_paths

    async def check_path(self, session, url):
        """Enhanced path checking with better response analysis"""
        if url in self.visited_urls:
            return None

        self.visited_urls.add(url)
        self.stats['requests_made'] += 1

        try:
            # DNS validation for subdomain fuzzing
            parsed = urlparse(url)
            if not await self.validate_dns_async(parsed.netloc):
                self.stats['dns_failures'] += 1
                return None

            async with session.get(url, timeout=8, ssl=False, allow_redirects=True) as response:
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
                        'content_type': content_type,
                        'interesting': True
                    }

                    print_status(f"Found: {url} - Status: {response.status}", "success")
                    return result

                self.stats['successful_responses'] += 1
                return None

        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
        except aiohttp.ClientConnectorError:
            self.stats['errors'] += 1
        except Exception as e:
            self.stats['errors'] += 1

        return None

    def _is_interesting_response(self, status_code: int, content_length: int, content_type: str) -> bool:
        """Determine if response is interesting for security testing"""
        # Success responses with content
        if 200 <= status_code < 300:
            # Skip common static files
            if any(static_type in content_type for static_type in ['image/', 'font/', 'video/', 'audio/']):
                return False

            # Skip very small responses (likely empty)
            if content_length < 50:
                return False

            # Interesting content types
            interesting_types = ['text/html', 'application/json', 'text/plain', 'application/xml']
            if any(interesting_type in content_type for interesting_type in interesting_types):
                return True

        # Redirects and client errors
        elif status_code in [301, 302, 303, 307, 308, 401, 403, 500, 502, 503]:
            return True

        return False

    async def fuzz_single_target(self, target: str, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fuzz a single target with given strategy"""
        if strategy['sample_size'] == 0:
            return []

        print_status(f"    🎯 Fuzzing {target} - {strategy['description']}", "cyan")

        base_url = f"https://{target}"
        wordlist = await self.load_wordlist()
        
        # Apply sampling if needed
        if strategy['sample_size'] < len(wordlist):
            # Take most relevant paths for sampling
            sampled_wordlist = wordlist[:strategy['sample_size']]
        else:
            sampled_wordlist = wordlist

        found_paths = []
        
        # Enhanced connector configuration
        connector = aiohttp.TCPConnector(
            limit=min(10, self.threads),
            limit_per_host=3,
            verify_ssl=False
        )

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            # Process in smaller batches for stability
            batch_size = min(20, len(sampled_wordlist))
            
            for i in range(0, len(sampled_wordlist), batch_size):
                batch = sampled_wordlist[i:i + batch_size]
                tasks = []
                
                for path in batch:
                    url = urljoin(base_url, path)
                    task = self.check_path(session, url)
                    tasks.append(task)

                # Process batch with timeout
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=strategy['timeout']
                    )
                    
                    for result in batch_results:
                        if isinstance(result, dict) and result:
                            found_paths.append(result)
                            
                except asyncio.TimeoutError:
                    print_status(f"    ⏰ Fuzzing timeout for {target}", "yellow")
                    break

                # Brief pause between batches
                await asyncio.sleep(0.05)

        self.stats['targets_fuzzed'] += 1
        return found_paths

    async def fuzz_directories(self, base_url):
        """Enhanced directory fuzzing with smart strategies"""
        print_status(f"Starting directory fuzzing for {base_url}", "info")
        
        # For single domain fuzzing (backward compatibility)
        wordlist = await self.load_wordlist()
        return await self._fuzz_single_domain(base_url, wordlist)

    async def fuzz_multiple_targets(self, subdomains: List[str]) -> List[Dict[str, Any]]:
        """
        Smart fuzzing across multiple subdomains with priority-based allocation
        """
        await self.init_dns_resolver()
        
        print_status(f"🎯 Smart fuzzing for {len(subdomains)} subdomains", "info")
        
        # Select priority targets
        priority_targets = self.select_priority_targets(subdomains)
        
        print_status(f"📊 Selected {len(priority_targets)} priority targets:", "info")
        for i, target in enumerate(priority_targets[:5]):
            score, priority = self.calculate_subdomain_score(target)
            print_status(f"    {i+1}. {target} [{priority}] (score: {score})", "cyan")
        if len(priority_targets) > 5:
            print_status(f"    ... and {len(priority_targets) - 5} more targets", "dim")
        
        # Load wordlist once for efficiency
        wordlist = await self.load_wordlist()
        
        total_found_paths = []
        
        # Fuzz each priority target with appropriate strategy
        for i, target in enumerate(priority_targets):
            score, priority = self.calculate_subdomain_score(target)
            strategy = self.get_adaptive_wordlist_strategy(target, priority, len(wordlist))
            
            if strategy['sample_size'] > 0:
                found_paths = await self.fuzz_single_target(target, strategy)
                total_found_paths.extend(found_paths)
                
                # Brief pause between targets
                if i < len(priority_targets) - 1:
                    await asyncio.sleep(0.5)

        print_status(f"✅ Multi-target fuzzing completed: {len(total_found_paths)} paths found", "success")
        return total_found_paths

    async def _fuzz_single_domain(self, base_url: str, wordlist: List[str]) -> List[Dict[str, Any]]:
        """Fuzz single domain (backward compatibility)"""
        self.stats['start_time'] = time.time()
        
        connector = aiohttp.TCPConnector(limit=self.threads, verify_ssl=False)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            batch_size = min(30, self.threads * 2)

            for i in range(0, len(wordlist), batch_size):
                batch = wordlist[i:i + batch_size]
                tasks = [self.check_path(session, urljoin(base_url, path)) for path in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, dict) and result:
                        self.found_paths.append(result)

                await asyncio.sleep(0.1)

        self.stats['end_time'] = time.time()
        self._print_fuzzing_summary()
        return self.found_paths

    async def find_admin_panels(self, base_url):
        """Enhanced admin panel discovery"""
        print_status(f"Scanning for admin panels on {base_url}", "info")

        found_panels = []
        connector = aiohttp.TCPConnector(limit=8)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = [self.check_path(session, urljoin(base_url, panel)) for panel in self.admin_panels]
            results = await asyncio.gather(*tasks)
            found_panels = [r for r in results if r]

        # Enhanced admin panel detection
        enhanced_panels = await self._enhanced_admin_detection(base_url)
        found_panels.extend(enhanced_panels)

        print_status(f"Admin panel scan completed. Found {len(found_panels)} panels", "success")
        return found_panels

    async def _enhanced_admin_detection(self, base_url):
        """Enhanced admin panel detection"""
        additional_paths = [
            '/user', '/account', '/profile', '/settings',
            '/config', '/system', '/root', '/moderator',
            '/operator', '/webmaster', '/superuser'
        ]

        connector = aiohttp.TCPConnector(limit=5)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = [self._check_admin_indicator(session, urljoin(base_url, path)) for path in additional_paths]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r]

    async def _check_admin_indicator(self, session, url):
        """Check if URL shows admin panel indicators"""
        try:
            async with session.get(url, timeout=6, ssl=False) as response:
                if response.status in [200, 301, 302, 403]:
                    content = await response.text(errors='ignore')
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
        print_status(f"  Targets Fuzzed: {self.stats['targets_fuzzed']}", "info")
        print_status(f"  Success Rate: {success_rate:.1f}%", "info")
        print_status(f"  DNS Failures: {self.stats['dns_failures']}", "warning" if self.stats['dns_failures'] > 0 else "info")
        print_status(f"  Timeouts: {self.stats['timeouts']}", "warning" if self.stats['timeouts'] > 0 else "info")
        print_status(f"  Errors: {self.stats['errors']}", "warning" if self.stats['errors'] > 0 else "info")
        print_status(f"  Duration: {duration:.2f}s", "info")
        
        # Priority breakdown if available
        if any(count > 0 for count in self.stats['priority_breakdown'].values()):
            print_status("  Priority Breakdown:", "info")
            for priority, count in self.stats['priority_breakdown'].items():
                if count > 0:
                    print_status(f"    • {priority}: {count} targets", "info")

    def get_stats(self):
        """Enhanced statistics with more details"""
        duration = self.stats['end_time'] - self.stats['start_time'] if self.stats['end_time'] > 0 else 0

        stats = {
            'total_paths_found': len(self.found_paths),
            'admin_panels_found': len([p for p in self.found_paths if any(admin in p['url'] for admin in self.admin_panels)]),
            'total_requests': self.stats['requests_made'],
            'interesting_finds': self.stats['interesting_finds'],
            'targets_fuzzed': self.stats['targets_fuzzed'],
            'success_rate': round((self.stats['successful_responses'] / self.stats['requests_made'] * 100), 1) if self.stats['requests_made'] > 0 else 0,
            'duration_seconds': round(duration, 2),
            'dns_failures': self.stats['dns_failures'],
            'timeouts': self.stats['timeouts'],
            'errors': self.stats['errors']
        }
        
        # Add priority breakdown
        stats['priority_breakdown'] = self.stats['priority_breakdown'].copy()
        
        return stats
