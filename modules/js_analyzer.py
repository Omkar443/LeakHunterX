import re
import os
import logging
import aiohttp
import asyncio
import hashlib
from urllib.parse import urljoin
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

# Import utils and other modules - use absolute imports
try:
    from modules.utils import save_json, save_to_file, color_text, print_status
except ImportError:
    # Define minimal fallbacks if imports fail
    def save_json(path, data):
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_to_file(path, content):
        with open(path, 'w') as f:
            f.write(content)
    
    def color_text(text, color):
        return text
    
    def print_status(message, level="info"):
        print(f"[{level.upper()}] {message}")

try:
    from modules.js_extractor import JSExtractor
except ImportError:
    # Fallback: define a minimal JSExtractor if import fails
    class JSExtractor:
        def __init__(self, base_url=None):
            self.base_url = base_url
        
        async def extract(self, js_content):
            return []

try:
    from modules.leak_detector import LeakDetector
except ImportError:
    # Fallback: define a minimal LeakDetector if import fails
    class LeakDetector:
        def __init__(self, aggressive=True):
            pass
        
        def check_content(self, content):
            return []

@dataclass
class JSAnalysisResult:
    """Enterprise JS analysis result with comprehensive metadata"""
    js_url: str
    endpoints: List[str]
    secrets: List[Dict]
    content_hash: str
    file_size: int
    analysis_time: float
    confidence_score: float
    success: bool
    error: Optional[str] = None
    content: Optional[str] = None

class EnterpriseJSAnalyzer:
    """
    ENHANCED Enterprise-Grade JavaScript Analyzer with Content Download
    - Actually downloads and analyzes JS content
    - Proper duplicate detection
    - Real secret scanning with leak detector
    - Comprehensive endpoint extraction
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.leaks = {}
        self.endpoints = set()
        self.results_dir = f"results/{domain}"
        os.makedirs(self.results_dir, exist_ok=True)

        # Enterprise features
        self.content_cache = {}
        self.analysis_cache = {}
        self.circuit_breaker = {}
        self.performance_metrics = {
            'total_files_analyzed': 0,
            'total_endpoints_found': 0,
            'total_secrets_found': 0,
            'avg_analysis_time': 0,
            'cache_hits': 0,
            'content_downloaded': 0,
            'duplicates_skipped': 0
        }

        # ENHANCED: Integration with other modules
        self.js_extractor = JSExtractor()
        self.leak_detector = LeakDetector(aggressive=True)
        
        # Enhanced connection pooling
        self.session = None
        self.semaphore = asyncio.Semaphore(10)
        
        # Enhanced content deduplication
        self.processed_content_hashes = set()
        
        # FIXED: Initialize HTTP client for robust downloads
        self.http_client = None
        
        # Logging
        self.logger = self._setup_logger()

        # FIXED: Corrected regex patterns - fixed escape sequences
        self.secrets_regex = {
            "aws_access_key": {
                "pattern": r"AKIA[0-9A-Z]{16}",
                "confidence": 0.9,
                "validation": self._validate_aws_key
            },
            "aws_secret_key": {
                "pattern": r"[a-zA-Z0-9+/]{40}",
                "confidence": 0.7,
                "validation": self._validate_generic_secret
            },
            "google_api_key": {
                "pattern": r"AIza[0-9A-Za-z\-_]{35}",
                "confidence": 0.8,
                "validation": self._validate_google_key
            },
            "google_oauth": {
                "pattern": r"ya29\.[0-9A-Za-z\-_]+",
                "confidence": 0.8,
                "validation": self._validate_oauth_token
            },
            "firebase_url": {
                "pattern": r"[a-zA-Z0-9.-]+\.firebaseio\.com",
                "confidence": 0.6,
                "validation": self._validate_firebase_url
            },
            "slack_token": {
                "pattern": r"xox[baprs]-[0-9a-zA-Z]{10,48}",
                "confidence": 0.8,
                "validation": self._validate_slack_token
            },
            "stripe_key": {
                "pattern": r"(sk|pk)_(test|live)_[0-9a-zA-Z]{24,99}",
                "confidence": 0.9,
                "validation": self._validate_stripe_key
            },
            "github_token": {
                "pattern": r"gh[pousr]_[A-Za-z0-9_]{36,255}",
                "confidence": 0.8,
                "validation": self._validate_github_token
            },
            "jwt_token": {
                "pattern": r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*",
                "confidence": 0.7,
                "validation": self._validate_jwt_token
            },
            "generic_api_key": {
                "pattern": r"(api[_-]?key|secret[_-]?key|private[_-]?key)[\s\"':=]+([A-Za-z0-9+/=]{20,100})",
                "confidence": 0.5,
                "validation": self._validate_generic_secret
            },
            "database_url": {
                "pattern": r"(mongodb|postgres|mysql)://[a-zA-Z0-9.-]+:[0-9]+/[a-zA-Z0-9_-]+",
                "confidence": 0.8,
                "validation": self._validate_database_url
            },
            "encryption_key": {
                "pattern": r"(encryption[_-]?key|secret[_-]?key)[\s\"':=]+([A-Za-z0-9+/=]{20,100})",
                "confidence": 0.6,
                "validation": self._validate_generic_secret
            }
        }

        # FIXED: Corrected endpoint regex patterns
        self.endpoint_regex = [
            # API endpoints
            r'["\'](https?://[^"\'\\s<>]+)["\']',
            r'["\'](/api/[^"\'\\s<>]+)["\']',
            r'["\'](/v[0-9]+/[^"\'\\s<>]+)["\']',
            r'["\'](/graphql[^"\'\\s<>]*)["\']',
            r'["\'](/rest/[^"\'\\s<>]+)["\']',
            
            # Admin and internal endpoints
            r'["\'](/admin[^"\'\\s<>]*)["\']',
            r'["\'](/internal[^"\'\\s<>]*)["\']',
            r'["\'](/private[^"\'\\s<>]*)["\']',
            r'["\'](/secure[^"\'\\s<>]*)["\']',
            
            # Authentication endpoints
            r'["\'](/auth[^"\'\\s<>]*)["\']',
            r'["\'](/login[^"\'\\s<>]*)["\']',
            r'["\'](/oauth[^"\'\\s<>]*)["\']',
            
            # JavaScript fetch patterns
            r'fetch\(["\']([^"\'\\s]+)["\']',
            r'axios\.(get|post|put|delete)\(["\']([^"\'\\s]+)["\']',
            r'\.ajax\([^)]*url["\']?:["\']([^"\'\\s]+)',
            
            # Window location and navigation
            r'window\.location[^=]*=["\']([^"\'\\s]+)',
            r'document\.location[^=]*=["\']([^"\'\\s]+)',
            r'location\.href[^=]*=["\']([^"\'\\s]+)',
            
            # Dynamic imports and requires
            r'import\(["\']([^"\'\\s]+)["\']',
            r'require\(["\']([^"\'\\s]+)["\']',
            
            # WebSocket connections
            r'new WebSocket\(["\']([^"\'\\s]+)["\']',
            
            # Image and asset URLs
            r'["\'](/static/[^"\'\\s<>]+)["\']',
            r'["\'](/assets/[^"\'\\s<>]+)["\']',
            r'["\'](/images?/[^"\'\\s<>]+)["\']',
            
            # Enhanced patterns for modern JS frameworks
            r'router\.(get|post|put|delete)\(["\']([^"\'\\s]+)["\']',
            r'app\.(get|post|put|delete)\(["\']([^"\'\\s]+)["\']',
            r'route\(["\']([^"\'\\s]+)["\']',
            
            # XMLHttpRequest patterns
            r'\.open\(["\'](GET|POST|PUT|DELETE)["\'],["\']([^"\'\\s]+)["\']',
            r'xhr\.open\(["\'](GET|POST|PUT|DELETE)["\'],["\']([^"\'\\s]+)["\']',
            
            # Vue.js and React patterns
            r'this\.\$http\.(get|post|put|delete)\(["\']([^"\'\\s]+)["\']',
            r'axios\([^)]*url:\s*["\']([^"\'\\s]+)["\']',
            r'fetch\([^)]*["\']([^"\'\\s]+)["\']',
            
            # Configuration objects
            r'baseURL:\s*["\']([^"\'\\s]+)["\']',
            r'apiUrl:\s*["\']([^"\'\\s]+)["\']',
            r'endpoint:\s*["\']([^"\'\\s]+)["\']',
            
            # Webpack and module loading
            r'__webpack_require__\(["\']([^"\'\\s]+)["\']',
            r'import\(["\']([^"\'\\s]+)["\']',
            
            # Service worker and PWA patterns
            r'serviceWorker\.register\(["\']([^"\'\\s]+)["\']',
            r'workbox\.precaching\.precacheAndRoute\([^)]*["\']([^"\'\\s]+)["\']',
            
            # Comment-based endpoints (often overlooked)
            r'//\s*(?:endpoint|api|url):\s*([^\s]+)',
            r'/\*\s*(?:endpoint|api|url):\s*([^*]+)\*/',
        ]

    def _setup_logger(self):
        """Enterprise logging setup"""
        logger = logging.getLogger('js_analyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _generate_content_hash(self, content: str) -> str:
        """ENHANCED: Generate hash for content deduplication with normalization"""
        if not content:
            return ""
        # Normalize content to avoid false duplicates
        normalized = content.strip().replace('\r\n', '\n').replace('  ', ' ')
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _is_duplicate_content(self, content: str) -> bool:
        """ENHANCED: Check if content is duplicate with proper deduplication"""
        if not content:
            return True
            
        content_hash = self._generate_content_hash(content)
        if content_hash in self.processed_content_hashes:
            self.performance_metrics['duplicates_skipped'] += 1
            return True
            
        self.processed_content_hashes.add(content_hash)
        return False

    def _check_circuit_breaker(self, url: str) -> bool:
        """Circuit breaker for failing URLs"""
        if url in self.circuit_breaker:
            failures, last_attempt = self.circuit_breaker[url]
            if failures >= 3 and time.time() - last_attempt < 300:  # 5 min cooldown
                return False
        return True

    def _record_failure(self, url: str):
        """Record URL failure in circuit breaker"""
        if url not in self.circuit_breaker:
            self.circuit_breaker[url] = [1, time.time()]
        else:
            self.circuit_breaker[url][0] += 1
            self.circuit_breaker[url][1] = time.time()

    def _record_success(self, url: str):
        """Reset circuit breaker on success"""
        if url in self.circuit_breaker:
            del self.circuit_breaker[url]

    # Secret validation methods
    def _validate_aws_key(self, value: str) -> bool:
        """Validate AWS key format"""
        return value.startswith('AKIA') and len(value) == 20

    def _validate_google_key(self, value: str) -> bool:
        """Validate Google API key format"""
        return value.startswith('AIza') and len(value) >= 39

    def _validate_oauth_token(self, value: str) -> bool:
        """Validate OAuth token format"""
        return value.startswith('ya29.')

    def _validate_slack_token(self, value: str) -> bool:
        """Validate Slack token format"""
        return value.startswith(('xoxb-', 'xoxp-', 'xoxa-', 'xoxr-', 'xoxs-'))

    def _validate_stripe_key(self, value: str) -> bool:
        """Validate Stripe key format"""
        return value.startswith(('sk_', 'pk_'))

    def _validate_github_token(self, value: str) -> bool:
        """Validate GitHub token format"""
        return value.startswith(('ghp_', 'gho_', 'ghu_', 'ghs_', 'ghr_'))

    def _validate_jwt_token(self, value: str) -> bool:
        """Basic JWT token validation"""
        parts = value.split('.')
        return len(parts) == 3 and all(part for part in parts)

    def _validate_firebase_url(self, value: str) -> bool:
        """Validate Firebase URL"""
        return 'firebaseio.com' in value

    def _validate_database_url(self, value: str) -> bool:
        """Validate database URL format"""
        return any(db in value for db in ['mongodb://', 'postgres://', 'mysql://'])

    def _validate_generic_secret(self, value: str) -> bool:
        """Generic secret validation - check for high entropy"""
        if len(value) < 20:
            return False
        # Basic entropy check (simplified)
        unique_chars = len(set(value))
        return unique_chars > 10

    async def _get_js_content(self, js_url: str) -> Tuple[Optional[str], int]:
        """FIXED: Use AsyncHTTPClient instead of raw aiohttp for robust downloads"""
        cache_key = f"content_{hashlib.md5(js_url.encode()).hexdigest()}"
        
        # Check cache first
        if cache_key in self.content_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.content_cache[cache_key]
        
        if not self._check_circuit_breaker(js_url):
            self.logger.debug(f"Circuit breaker active for {js_url}")
            return None, 0

        try:
            # FIXED: Use AsyncHTTPClient instead of raw aiohttp
            if self.http_client is None:
                from modules.http_client import AsyncHTTPClient
                self.http_client = AsyncHTTPClient(timeout=15.0, max_retries=2)
                await self.http_client.init_dns_resolver()

            print_status(f"📥 Downloading JS content: {js_url}", "cyan")
            
            # Use the robust fetch_with_fallback method
            fetched_url, content, status_code, state = await self.http_client.fetch_with_fallback(js_url)
            
            if status_code == 200 and content:
                file_size = len(content.encode('utf-8'))
                
                # Detect binary-like responses
                sample = content[:200]
                try:
                    printable_ratio = sum(1 for c in sample if c.isprintable()) / max(1, len(sample))
                except Exception:
                    printable_ratio = 0

                if '\x00' in sample or printable_ratio < 0.7:
                    self._record_failure(js_url)
                    print_status(f"⚠️ Skipping binary/malformed JS content from {js_url}", "debug")
                    return None, 0

                # Track successful download
                self.performance_metrics['content_downloaded'] += 1

                # Cache the content
                self.content_cache[cache_key] = (content, file_size)
                self._record_success(js_url)

                print_status(f"✅ Downloaded {file_size} bytes from {js_url}", "green")
                return content, file_size
            else:
                self._record_failure(js_url)
                print_status(f"❌ Failed to download {js_url}: HTTP {status_code} - {state}", "red")
                return None, 0

        except Exception as e:
            self.logger.debug(f"Error fetching {js_url}: {e}")
            self._record_failure(js_url)
            print_status(f"❌ Error downloading {js_url}: {e}", "red")
            return None, 0

    async def extract_endpoints(self, js_url: str) -> List[str]:
        """ENHANCED: Endpoint extraction with actual content analysis"""
        start_time = time.time()
        
        try:
            print_status(f"      Analyzing: {js_url}", "cyan")

            # Check analysis cache
            cache_key = f"analysis_{hashlib.md5(js_url.encode()).hexdigest()}"
            if cache_key in self.analysis_cache:
                self.performance_metrics['cache_hits'] += 1
                return self.analysis_cache[cache_key]['endpoints']

            # ENHANCED: Actually download content
            content, file_size = await self._get_js_content(js_url)
            if not content:
                return []

            # ENHANCED: Check for duplicate content
            if self._is_duplicate_content(content):
                print_status(f"      🔄 Skipping duplicate content: {js_url}", "yellow")
                return []

            found_endpoints = set()

            # ENHANCED: Use JS extractor for comprehensive endpoint extraction
            try:
                extractor_endpoints = await self.js_extractor.extract(content)
                found_endpoints.update(extractor_endpoints)
            except Exception as e:
                self.logger.debug(f"JS extractor failed: {e}")

            # Enhanced endpoint extraction with context
            for pattern in self.endpoint_regex:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[-1]  # Take the last group

                    if match and isinstance(match, str):
                        # Convert relative URLs to absolute
                        if match.startswith('/'):
                            full_url = urljoin(js_url, match)
                            found_endpoints.add(full_url)
                        elif match.startswith(('http://', 'https://')):
                            # Validate URL format
                            if self._is_valid_url(match):
                                found_endpoints.add(match)
                        else:
                            # Handle relative paths without leading slash
                            full_url = urljoin(js_url, '/' + match.lstrip('/'))
                            found_endpoints.add(full_url)

            # ENHANCED: Save JS content for analysis
            js_filename = self._sanitize_filename(js_url.split('/')[-1] or f"js_{hash(js_url)}.js")
            js_save_path = os.path.join(self.results_dir, "js_files")
            os.makedirs(js_save_path, exist_ok=True)

            js_file_path = os.path.join(js_save_path, js_filename)
            with open(js_file_path, 'w', encoding='utf-8') as f:
                f.write(f"// Source: {js_url}\n")
                f.write(f"// Analysis Time: {time.ctime()}\n")
                f.write(f"// File Size: {file_size} bytes\n\n")
                f.write(content)

            endpoints_list = list(found_endpoints)
            
            # Cache the analysis results
            analysis_time = time.time() - start_time
            self.analysis_cache[cache_key] = {
                'endpoints': endpoints_list,
                'analysis_time': analysis_time,
                'file_size': file_size,
                'content': content  # Cache content for leak detection
            }

            if endpoints_list:
                print_status(f"        🔍 Found {len(endpoints_list)} endpoints", "green")

            self.performance_metrics['total_files_analyzed'] += 1
            self.performance_metrics['total_endpoints_found'] += len(endpoints_list)
            
            return endpoints_list

        except Exception as e:
            self.logger.debug(f"Error analyzing {js_url}: {e}")
            return []

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe saving"""
        # Remove or replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:100]  # Limit filename length

    def detect_secrets(self, content: str, js_url: str) -> List[Dict]:
        """ENHANCED: Secret detection using leak detector"""
        findings = []

        if not content:
            return findings

        try:
            # ENHANCED: Use leak detector for comprehensive secret scanning
            leak_findings = self.leak_detector.check_content(content)
            
            for leak in leak_findings:
                findings.append({
                    "type": leak.get("type", "unknown"),
                    "value": leak.get("value", "")[:100],  # Truncate for safety
                    "severity": leak.get("severity", "LOW"),
                    "confidence": leak.get("confidence", 0.0),
                    "context": leak.get("context", ""),
                    "url": js_url,
                    "validation_status": leak.get("validation_status", "unknown")
                })
                
                self.performance_metrics['total_secrets_found'] += 1

        except Exception as e:
            self.logger.debug(f"Leak detector error: {e}")
            # Fallback to regex-based detection
            findings.extend(self._detect_secrets_regex(content, js_url))

        return findings

    def _detect_secrets_regex(self, content: str, js_url: str) -> List[Dict]:
        """Fallback regex-based secret detection"""
        findings = []

        for secret_type, config in self.secrets_regex.items():
            try:
                pattern = config["pattern"]
                base_confidence = config["confidence"]
                validation_func = config["validation"]

                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[-1]  # Take the last group from tuple

                        secret_value = str(match)
                        
                        # Apply validation
                        is_valid = validation_func(secret_value)
                        confidence = base_confidence if is_valid else base_confidence * 0.5

                        findings.append({
                            "type": secret_type,
                            "value": secret_value[:100],
                            "url": js_url,
                            "confidence": confidence,
                            "validated": "format_valid" if is_valid else "format_suspicious",
                            "context": self._get_secret_context(content, secret_value)
                        })
                        
                        self.performance_metrics['total_secrets_found'] += 1

            except Exception as e:
                self.logger.debug(f"Regex error for {secret_type}: {e}")

        return findings

    def _get_secret_context(self, content: str, secret: str) -> str:
        """Extract context around the found secret"""
        try:
            index = content.find(secret)
            if index == -1:
                return ""

            # Get 50 characters before and after the secret
            start = max(0, index - 50)
            end = min(len(content), index + len(secret) + 50)
            context = content[start:end]
            
            # Clean up the context
            context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
            return context.strip()
            
        except Exception:
            return ""

    async def analyze_js_files(self, js_urls: List[str]) -> Tuple[List[str], List[Dict]]:
        """ENHANCED: JS file analysis with actual content processing"""
        all_endpoints = []
        all_leaks = []
        
        self.logger.info(f"Starting ENHANCED analysis of {len(js_urls)} JS files")

        # Process files in batches for better performance
        batch_size = 5
        for i in range(0, len(js_urls), batch_size):
            batch = js_urls[i:i + batch_size]
            self.logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch)} files")

            # Create tasks for current batch
            tasks = []
            for js_url in batch:
                task = self._analyze_single_js_file(js_url)
                tasks.append(task)

            # Process batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.debug(f"Batch analysis error: {result}")
                    continue
                    
                if result:
                    endpoints, leaks = result
                    all_endpoints.extend(endpoints)
                    all_leaks.extend(leaks)

            # Progress reporting
            processed = min(i + batch_size, len(js_urls))
            print_status(f"📊 JS Analysis Progress: {processed}/{len(js_urls)} files", "info")

        # Save comprehensive results
        leaks_data = {
            "domain": self.domain,
            "total_leaks_found": len(all_leaks),
            "leaks": all_leaks,
            "endpoints_found": len(all_endpoints),
            "performance_metrics": self.performance_metrics,
            "analysis_timestamp": time.time()
        }

        save_json(os.path.join(self.results_dir, "js_analysis.json"), leaks_data)

        # Save endpoints to file
        endpoints_file = os.path.join(self.results_dir, "endpoints.txt")
        with open(endpoints_file, 'w') as f:
            for endpoint in set(all_endpoints):
                f.write(f"{endpoint}\n")

        # Print enterprise summary
        self._print_analysis_summary(len(js_urls), len(all_endpoints), len(all_leaks))

        return list(set(all_endpoints)), all_leaks

    async def _analyze_single_js_file(self, js_url: str) -> Tuple[List[str], List[Dict]]:
        """ENHANCED: Analyze a single JS file with content download"""
        try:
            # Extract endpoints
            endpoints = await self.extract_endpoints(js_url)

            # Get content for secret detection
            content, file_size = await self._get_js_content(js_url)
            leaks = []
            if content:
                leaks = self.detect_secrets(content, js_url)

            return endpoints, leaks

        except Exception as e:
            self.logger.debug(f"Error processing {js_url}: {e}")
            return [], []

    async def analyze_js_file(self, js_url: str) -> JSAnalysisResult:
        """
        ENHANCED: Single JS file analysis with comprehensive results
        Main method called by main.py
        """
        start_time = time.time()
        
        try:
            print_status(f"🔍 Analyzing JS file: {js_url}", "cyan")
            
            # Download and analyze content
            content, file_size = await self._get_js_content(js_url)
            if not content:
                return JSAnalysisResult(
                    js_url=js_url,
                    endpoints=[],
                    secrets=[],
                    content_hash="",
                    file_size=0,
                    analysis_time=time.time() - start_time,
                    confidence_score=0.0,
                    success=False,
                    error="Failed to download content"
                )

            # Check for duplicate content
            content_hash = self._generate_content_hash(content)
            if self._is_duplicate_content(content):
                print_status(f"🔄 Skipping duplicate content: {js_url}", "yellow")
                return JSAnalysisResult(
                    js_url=js_url,
                    endpoints=[],
                    secrets=[],
                    content_hash=content_hash,
                    file_size=file_size,
                    analysis_time=time.time() - start_time,
                    confidence_score=0.0,
                    success=True,
                    content=content
                )

            # Extract endpoints
            endpoints = await self.extract_endpoints(js_url)
            
            # Detect secrets
            secrets = self.detect_secrets(content, js_url)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(endpoints, secrets, file_size)
            
            print_status(f"✅ Analyzed: {js_url} - {len(endpoints)} endpoints, {len(secrets)} secrets", "green")
            
            return JSAnalysisResult(
                js_url=js_url,
                endpoints=endpoints,
                secrets=secrets,
                content_hash=content_hash,
                file_size=file_size,
                analysis_time=time.time() - start_time,
                confidence_score=confidence,
                success=True,
                content=content
            )
            
        except Exception as e:
            print_status(f"❌ JS analysis failed for {js_url}: {e}", "red")
            return JSAnalysisResult(
                js_url=js_url,
                endpoints=[],
                secrets=[],
                content_hash="",
                file_size=0,
                analysis_time=time.time() - start_time,
                confidence_score=0.0,
                success=False,
                error=str(e)
            )

    def _calculate_confidence_score(self, endpoints: List[str], secrets: List[Dict], file_size: int) -> float:
        """Calculate confidence score for JS file importance"""
        score = 0.0
        
        # Endpoints contribute to score
        if endpoints:
            score += min(len(endpoints) * 0.1, 0.5)  # Max 0.5 for endpoints
        
        # Secrets contribute significantly
        if secrets:
            secret_confidence = sum(secret.get('confidence', 0) for secret in secrets)
            score += min(secret_confidence * 0.3, 0.8)  # Max 0.8 for secrets
        
        # File size indicates importance (larger = more complex)
        if file_size > 10000:  # Files > 10KB are likely important
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0

    def _print_analysis_summary(self, total_files: int, total_endpoints: int, total_leaks: int):
        """Print enterprise analysis summary"""
        print_status("\n📊 ENHANCED JS ANALYSIS SUMMARY", "cyan")
        print_status(f"   • Files Downloaded: {self.performance_metrics['content_downloaded']}", "info")
        print_status(f"   • Files Analyzed: {total_files}", "info")
        print_status(f"   • Endpoints Found: {total_endpoints}", "info")
        print_status(f"   • Secrets Detected: {total_leaks}", "info")
        print_status(f"   • Duplicates Skipped: {self.performance_metrics['duplicates_skipped']}", "info")
        print_status(f"   • Cache Efficiency: {self.performance_metrics['cache_hits']} hits", "info")
        
        if total_files > 0:
            success_rate = (self.performance_metrics['total_files_analyzed'] / total_files) * 100
            print_status(f"   • Success Rate: {success_rate:.1f}%", "info")

    def get_stats(self) -> Dict:
        """Get comprehensive analysis statistics"""
        return {
            "js_files_analyzed": self.performance_metrics['total_files_analyzed'],
            "endpoints_found": self.performance_metrics['total_endpoints_found'],
            "leaks_detected": self.performance_metrics['total_secrets_found'],
            "cache_hits": self.performance_metrics['cache_hits'],
            "avg_analysis_time": self.performance_metrics['avg_analysis_time'],
            "content_downloaded": self.performance_metrics['content_downloaded'],
            "duplicates_skipped": self.performance_metrics['duplicates_skipped'],
            "circuit_breaker_active": len(self.circuit_breaker)
        }

    async def close(self):
        """FIXED: Cleanup resources including HTTP client"""
        if self.http_client:
            await self.http_client.close()
        if self.session:
            await self.session.close()

# Backward compatibility - original class name
JSAnalyzer = EnterpriseJSAnalyzer