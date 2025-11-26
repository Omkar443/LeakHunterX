import re
import sys
import importlib
from urllib.parse import urljoin

# Import utils - use absolute import since we're running from root
try:
    from modules.utils import is_valid_url, normalize_url
except ImportError:
    # Fallback if absolute import fails
    try:
        from utils import is_valid_url, normalize_url
    except ImportError:
        # Final fallback: define minimal versions
        def is_valid_url(url):
            try:
                from urllib.parse import urlparse
                result = urlparse(url)
                return all([result.scheme, result.netloc])
            except:
                return False
        
        def normalize_url(url):
            return url.rstrip('/')

class JSExtractor:
    """
    Extracts endpoints and possibly sensitive URLs inside JavaScript code
    using enhanced Regex patterns for comprehensive detection.
    """

    # Main URL regex pattern
    URL_REGEX = re.compile(
        r"""
        (?:"|')                                  # Start quotes
        (
            (?:\/|https?:\/\/)                   # Relative or absolute
            [^"'\s]+?                            # URL body
            (?:\?.*?)?                           # Optional query
        )
        (?:"|')                                  # End quotes
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    # Enhanced patterns for comprehensive JavaScript URL extraction
    ENHANCED_PATTERNS = [
        # API endpoints
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
        
        # Template literals and backticks
        r'`([^`\s]+)`',
        r'\$\{([^}]+)\}',
        
        # JSON-like structures
        r'["\']url["\']\s*:\s*["\']([^"\'\\s]+)["\']',
        r'["\']endpoint["\']\s*:\s*["\']([^"\'\\s]+)["\']',
        r'["\']path["\']\s*:\s*["\']([^"\'\\s]+)["\']',
    ]

    API_HINTS = [
        "api", "auth", "v1", "v2", "v3",
        "token", "secret", "jwt", "key",
        "login", "user", "admin", "auth",
        "oauth", "verify", "validate",
        "password", "reset", "register",
        "endpoint", "url", "baseurl",
        "graphql", "rest", "websocket",
        "webhook", "callback", "redirect"
    ]

    def __init__(self, base_url=None):
        self.base_url = base_url
        # Compile all enhanced patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ENHANCED_PATTERNS]

    async def extract(self, js_content: str):
        """
        Extract URLs from JavaScript content using comprehensive regex patterns
        """
        if not js_content:
            return []

        urls = set()

        # Basic URL regex extraction
        regex_urls = re.findall(self.URL_REGEX, js_content)
        for u in regex_urls:
            full = self._process_url(u)
            if full:
                urls.add(full)

        # Enhanced pattern extraction
        for pattern in self.compiled_patterns:
            matches = pattern.findall(js_content)
            for match in matches:
                url_candidate = self._extract_url_from_match(match)
                full_url = self._process_url(url_candidate)
                if full_url:
                    urls.add(full_url)

        # Context-based extraction for API hints
        urls.update(self._extract_with_context_hints(js_content))

        # Deep pattern extraction for complex cases
        urls.update(self._extract_complex_patterns(js_content))

        return sorted(urls)

    def _process_url(self, url_candidate):
        """Process and validate a URL candidate"""
        if not url_candidate or not isinstance(url_candidate, str):
            return None
            
        # Clean the URL
        url_candidate = url_candidate.strip()
        if not url_candidate:
            return None
            
        full = urljoin(self.base_url, url_candidate) if self.base_url else url_candidate
        full = normalize_url(full)
        if is_valid_url(full):
            return full
        return None

    def _extract_url_from_match(self, match):
        """Extract URL string from regex match (handles tuples)"""
        if isinstance(match, tuple):
            # Take the last non-empty group from tuple matches
            for item in reversed(match):
                if item and isinstance(item, str) and item.strip():
                    return item.strip()
            return match[0] if match else ""
        return match

    def _extract_with_context_hints(self, js_content: str):
        """
        Extract URLs that appear in context with API-related keywords
        """
        api_urls = set()
        
        # Find lines containing API hints
        lines = js_content.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for hint in self.API_HINTS:
                if hint in line_lower:
                    # Extract URLs from lines with API hints
                    url_matches = re.findall(self.URL_REGEX, line)
                    for url_match in url_matches:
                        full_url = self._process_url(url_match)
                        if full_url:
                            api_urls.add(full_url)
                    
                    # Also check enhanced patterns on these lines
                    for pattern in self.compiled_patterns:
                        matches = pattern.findall(line)
                        for match in matches:
                            url_candidate = self._extract_url_from_match(match)
                            full_url = self._process_url(url_candidate)
                            if full_url:
                                api_urls.add(full_url)
        
        return api_urls

    def _extract_complex_patterns(self, js_content: str):
        """
        Extract URLs from complex JavaScript patterns and structures
        """
        complex_urls = set()
        
        # Extract from object assignments
        object_patterns = [
            r'(?:const|let|var)\s+\w+\s*=\s*["\']([^"\'\\s]+)["\']',
            r'\w+\.(?:url|endpoint|api|path)\s*=\s*["\']([^"\'\\s]+)["\']',
            r'(?:url|endpoint|api):\s*["\']([^"\'\\s]+)["\']',
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, js_content, re.IGNORECASE)
            for match in matches:
                full_url = self._process_url(match)
                if full_url:
                    complex_urls.add(full_url)
        
        # Extract from function calls and parameters
        function_patterns = [
            r'\.(?:get|post|put|delete|patch)\(["\']([^"\'\\s]+)["\']',
            r'\.request\([^)]*["\']([^"\'\\s]+)["\']',
            r'\.(?:load|open)\([^)]*["\']([^"\'\\s]+)["\']',
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, js_content, re.IGNORECASE)
            for match in matches:
                full_url = self._process_url(match)
                if full_url:
                    complex_urls.add(full_url)
        
        return complex_urls

    async def extract_detailed(self, js_content: str):
        """
        Extract URLs with additional context and metadata
        """
        if not js_content:
            return []

        detailed_results = []
        
        # Use the main extraction method
        urls = await self.extract(js_content)
        
        for url in urls:
            # Get context around the URL
            context = self._get_url_context(js_content, url)
            
            detailed_results.append({
                'url': url,
                'type': self._classify_url(url),
                'confidence': self._calculate_confidence(url, context),
                'context': context
            })

        return detailed_results

    def _get_url_context(self, content: str, url: str, context_size: int = 100):
        """
        Extract context around the found URL
        """
        try:
            index = content.find(url)
            if index == -1:
                return ""

            start = max(0, index - context_size)
            end = min(len(content), index + len(url) + context_size)
            context = content[start:end]
            
            # Clean up the context
            context = re.sub(r'\s+', ' ', context)
            return context.strip()
            
        except Exception:
            return ""

    def _classify_url(self, url: str):
        """
        Classify URL type based on patterns
        """
        url_lower = url.lower()
        
        if any(api_hint in url_lower for api_hint in ['/api/', 'api.', 'v1/', 'v2/', 'v3/', 'graphql']):
            return 'api_endpoint'
        elif any(auth_hint in url_lower for auth_hint in ['auth', 'login', 'token', 'oauth', 'jwt', 'session']):
            return 'auth_endpoint'
        elif any(admin_hint in url_lower for admin_hint in ['admin', 'dashboard', 'manage', 'private', 'internal']):
            return 'admin_endpoint'
        elif any(static_hint in url_lower for static_hint in ['static', 'assets', 'images', 'css', 'js', '.png', '.jpg', '.svg', '.ico']):
            return 'static_resource'
        elif any(ws_hint in url_lower for ws_hint in ['ws://', 'wss://', 'websocket']):
            return 'websocket'
        elif any(webhook_hint in url_lower for webhook_hint in ['webhook', 'callback', 'hook']):
            return 'webhook'
        else:
            return 'general_endpoint'

    def _calculate_confidence(self, url: str, context: str):
        """
        Calculate confidence score for URL importance
        """
        confidence = 0.5  # Base confidence
        
        # URL pattern boosts
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in ['/api/', 'auth', 'token', 'secret', 'admin']):
            confidence += 0.3
        if any(pattern in url_lower for pattern in ['graphql', 'rest', 'v1', 'v2', 'v3']):
            confidence += 0.2
        if any(pattern in url_lower for pattern in ['webhook', 'callback', 'websocket']):
            confidence += 0.2
            
        # Context boosts
        context_lower = context.lower()
        if any(keyword in context_lower for keyword in ['fetch', 'axios', 'ajax', 'xhr', 'request']):
            confidence += 0.1
        if any(keyword in context_lower for keyword in ['api', 'endpoint', 'url', 'baseurl']):
            confidence += 0.1
        if any(keyword in context_lower for keyword in ['secret', 'token', 'key', 'password']):
            confidence += 0.2
            
        return min(confidence, 1.0)

    async def get_extraction_stats(self, js_content: str):
        """
        Get statistics about the extraction process
        """
        urls = await self.extract(js_content)
        
        stats = {
            'total_urls_found': len(urls),
            'url_types': {},
            'confidence_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        for url in urls:
            url_type = self._classify_url(url)
            if url_type not in stats['url_types']:
                stats['url_types'][url_type] = 0
            stats['url_types'][url_type] += 1
            
            # Calculate confidence for categorization
            context = self._get_url_context(js_content, url)
            confidence = self._calculate_confidence(url, context)
            
            if confidence >= 0.7:
                stats['confidence_distribution']['high'] += 1
            elif confidence >= 0.4:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        return stats