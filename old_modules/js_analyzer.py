import re
import os
import logging
import aiohttp
from urllib.parse import urljoin
from typing import List, Dict, Set, Tuple
from .utils import save_json, save_to_file, color_text

class JSAnalyzer:
    def __init__(self, domain: str):
        self.domain = domain
        self.leaks = {}
        self.endpoints = set()
        self.results_dir = f"results/{domain}"
        os.makedirs(self.results_dir, exist_ok=True)

        # Enhanced secrets regex patterns
        self.secrets_regex = {
            "aws_access_key": r"AKIA[0-9A-Z]{16}",
            "aws_secret_key": r"[a-zA-Z0-9+/]{40}",
            "google_api_key": r"AIza[0-9A-Za-z\\-_]{35}",
            "google_oauth": r"ya29\\.[0-9A-Za-z\\-_]+",
            "firebase_url": r"[a-zA-Z0-9.-]+\\.firebaseio\\.com",
            "firebase_key": r"AAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140}",
            "slack_token": r"xox[baprs]-[0-9a-zA-Z]{10,48}",
            "slack_webhook": r"https://hooks\\.slack\\.com/services/[A-Za-z0-9+/]+",
            "stripe_key": r"(sk|pk)_(test|live)_[0-9a-zA-Z]{24,99}",
            "twilio_key": r"SK[0-9a-fA-F]{32}",
            "github_token": r"gh[pousr]_[A-Za-z0-9_]{36,255}",
            "mailgun_key": r"key-[0-9a-zA-Z]{32}",
            "heroku_key": r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "jwt_token": r"eyJ[A-Za-z0-9-_=]+\\.[A-Za-z0-9-_=]+\\.?[A-Za-z0-9-_.+/=]*",
            "bearer_token": r"Bearer\\s+[A-Za-z0-9\\-._~+/]+=*",
            "basic_auth": r"[A-Za-z0-9+/]{20,}={0,2}:[A-Za-z0-9+/]{20,}={0,2}",
            "generic_api_key": r"(api[_-]?key|secret[_-]?key|private[_-]?key)[\\s\"':=]+([A-Za-z0-9+/]{20,})",
        }

        # Enhanced endpoint patterns
        self.endpoint_regex = [
            r'["\'](https?:\\/\\/[^"\'\\s<>]+)["\']',
            r'["\'](\\/api\\/[^"\'\\s<>]+)["\']',
            r'["\'](\\/v[0-9]+\\/[^"\'\\s<>]+)["\']',
            r'["\'](\\/graphql[^"\'\\s<>]*)["\']',
            r'["\'](\\/admin[^"\'\\s<>]*)["\']',
            r'["\'](\\/internal[^"\'\\s<>]*)["\']',
            r'["\'](\\/private[^"\'\\s<>]*)["\']',
            r'fetch\\(["\']([^"\'\\s]+)["\']',
            r'axios\\.(get|post|put|delete)\\(["\']([^"\'\\s]+)["\']',
            r'\\.ajax\\([^)]*url["\']?:["\']([^"\'\\s]+)',
            r'window\\.location[^=]*=["\']([^"\'\\s]+)',
        ]

    async def extract_endpoints(self, js_url: str) -> List[str]:
        """Extract endpoints from JavaScript file - FIXED METHOD SIGNATURE"""
        try:
            print(color_text(f"      Analyzing: {js_url}", "cyan"))
            
            async with aiohttp.ClientSession() as session:
                async with session.get(js_url, timeout=15, ssl=False) as response:
                    if response.status != 200:
                        return []
                    
                    content = await response.text(errors='ignore')
                    found_endpoints = set()
                    
                    # Extract endpoints using regex patterns
                    for pattern in self.endpoint_regex:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            # Handle tuple matches (some patterns have groups)
                            if isinstance(match, tuple):
                                match = match[-1]  # Take the last group
                            
                            if match and isinstance(match, str):
                                # Convert relative URLs to absolute
                                if match.startswith('/'):
                                    full_url = urljoin(js_url, match)
                                    found_endpoints.add(full_url)
                                elif match.startswith(('http://', 'https://')):
                                    found_endpoints.add(match)
                    
                    # Save JS content for analysis
                    js_filename = js_url.split('/')[-1] or f"js_{hash(js_url)}.js"
                    js_save_path = os.path.join(self.results_dir, "js_files")
                    os.makedirs(js_save_path, exist_ok=True)
                    
                    with open(os.path.join(js_save_path, js_filename), 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    endpoints_list = list(found_endpoints)
                    if endpoints_list:
                        print(color_text(f"        Found {len(endpoints_list)} endpoints", "green"))
                    
                    return endpoints_list
                    
        except Exception as e:
            logging.debug(f"Error analyzing {js_url}: {e}")
            return []

    def detect_secrets(self, content: str, js_url: str) -> List[Dict]:
        """Detect secrets in JavaScript content"""
        findings = []
        
        for secret_type, pattern in self.secrets_regex.items():
            try:
                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[-1]  # Take the last group from tuple
                        
                        findings.append({
                            "type": secret_type,
                            "value": str(match)[:100],  # Truncate for safety
                            "url": js_url,
                            "confidence": 0.8,
                            "validated": "format_valid"
                        })
            except Exception as e:
                logging.debug(f"Regex error for {secret_type}: {e}")
        
        return findings

    async def analyze_js_files(self, js_urls: List[str]) -> Tuple[List[str], List[Dict]]:
        """Analyze multiple JS files and return endpoints and leaks"""
        all_endpoints = []
        all_leaks = []
        
        for js_url in js_urls:
            try:
                # Extract endpoints from JS file
                endpoints = await self.extract_endpoints(js_url)
                all_endpoints.extend(endpoints)
                
                # Get content for secret detection (we could optimize this)
                async with aiohttp.ClientSession() as session:
                    async with session.get(js_url, timeout=15, ssl=False) as response:
                        if response.status == 200:
                            content = await response.text(errors='ignore')
                            leaks = self.detect_secrets(content, js_url)
                            all_leaks.extend(leaks)
                            
            except Exception as e:
                logging.debug(f"Error processing {js_url}: {e}")
        
        # Save results
        leaks_data = {
            "domain": self.domain,
            "total_leaks_found": len(all_leaks),
            "leaks": all_leaks,
            "endpoints_found": len(all_endpoints)
        }
        
        save_json(os.path.join(self.results_dir, "js_analysis.json"), leaks_data)
        
        # Save endpoints to file
        endpoints_file = os.path.join(self.results_dir, "endpoints.txt")
        with open(endpoints_file, 'w') as f:
            for endpoint in set(all_endpoints):
                f.write(f"{endpoint}\n")
        
        return list(set(all_endpoints)), all_leaks

    def get_stats(self) -> Dict:
        """Get analysis statistics"""
        return {
            "js_files_analyzed": len(self.leaks),
            "endpoints_found": len(self.endpoints),
            "leaks_detected": sum(len(v) for v in self.leaks.values())
        }
