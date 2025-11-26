# modules/scoring_engine.py
import yaml
import aiohttp
import asyncio
import socket
from urllib.parse import urlparse
from typing import Dict, List, Any
from .utils import print_status

class ScoringEngine:
    def __init__(self, config_path: str = "config/scoring_rules.yaml"):
        self.config_path = config_path
        self.scoring_rules = self.load_scoring_rules()
        self.checked_domains = {}  # Cache for live checks
        
    def load_scoring_rules(self) -> Dict:
        """Load scoring rules from YAML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)['scoring']
        except FileNotFoundError:
            print_status("Scoring config not found, using defaults", "warning")
            return self.get_default_rules()
        except Exception as e:
            print_status(f"Error loading scoring rules: {e}, using defaults", "error")
            return self.get_default_rules()
    
    def get_default_rules(self) -> Dict:
        """Default scoring rules as fallback"""
        return {
            'keywords': {
                'api': 15, 'admin': 12, 'auth': 10, 'internal': 10,
                'login': 8, 'secure': 8, 'sso': 10, 'oauth': 10
            },
            'status': {
                'alive': 25, 'status_200': 20, 'status_302': 15,
                'js_present': 15, 'unique_ip': 5
            },
            'cloud': {
                's3': 10, 'azure': 10, 'gcp': 10, 'cloudfront': 8
            },
            'tech_stack': {
                'jenkins': 12, 'grafana': 12, 'kibana': 12, 'phpmyadmin': 15
            },
            'domain_length': {
                'short_bonus': 5, 'max_length': 50, 'length_penalty': 1
            }
        }
    
    def calculate_keyword_score(self, subdomain: str) -> int:
        """Calculate score based on keywords in subdomain"""
        score = 0
        subdomain_lower = subdomain.lower()
        
        for keyword, points in self.scoring_rules['keywords'].items():
            if keyword in subdomain_lower:
                score += points
                print_status(f"    +{points} for keyword: {keyword}", "debug")
        
        return score
    
    def calculate_length_score(self, subdomain: str) -> int:
        """Calculate score based on domain length"""
        domain_part = subdomain.split('.')[0]  # Get subdomain part only
        score = 0
        
        # Shorter domains get bonus points
        if len(domain_part) <= 20:
            bonus = self.scoring_rules['domain_length']['short_bonus']
            score += max(0, bonus - (len(domain_part) // 4))
            print_status(f"    +{score} for short domain", "debug")
        
        return score
    
    def detect_cloud_indicators(self, subdomain: str) -> int:
        """Detect cloud service indicators"""
        score = 0
        subdomain_lower = subdomain.lower()
        
        for service, points in self.scoring_rules['cloud'].items():
            if service in subdomain_lower:
                score += points
                print_status(f"    +{points} for cloud: {service}", "debug")
        
        return score
    
    async def check_dns_resolution(self, domain: str) -> bool:
        """Check if domain resolves via DNS"""
        try:
            socket.getaddrinfo(domain, 443, family=socket.AF_INET)
            return True
        except socket.gaierror:
            return False
    
    async def check_http_status(self, domain: str) -> Dict[str, Any]:
        """Check HTTP status and gather response info"""
        result = {
            'alive': False,
            'status_code': 0,
            'has_js': False,
            'headers': {},
            'ip': None
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://{domain}", 
                    timeout=5, 
                    ssl=False,
                    allow_redirects=True
                ) as response:
                    
                    result['alive'] = True
                    result['status_code'] = response.status
                    result['headers'] = dict(response.headers)
                    
                    # Check for JavaScript in response
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type:
                        text = await response.text(errors='ignore')
                        if any(js_indicator in text.lower() for js_indicator in 
                              ['<script', 'function', 'var ', 'const ', 'let ']):
                            result['has_js'] = True
                    
                    # Get IP address
                    result['ip'] = str(response.connection.transport.get_extra_info('peername')[0])
                    
        except Exception:
            pass  # Domain is not accessible
        
        return result
    
    def calculate_live_score(self, http_info: Dict) -> int:
        """Calculate score based on live checking results"""
        score = 0
        rules = self.scoring_rules['status']
        
        if http_info['alive']:
            score += rules['alive']
            print_status(f"    +{rules['alive']} for alive domain", "debug")
            
            status_code = http_info['status_code']
            if status_code == 200:
                score += rules['status_200']
                print_status(f"    +{rules['status_200']} for status 200", "debug")
            elif status_code == 302:
                score += rules['status_302']
                print_status(f"    +{rules['status_302']} for status 302", "debug")
            elif status_code == 401:
                score += rules['status_401']
                print_status(f"    +{rules['status_401']} for status 401", "debug")
            elif status_code == 403:
                score += rules['status_403']
                print_status(f"    +{rules['status_403']} for status 403", "debug")
            
            if http_info['has_js']:
                score += rules['js_present']
                print_status(f"    +{rules['js_present']} for JS present", "debug")
        
        return score
    
    async def score_subdomain(self, subdomain: str) -> Dict[str, Any]:
        """Calculate comprehensive score for a subdomain"""
        print_status(f"  Scoring: {subdomain}", "debug")
        
        score_breakdown = {}
        total_score = 0
        
        # Keyword-based scoring
        keyword_score = self.calculate_keyword_score(subdomain)
        score_breakdown['keywords'] = keyword_score
        total_score += keyword_score
        
        # Length-based scoring
        length_score = self.calculate_length_score(subdomain)
        score_breakdown['length'] = length_score
        total_score += length_score
        
        # Cloud indicator scoring
        cloud_score = self.detect_cloud_indicators(subdomain)
        score_breakdown['cloud'] = cloud_score
        total_score += cloud_score
        
        # Live checking (DNS + HTTP)
        if await self.check_dns_resolution(subdomain):
            http_info = await self.check_http_status(subdomain)
            live_score = self.calculate_live_score(http_info)
            score_breakdown['live'] = live_score
            total_score += live_score
            score_breakdown['http_info'] = http_info
        else:
            score_breakdown['live'] = 0
            score_breakdown['http_info'] = {'alive': False}
        
        score_breakdown['total'] = total_score
        
        print_status(f"  Total score: {total_score}", "debug")
        
        return {
            'subdomain': subdomain,
            'score': total_score,
            'breakdown': score_breakdown
        }
    
    async def score_subdomains_batch(self, subdomains: List[str], max_workers: int = 50) -> List[Dict]:
        """Score multiple subdomains concurrently"""
        print_status(f"Scoring {len(subdomains)} subdomains...", "info")
        
        # Process in batches to avoid overwhelming
        batch_size = max_workers
        all_scored = []
        
        for i in range(0, len(subdomains), batch_size):
            batch = subdomains[i:i + batch_size]
            print_status(f"  Batch {i//batch_size + 1}: {len(batch)} subdomains", "debug")
            
            tasks = [self.score_subdomain(subdomain) for subdomain in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in batch_results if isinstance(r, dict)]
            all_scored.extend(valid_results)
            
            await asyncio.sleep(0.1)  # Small delay between batches
        
        # Sort by score descending
        all_scored.sort(key=lambda x: x['score'], reverse=True)
        
        return all_scored
    
    def get_top_scorers(self, scored_domains: List[Dict], top_n: int = 2000) -> List[Dict]:
        """Get top N scoring subdomains"""
        return scored_domains[:top_n]
    
    def print_scoring_summary(self, scored_domains: List[Dict]):
        """Print summary of scoring results"""
        if not scored_domains:
            return
            
        top_score = scored_domains[0]['score']
        avg_score = sum(s['score'] for s in scored_domains) / len(scored_domains)
        
        print_status(f"📊 Scoring Summary:", "info")
        print_status(f"   Top score: {top_score}", "info")
        print_status(f"   Average score: {avg_score:.1f}", "info")
        print_status(f"   Domains scored: {len(scored_domains)}", "info")
        
        # Show top 5 for reference
        print_status("   Top 5 subdomains:", "info")
        for i, domain in enumerate(scored_domains[:5]):
            print_status(f"     {i+1}. {domain['subdomain']} - Score: {domain['score']}", "info")
