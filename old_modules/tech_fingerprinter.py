# modules/tech_fingerprinter.py
import re
from typing import Dict, List

class TechFingerprinter:
    def __init__(self):
        self.tech_patterns = self.load_tech_patterns()
    
    def load_tech_patterns(self) -> Dict:
        """Define technology detection patterns"""
        return {
            'wordpress': {
                'headers': ['x-powered-by: wordpress', 'x-wp-something'],
                'body': ['wp-content', 'wp-includes', 'wordpress'],
                'meta': ['generator.*wordpress']
            },
            'jenkins': {
                'headers': ['x-jenkins', 'jenkins'],
                'body': ['jenkins'],
                'title': ['jenkins']
            },
            'grafana': {
                'headers': ['grafana'],
                'body': ['grafana-app'],
                'cookies': ['grafana_session']
            },
            'react': {
                'body': ['react', 'react-dom'],
                'scripts': ['react', 'react.production.min.js']
            },
            'django': {
                'headers': ['server: wsgi', 'x-frame-options: deny'],
                'body': ['csrfmiddlewaretoken'],
                'cookies': ['csrftoken', 'sessionid']
            }
        }
    
    def fingerprint_from_headers(self, headers: Dict) -> List[str]:
        """Detect technologies from HTTP headers"""
        detected_tech = []
        header_str = ' '.join([f"{k}: {v}" for k, v in headers.items()]).lower()
        
        for tech, patterns in self.tech_patterns.items():
            if any(pattern in header_str for pattern in patterns.get('headers', [])):
                detected_tech.append(tech)
        
        return detected_tech
    
    def fingerprint_from_body(self, body: str) -> List[str]:
        """Detect technologies from response body"""
        if not body:
            return []
            
        detected_tech = []
        body_lower = body.lower()
        
        for tech, patterns in self.tech_patterns.items():
            if any(pattern in body_lower for pattern in patterns.get('body', [])):
                detected_tech.append(tech)
        
        return detected_tech
