import asyncio
from urllib.parse import urlparse
from typing import Set, Tuple, Optional, List
import logging
from collections import deque

class DomainManager:
    def __init__(self, base_domain: str, max_depth: int = 5, aggressive: bool = False):
        self.base_domain = base_domain
        self.queue = deque()  # Use deque for efficient popping
        self.seen = set()     # Already processed domains/URLs
        self.seen_domains = set()  # Track unique domains
        self.max_depth = max_depth
        self.aggressive = aggressive
        self.url_depths = {}  # Track depth for each URL
        self.discovered_subdomains = set()
        
        # Initialize with base domain
        self.add_target(base_domain, depth=0)

    def is_in_scope(self, url: str) -> bool:
        """Check if URL is within scope of our target"""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
                
            # Check if domain matches base domain or subdomain
            domain = parsed.netloc.lower()
            base_domain_parts = self.base_domain.lower().split('.')
            
            if domain == self.base_domain.lower():
                return True
                
            # Check for subdomains
            if domain.endswith('.' + self.base_domain.lower()):
                return True
                
            # In aggressive mode, allow related domains
            if self.aggressive:
                # Allow same second-level domain
                domain_parts = domain.split('.')
                if len(domain_parts) >= 2 and len(base_domain_parts) >= 2:
                    if domain_parts[-2:] == base_domain_parts[-2:]:
                        return True
                        
            return False
            
        except Exception as e:
            logging.debug(f"Error checking scope for {url}: {e}")
            return False

    def add_target(self, target: str, depth: int = 0):
        """Add a target (domain or URL) to queue if not already seen and in scope"""
        try:
            # Validate and normalize the target
            if not target or not isinstance(target, str):
                return False
                
            # Ensure URL has scheme
            if not target.startswith(('http://', 'https://')):
                target = f"https://{target}"
                
            # Check if target is in scope
            if not self.is_in_scope(target):
                return False
                
            # Check depth limit
            if depth > self.max_depth:
                return False
                
            # Check if we've already seen this target
            if target in self.seen:
                return False
                
            # Add to queues
            self.queue.append((target, depth))
            self.seen.add(target)
            
            # Track unique domains
            parsed = urlparse(target)
            domain = parsed.netloc.lower()
            self.seen_domains.add(domain)
            
            # Track subdomains
            if domain != self.base_domain.lower() and domain.endswith('.' + self.base_domain.lower()):
                self.discovered_subdomains.add(domain)
                
            logging.debug(f"Added target: {target} (depth: {depth})")
            return True
            
        except Exception as e:
            logging.debug(f"Error adding target {target}: {e}")
            return False

    def add_discovered(self, url: str, depth: int = 0):
        """Wrapper for adding discovered URLs with depth tracking"""
        return self.add_target(url, depth)

    def add_subdomain(self, subdomain: str):
        """Add discovered subdomain to scanning queue"""
        return self.add_target(subdomain, depth=0)

    def get_next_target(self) -> Tuple[Optional[str], int]:
        """Get next target from queue with its depth"""
        try:
            if self.queue:
                target, depth = self.queue.popleft()
                self.url_depths[target] = depth
                return target, depth
            return None, 0
        except Exception as e:
            logging.debug(f"Error getting next target: {e}")
            return None, 0

    def has_targets(self) -> bool:
        """Check if queue has any targets left"""
        return len(self.queue) > 0

    def get_stats(self) -> dict:
        """Get domain manager statistics"""
        return {
            'total_targets_queued': len(self.seen),
            'targets_remaining': len(self.queue),
            'unique_domains': len(self.seen_domains),
            'discovered_subdomains': len(self.discovered_subdomains),
            'max_depth': max(self.url_depths.values()) if self.url_depths else 0
        }

    def get_discovered_subdomains(self) -> List[str]:
        """Get list of discovered subdomains"""
        return list(self.discovered_subdomains)

    def get_all_seen_urls(self) -> List[str]:
        """Get all seen URLs (for reporting)"""
        return list(self.seen)

    def clear_queue(self):
        """Clear the queue (for reset scenarios)"""
        self.queue.clear()

    def is_empty(self) -> bool:
        """Check if manager is completely empty"""
        return len(self.queue) == 0 and len(self.seen) == 0

    def get_progress(self) -> dict:
        """Get progress information"""
        total = len(self.seen)
        remaining = len(self.queue)
        processed = total - remaining
        
        return {
            'processed': processed,
            'remaining': remaining,
            'total': total,
            'progress_percentage': (processed / total * 100) if total > 0 else 0
        }
