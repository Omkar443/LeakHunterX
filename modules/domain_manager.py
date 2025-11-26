import asyncio
from urllib.parse import urlparse
from typing import Set, Tuple, Optional, List, Dict
import logging
from collections import deque
from enum import Enum

class URLState(Enum):
    """Exact state tracking for each URL"""
    DISCOVERED = "discovered"
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESS = "success"
    DNS_FAILURE = "dns_failure"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SSL_ERROR = "ssl_error"
    HTTP_ERROR = "http_error"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"
    OUT_OF_SCOPE = "out_of_scope"

class DomainManager:
    """
    ENHANCED Domain Manager with Priority Queue System
    - Priority-based URL processing
    - High-score domains processed first
    - Comprehensive statistics
    - Data consistency guarantees
    """

    def __init__(self, base_domain: str, max_depth: int = 5, aggressive: bool = False):
        self.base_domain = base_domain.lower()
        self.max_depth = max_depth
        self.aggressive = aggressive
        
        # ENHANCED: Priority-based queue system
        self.high_priority_queue = deque()    # High-score domains (processed first)
        self.medium_priority_queue = deque()  # Medium-score domains
        self.low_priority_queue = deque()     # Low-score domains
        self.standard_queue = deque()         # Unscored domains (legacy)
        
        # Enhanced state tracking
        self.url_states: Dict[str, URLState] = {}  # Exact state for each URL
        self.url_depths: Dict[str, int] = {}  # Depth tracking
        self.url_errors: Dict[str, str] = {}  # Error categorization
        self.url_scores: Dict[str, int] = {}  # Score tracking for prioritization
        
        # Enhanced domain tracking
        self.seen_domains: Set[str] = set()
        self.discovered_subdomains: Set[str] = set()
        self.processed_urls: Set[str] = set()
        
        # Comprehensive statistics
        self.stats = {
            'urls_discovered': 0,
            'urls_queued': 0,
            'urls_processed_success': 0,
            'urls_processed_failure': 0,
            'urls_skipped': 0,
            'domains_discovered': 0,
            'subdomains_discovered': 0,
            'priority_queues': {
                'high_priority': 0,
                'medium_priority': 0,
                'low_priority': 0,
                'standard': 0
            },
            'error_breakdown': {
                'dns_failures': 0,
                'timeouts': 0,
                'connection_errors': 0,
                'ssl_errors': 0,
                'http_errors': 0,
                'other_errors': 0
            }
        }
        
        # Initialize with base domain as high priority
        self.add_priority_target(base_domain, depth=0, score=100)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL to ensure consistent tracking"""
        if not url or not isinstance(url, str):
            return ""
            
        # Ensure URL has scheme
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
            
        # Parse and reconstruct for consistency
        try:
            parsed = urlparse(url)
            # Remove fragments and normalize
            normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"
            if parsed.query:
                normalized += f"?{parsed.query}"
            return normalized
        except Exception:
            return url.lower()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL consistently"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""

    def is_in_scope(self, url: str) -> bool:
        """Enhanced scope checking with exact domain matching"""
        try:
            domain = self._extract_domain(url)
            if not domain:
                return False

            # Exact domain match
            if domain == self.base_domain:
                return True

            # Subdomain match
            if domain.endswith('.' + self.base_domain):
                return True

            # Aggressive mode: same second-level domain
            if self.aggressive:
                base_parts = self.base_domain.split('.')
                domain_parts = domain.split('.')
                
                if len(base_parts) >= 2 and len(domain_parts) >= 2:
                    if domain_parts[-2:] == base_parts[-2:]:
                        return True

            return False

        except Exception as e:
            logging.debug(f"Scope check error for {url}: {e}")
            return False

    def _should_skip_url(self, url: str) -> Tuple[bool, str]:
        """Determine if URL should be skipped with exact reason"""
        normalized_url = self._normalize_url(url)
        
        # Already processed or queued
        if normalized_url in self.url_states:
            current_state = self.url_states[normalized_url]
            return True, f"already_{current_state.value}"
            
        # Out of scope
        if not self.is_in_scope(normalized_url):
            return True, "out_of_scope"
            
        # Invalid URL
        if not normalized_url or not normalized_url.startswith(('http://', 'https://')):
            return True, "invalid_url"
            
        return False, ""

    def _add_to_appropriate_queue(self, url: str, depth: int, score: int = 0):
        """Add URL to appropriate priority queue based on score"""
        # Determine priority based on score
        if score >= 70:
            queue = self.high_priority_queue
            priority_key = 'high_priority'
        elif score >= 40:
            queue = self.medium_priority_queue
            priority_key = 'medium_priority'
        elif score >= 10:
            queue = self.low_priority_queue
            priority_key = 'low_priority'
        else:
            queue = self.standard_queue
            priority_key = 'standard'
        
        # Add to queue
        queue.append((url, depth))
        self.stats['priority_queues'][priority_key] += 1
        
        return priority_key

    def add_target(self, target: str, depth: int = 0) -> bool:
        """
        Add target with exact state tracking (legacy method)
        Returns True if added, False if skipped with reason tracked
        """
        return self.add_priority_target(target, depth, score=0)

    def add_priority_target(self, target: str, depth: int = 0, score: int = 0) -> bool:
        """
        ENHANCED: Add target with priority scoring
        High-score domains get processed first
        """
        try:
            normalized_url = self._normalize_url(target)
            if not normalized_url:
                return False

            # Check if should skip
            should_skip, skip_reason = self._should_skip_url(normalized_url)
            if should_skip:
                self.url_states[normalized_url] = URLState.SKIPPED
                self.stats['urls_skipped'] += 1
                logging.debug(f"Skipped {normalized_url}: {skip_reason}")
                return False

            # Validate depth
            if depth > self.max_depth:
                self.url_states[normalized_url] = URLState.SKIPPED
                self.stats['urls_skipped'] += 1
                return False

            # Add to appropriate priority queue
            priority_key = self._add_to_appropriate_queue(normalized_url, depth, score)
            
            # Update state tracking
            self.url_states[normalized_url] = URLState.QUEUED
            self.url_depths[normalized_url] = depth
            self.url_scores[normalized_url] = score
            
            # Update domain tracking
            domain = self._extract_domain(normalized_url)
            if domain:
                self.seen_domains.add(domain)
                if domain != self.base_domain and domain.endswith('.' + self.base_domain):
                    self.discovered_subdomains.add(domain)

            # Update statistics
            self.stats['urls_discovered'] += 1
            self.stats['urls_queued'] += 1
            self.stats['domains_discovered'] = len(self.seen_domains)
            self.stats['subdomains_discovered'] = len(self.discovered_subdomains)

            logging.debug(f"Queued ({priority_key}): {normalized_url} (depth: {depth}, score: {score})")
            return True

        except Exception as e:
            logging.debug(f"Error adding target {target}: {e}")
            return False

    def add_discovered(self, url: str, depth: int = 0) -> bool:
        """Wrapper for discovered URLs with depth tracking"""
        return self.add_priority_target(url, depth, score=0)

    def add_subdomain(self, subdomain: str) -> bool:
        """Add discovered subdomain with proper URL formatting"""
        return self.add_priority_target(subdomain, depth=0, score=50)  # Medium priority by default

    def add_scored_subdomains(self, scored_domains: List[Dict]) -> int:
        """
        ENHANCED: Bulk add subdomains from scoring engine with proper prioritization
        Returns number successfully added
        """
        added_count = 0
        for domain_data in scored_domains:
            subdomain = domain_data.get('subdomain', '')
            score = domain_data.get('score', 0)
            
            # Only add domains with reasonable scores
            if score > 5:  # Even low-scoring but valid domains
                if self.add_priority_target(subdomain, depth=0, score=score):
                    added_count += 1
                    logging.debug(f"Added scored subdomain: {subdomain} (score: {score})")
                
        return added_count

    def mark_url_processing(self, url: str) -> bool:
        """Mark URL as being processed"""
        normalized_url = self._normalize_url(url)
        if normalized_url in self.url_states and self.url_states[normalized_url] == URLState.QUEUED:
            self.url_states[normalized_url] = URLState.PROCESSING
            return True
        return False

    def mark_url_complete(self, url: str, success: bool = True, error_type: str = None) -> bool:
        """
        Mark URL processing complete with exact state
        Returns True if state was updated
        """
        normalized_url = self._normalize_url(url)
        
        if normalized_url not in self.url_states:
            return False

        if success:
            self.url_states[normalized_url] = URLState.SUCCESS
            self.processed_urls.add(normalized_url)
            self.stats['urls_processed_success'] += 1
        else:
            # Map error type to exact state
            error_state_map = {
                'dns_failure': URLState.DNS_FAILURE,
                'timeout': URLState.TIMEOUT,
                'connection_error': URLState.CONNECTION_ERROR,
                'ssl_error': URLState.SSL_ERROR,
                'http_error': URLState.HTTP_ERROR
            }
            
            state = error_state_map.get(error_type, URLState.CONNECTION_ERROR)
            self.url_states[normalized_url] = state
            self.stats['urls_processed_failure'] += 1
            
            # Update error breakdown
            if error_type in self.stats['error_breakdown']:
                self.stats['error_breakdown'][error_type] += 1
            else:
                self.stats['error_breakdown']['other_errors'] += 1
                
            # Store error details
            self.url_errors[normalized_url] = error_type or "unknown_error"

        return True

    def get_next_target(self) -> Tuple[Optional[str], int]:
        """
        ENHANCED: Get next target with priority-based selection
        High-priority targets are processed first
        """
        try:
            # Check queues in priority order
            if self.high_priority_queue:
                url, depth = self.high_priority_queue.popleft()
                self.stats['priority_queues']['high_priority'] -= 1
            elif self.medium_priority_queue:
                url, depth = self.medium_priority_queue.popleft()
                self.stats['priority_queues']['medium_priority'] -= 1
            elif self.low_priority_queue:
                url, depth = self.low_priority_queue.popleft()
                self.stats['priority_queues']['low_priority'] -= 1
            elif self.standard_queue:
                url, depth = self.standard_queue.popleft()
                self.stats['priority_queues']['standard'] -= 1
            else:
                return None, 0
                
            # Mark as processing
            if self.mark_url_processing(url):
                self.stats['urls_queued'] -= 1
                return url, depth
            else:
                # This shouldn't happen with proper state management
                logging.warning(f"State inconsistency for {url}")
                return None, 0
                
        except Exception as e:
            logging.debug(f"Error getting next target: {e}")
            return None, 0

    def has_targets(self) -> bool:
        """Check if any queue has targets"""
        return (len(self.high_priority_queue) > 0 or 
                len(self.medium_priority_queue) > 0 or 
                len(self.low_priority_queue) > 0 or 
                len(self.standard_queue) > 0)

    def get_queue_stats(self) -> Dict[str, int]:
        """Get detailed queue statistics"""
        return {
            'high_priority': len(self.high_priority_queue),
            'medium_priority': len(self.medium_priority_queue),
            'low_priority': len(self.low_priority_queue),
            'standard': len(self.standard_queue),
            'total_queued': (len(self.high_priority_queue) + 
                           len(self.medium_priority_queue) + 
                           len(self.low_priority_queue) + 
                           len(self.standard_queue))
        }

    def get_url_state(self, url: str) -> Optional[URLState]:
        """Get exact state of a URL"""
        normalized_url = self._normalize_url(url)
        return self.url_states.get(normalized_url)

    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics with exact counts"""
        total_processed = self.stats['urls_processed_success'] + self.stats['urls_processed_failure']
        queue_stats = self.get_queue_stats()
        
        return {
            # URL counts (exact, no duplicates)
            'urls_discovered': self.stats['urls_discovered'],
            'urls_queued': queue_stats['total_queued'],
            'urls_processed_success': self.stats['urls_processed_success'],
            'urls_processed_failure': self.stats['urls_processed_failure'],
            'urls_skipped': self.stats['urls_skipped'],
            'urls_remaining': queue_stats['total_queued'],
            'total_processed': total_processed,
            
            # Domain counts
            'unique_domains': len(self.seen_domains),
            'discovered_subdomains': len(self.discovered_subdomains),
            'base_domain': self.base_domain,
            
            # Priority queue breakdown
            'priority_queues': queue_stats,
            
            # Error breakdown
            'error_breakdown': self.stats['error_breakdown'].copy(),
            
            # Progress
            'progress_percentage': (total_processed / self.stats['urls_discovered'] * 100) 
                                if self.stats['urls_discovered'] > 0 else 0,
            'max_depth_used': max(self.url_depths.values()) if self.url_depths else 0
        }

    def get_discovered_subdomains(self) -> List[str]:
        """Get list of discovered subdomains"""
        return sorted(list(self.discovered_subdomains))

    def get_all_seen_urls(self) -> List[str]:
        """Get all seen URLs with their states"""
        return sorted(self.url_states.keys())

    def get_urls_by_state(self, state: URLState) -> List[str]:
        """Get URLs by specific state"""
        return [url for url, s in self.url_states.items() if s == state]

    def get_successful_urls(self) -> List[str]:
        """Get successfully processed URLs"""
        return self.get_urls_by_state(URLState.SUCCESS)

    def get_failed_urls(self) -> List[str]:
        """Get URLs that failed processing"""
        failed_states = [URLState.DNS_FAILURE, URLState.TIMEOUT, URLState.CONNECTION_ERROR, 
                        URLState.SSL_ERROR, URLState.HTTP_ERROR]
        failed_urls = []
        for state in failed_states:
            failed_urls.extend(self.get_urls_by_state(state))
        return failed_urls

    def get_skipped_urls(self) -> List[str]:
        """Get skipped URLs with reasons"""
        return self.get_urls_by_state(URLState.SKIPPED)

    def clear_queues(self):
        """Clear all queues while maintaining state tracking"""
        # Mark queued items as skipped
        for queue in [self.high_priority_queue, self.medium_priority_queue, 
                     self.low_priority_queue, self.standard_queue]:
            for url, depth in list(queue):
                self.url_states[url] = URLState.SKIPPED
                self.stats['urls_skipped'] += 1
            queue.clear()
        
        # Reset queue stats
        self.stats['priority_queues'] = {
            'high_priority': 0,
            'medium_priority': 0,
            'low_priority': 0,
            'standard': 0
        }
        self.stats['urls_queued'] = 0

    def is_empty(self) -> bool:
        """Check if manager is completely empty"""
        return (len(self.high_priority_queue) == 0 and 
                len(self.medium_priority_queue) == 0 and 
                len(self.low_priority_queue) == 0 and 
                len(self.standard_queue) == 0 and 
                len(self.url_states) == 0)

    def reset(self):
        """Complete reset for new scan"""
        self.high_priority_queue.clear()
        self.medium_priority_queue.clear()
        self.low_priority_queue.clear()
        self.standard_queue.clear()
        self.url_states.clear()
        self.url_depths.clear()
        self.url_errors.clear()
        self.url_scores.clear()
        self.seen_domains.clear()
        self.discovered_subdomains.clear()
        self.processed_urls.clear()
        
        # Reset stats but keep structure
        self.stats = {
            'urls_discovered': 0,
            'urls_queued': 0,
            'urls_processed_success': 0,
            'urls_processed_failure': 0,
            'urls_skipped': 0,
            'domains_discovered': 0,
            'subdomains_discovered': 0,
            'priority_queues': {
                'high_priority': 0,
                'medium_priority': 0,
                'low_priority': 0,
                'standard': 0
            },
            'error_breakdown': {
                'dns_failures': 0,
                'timeouts': 0,
                'connection_errors': 0,
                'ssl_errors': 0,
                'http_errors': 0,
                'other_errors': 0
            }
        }
        
        # Re-add base domain as high priority
        self.add_priority_target(self.base_domain, depth=0, score=100)

    def export_state(self) -> Dict[str, any]:
        """Export complete state for debugging or persistence"""
        return {
            'base_domain': self.base_domain,
            'url_states': {url: state.value for url, state in self.url_states.items()},
            'url_depths': self.url_depths.copy(),
            'url_errors': self.url_errors.copy(),
            'url_scores': self.url_scores.copy(),
            'seen_domains': list(self.seen_domains),
            'discovered_subdomains': list(self.discovered_subdomains),
            'stats': self.stats.copy(),
            'queue_sizes': self.get_queue_stats(),
            'high_priority_queue_contents': list(self.high_priority_queue),
            'medium_priority_queue_contents': list(self.medium_priority_queue),
            'low_priority_queue_contents': list(self.low_priority_queue),
            'standard_queue_contents': list(self.standard_queue)
        }

    def validate_state_consistency(self) -> List[str]:
        """Validate state consistency and return any issues found"""
        issues = []
        
        # Check URL counts consistency
        total_states = len(self.url_states)
        calculated_total = (self.stats['urls_processed_success'] + 
                          self.stats['urls_processed_failure'] + 
                          self.stats['urls_skipped'] + 
                          self.stats['urls_queued'])
        
        if total_states != calculated_total:
            issues.append(f"State count mismatch: {total_states} states vs {calculated_total} calculated")
            
        # Check queue consistency
        total_queued = self.get_queue_stats()['total_queued']
        if total_queued != self.stats['urls_queued']:
            issues.append(f"Queue count mismatch: {total_queued} in queues vs {self.stats['urls_queued']} in stats")
            
        return issues