import asyncio
from urllib.parse import urlparse
from typing import Set, Tuple, Optional, List, Dict
import logging
from collections import deque
from enum import Enum
import time

class URLState(Enum):
    """Enhanced state tracking for each URL with conflict resolution"""
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
    STUCK = "stuck"  # NEW: For URLs stuck in processing state

class DomainManager:
    """
    COMPLETELY FIXED: Domain Manager with Robust State Management
    - FIXED: State initialization issues in add_priority_target()
    - FIXED: get_next_target() state validation logic
    - FIXED: URL state transitions and conflict resolution
    - Enhanced error recovery for stuck URLs
    """

    def __init__(self, base_domain: str, max_depth: int = 5, aggressive: bool = False):
        self.base_domain = base_domain.lower()
        self.max_depth = max_depth
        self.aggressive = aggressive

        # FIXED: Enhanced priority-based queue system
        self.high_priority_queue = deque()
        self.medium_priority_queue = deque()
        self.low_priority_queue = deque()
        self.standard_queue = deque()

        # FIXED: Enhanced state tracking with conflict resolution
        self.url_states: Dict[str, URLState] = {}
        self.url_depths: Dict[str, int] = {}
        self.url_errors: Dict[str, str] = {}
        self.url_scores: Dict[str, int] = {}
        self.url_timestamps: Dict[str, float] = {}
        self.processing_timeouts: Dict[str, float] = {}

        # Enhanced domain tracking
        self.seen_domains: Set[str] = set()
        self.discovered_subdomains: Set[str] = set()
        self.processed_urls: Set[str] = set()

        # FIXED: State lock for thread safety
        self._state_lock = asyncio.Lock()

        # Comprehensive statistics with enhanced tracking
        self.stats = {
            'urls_discovered': 0,
            'urls_queued': 0,
            'urls_processed_success': 0,
            'urls_processed_failure': 0,
            'urls_skipped': 0,
            'urls_retried': 0,
            'domains_discovered': 0,
            'subdomains_discovered': 0,
            'state_conflicts': 0,
            'stuck_urls_recovered': 0,
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

        # Check if URL is already in a terminal state
        if normalized_url in self.url_states:
            current_state = self.url_states[normalized_url]

            # FIXED: Allow retry for certain failure states
            retryable_states = [
                URLState.DNS_FAILURE,
                URLState.TIMEOUT,
                URLState.CONNECTION_ERROR,
                URLState.HTTP_ERROR,
                URLState.STUCK
            ]

            if current_state in [URLState.SUCCESS, URLState.SKIPPED, URLState.DUPLICATE, URLState.OUT_OF_SCOPE]:
                return True, f"already_{current_state.value}"
            elif current_state == URLState.PROCESSING:
                # FIXED: Check if URL is stuck in processing state
                if normalized_url in self.url_timestamps:
                    processing_time = time.time() - self.url_timestamps[normalized_url]
                    if processing_time > 300:  # 5 minutes timeout
                        self.url_states[normalized_url] = URLState.STUCK
                        self.stats['stuck_urls_recovered'] += 1
                        logging.debug(f"🔧 Recovered stuck URL: {normalized_url} (was processing for {processing_time:.1f}s)")
                        return False, "recovered_from_stuck"
                return True, "already_processing"
            elif current_state in retryable_states:
                # FIXED: Allow limited retries for failed URLs
                retry_count = self.url_errors.get(f"{normalized_url}_retries", 0)
                if retry_count < 2:
                    self.url_errors[f"{normalized_url}_retries"] = retry_count + 1
                    self.stats['urls_retried'] += 1
                    logging.debug(f"🔄 Allowing retry for {normalized_url} (attempt {retry_count + 1})")
                    return False, f"retry_{current_state.value}"
                else:
                    return True, f"max_retries_exceeded_{current_state.value}"

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

    async def add_target(self, target: str, depth: int = 0) -> bool:
        """Async version with state locking"""
        async with self._state_lock:
            return self.add_priority_target(target, depth, score=0)

    def add_priority_target(self, target: str, depth: int = 0, score: int = 0) -> bool:
        """
        COMPLETELY FIXED: Add target with proper state initialization
        - URLs now correctly start in QUEUED state when added to queues
        - Enhanced state validation and conflict resolution
        """
        try:
            normalized_url = self._normalize_url(target)
            if not normalized_url:
                return False

            # Check if should skip
            should_skip, skip_reason = self._should_skip_url(normalized_url)
            if should_skip:
                # Only mark as skipped if not already in a state
                if normalized_url not in self.url_states:
                    self.url_states[normalized_url] = URLState.SKIPPED
                    self.stats['urls_skipped'] += 1
                logging.debug(f"Skipped {normalized_url}: {skip_reason}")
                return False

            # Validate depth
            if depth > self.max_depth:
                self.url_states[normalized_url] = URLState.SKIPPED
                self.stats['urls_skipped'] += 1
                return False

            # FIXED: Set state to QUEUED BEFORE adding to queue
            self.url_states[normalized_url] = URLState.QUEUED
            self.url_depths[normalized_url] = depth
            self.url_scores[normalized_url] = score
            self.url_timestamps[normalized_url] = time.time()

            # Add to appropriate priority queue
            priority_key = self._add_to_appropriate_queue(normalized_url, depth, score)

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

            logging.debug(f"✅ Queued ({priority_key}): {normalized_url} (depth: {depth}, score: {score}, state: QUEUED)")
            return True

        except Exception as e:
            logging.debug(f"❌ Error adding target {target}: {e}")
            self.stats['state_conflicts'] += 1
            return False

    def add_discovered(self, url: str, depth: int = 0) -> bool:
        """Wrapper for discovered URLs with depth tracking"""
        return self.add_priority_target(url, depth, score=0)

    def add_subdomain(self, subdomain: str) -> bool:
        """Add discovered subdomain with proper URL formatting"""
        return self.add_priority_target(subdomain, depth=0, score=50)

    def add_scored_subdomains(self, scored_domains: List[Dict]) -> int:
        """Bulk add subdomains from scoring engine"""
        added_count = 0
        for domain_data in scored_domains:
            subdomain = domain_data.get('subdomain', '')
            score = domain_data.get('score', 0)

            # Only add domains with reasonable scores
            if score > 5:
                if self.add_priority_target(subdomain, depth=0, score=score):
                    added_count += 1
                    logging.debug(f"Added scored subdomain: {subdomain} (score: {score})")

        return added_count

    def mark_url_processing(self, url: str) -> bool:
        """
        COMPLETELY FIXED: Mark URL as being processed with conflict resolution
        """
        try:
            normalized_url = self._normalize_url(url)
            if not normalized_url:
                return False

            # FIXED: Enhanced state transition logic
            if normalized_url in self.url_states:
                current_state = self.url_states[normalized_url]

                # Allow transition from QUEUED to PROCESSING
                if current_state == URLState.QUEUED:
                    self.url_states[normalized_url] = URLState.PROCESSING
                    self.url_timestamps[normalized_url] = time.time()
                    self.processing_timeouts[normalized_url] = time.time()
                    self.stats['urls_queued'] -= 1
                    logging.debug(f"✅ Marked as processing: {normalized_url}")
                    return True

                # FIXED: Allow retry from certain failure states
                elif current_state in [URLState.DNS_FAILURE, URLState.TIMEOUT, URLState.CONNECTION_ERROR, URLState.HTTP_ERROR, URLState.STUCK]:
                    self.url_states[normalized_url] = URLState.PROCESSING
                    self.url_timestamps[normalized_url] = time.time()
                    self.processing_timeouts[normalized_url] = time.time()
                    logging.debug(f"🔄 Retrying from {current_state.value}: {normalized_url}")
                    return True

                # Already processing or in terminal state
                else:
                    logging.debug(f"⚠️ Cannot mark as processing: {normalized_url} is {current_state.value}")
                    self.stats['state_conflicts'] += 1
                    return False
            else:
                logging.debug(f"❌ URL not found in states: {normalized_url}")
                self.stats['state_conflicts'] += 1
                return False

        except Exception as e:
            logging.debug(f"❌ Error marking URL as processing {url}: {e}")
            self.stats['state_conflicts'] += 1
            return False

    def mark_url_complete(self, url: str, success: bool = True, error_type: str = None) -> bool:
        """
        COMPLETELY FIXED: Mark URL processing complete with robust state management
        """
        try:
            normalized_url = self._normalize_url(url)
            if normalized_url not in self.url_states:
                logging.debug(f"❌ URL not found for completion: {normalized_url}")
                self.stats['state_conflicts'] += 1
                return False

            current_state = self.url_states[normalized_url]

            # FIXED: Allow completion from PROCESSING state
            if current_state == URLState.PROCESSING:
                if success:
                    self.url_states[normalized_url] = URLState.SUCCESS
                    self.processed_urls.add(normalized_url)
                    self.stats['urls_processed_success'] += 1
                    logging.debug(f"✅ Marked as success: {normalized_url}")
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
                    logging.debug(f"❌ Marked as {state.value}: {normalized_url}")

                # Clean up processing timeout tracking
                if normalized_url in self.processing_timeouts:
                    del self.processing_timeouts[normalized_url]

                return True
            else:
                logging.debug(f"⚠️ Unexpected state for completion: {normalized_url} is {current_state.value}")
                self.stats['state_conflicts'] += 1
                return False

        except Exception as e:
            logging.debug(f"❌ Error marking URL complete {url}: {e}")
            self.stats['state_conflicts'] += 1
            return False

    def get_next_target(self) -> Tuple[Optional[str], int]:
        """
        COMPLETELY FIXED: Get next target with enhanced state validation
        - FIXED: Now handles URLs that might not be in perfect QUEUED state
        - FIXED: Emergency state recovery for stuck URLs
        - FIXED: Better conflict resolution
        """
        try:
            # FIXED: Clean up any stuck processing URLs first
            self._cleanup_stuck_urls()

            # FIXED: Emergency state reset if no URLs are being processed but queues are stuck
            if (self.stats['urls_queued'] > 0 and 
                len(self.processing_timeouts) == 0 and 
                not self._has_valid_queued_urls()):
                logging.debug("🔧 Emergency state reset for stuck queues")
                self._emergency_queue_reset()

            # Check queues in priority order
            queues_to_check = [
                (self.high_priority_queue, 'high_priority'),
                (self.medium_priority_queue, 'medium_priority'),
                (self.low_priority_queue, 'low_priority'),
                (self.standard_queue, 'standard')
            ]

            for queue, queue_name in queues_to_check:
                if queue:
                    # FIXED: Check multiple URLs in queue to find valid ones
                    for _ in range(min(len(queue), 5)):  # Check up to 5 URLs
                        url, depth = queue[0]  # Peek at first URL
                        
                        # FIXED: Enhanced state validation with recovery
                        if self._is_url_ready_for_processing(url):
                            # Remove from queue and mark as processing
                            url, depth = queue.popleft()
                            self.stats['priority_queues'][queue_name] -= 1
                            
                            if self.mark_url_processing(url):
                                logging.debug(f"🎯 Got target from {queue_name}: {url}")
                                return url, depth
                            else:
                                # Couldn't mark as processing, skip this URL
                                logging.debug(f"⚠️ Failed to mark as processing, skipping: {url}")
                                self.stats['state_conflicts'] += 1
                                continue
                        else:
                            # URL not ready, rotate to end of queue
                            queue.rotate(-1)
                            logging.debug(f"🔄 Rotating stuck URL in {queue_name}: {url}")

            # No valid targets found
            logging.debug("📭 No valid targets found in any queue")
            return None, 0

        except Exception as e:
            logging.debug(f"❌ Error getting next target: {e}")
            self.stats['state_conflicts'] += 1
            return None, 0

    def _is_url_ready_for_processing(self, url: str) -> bool:
        """
        FIXED: Enhanced URL readiness check with state recovery
        """
        normalized_url = self._normalize_url(url)
        if normalized_url not in self.url_states:
            return False

        current_state = self.url_states[normalized_url]

        # FIXED: Allow URLs in QUEUED state (primary case)
        if current_state == URLState.QUEUED:
            return True

        # FIXED: Allow retry for certain failure states
        retryable_states = [
            URLState.DNS_FAILURE,
            URLState.TIMEOUT, 
            URLState.CONNECTION_ERROR,
            URLState.HTTP_ERROR,
            URLState.STUCK
        ]

        if current_state in retryable_states:
            retry_count = self.url_errors.get(f"{normalized_url}_retries", 0)
            if retry_count < 2:
                logging.debug(f"🔄 Allowing retry for {normalized_url} from {current_state.value}")
                return True

        # FIXED: Recover URLs stuck in unexpected states
        if current_state == URLState.PROCESSING:
            # Check if stuck in processing
            if normalized_url in self.url_timestamps:
                processing_time = time.time() - self.url_timestamps[normalized_url]
                if processing_time > 300:  # 5 minutes
                    logging.debug(f"🔧 Recovering stuck URL from processing: {normalized_url}")
                    self.url_states[normalized_url] = URLState.QUEUED
                    return True

        return False

    def _has_valid_queued_urls(self) -> bool:
        """Check if any queue has URLs ready for processing"""
        all_queues = [
            self.high_priority_queue,
            self.medium_priority_queue, 
            self.low_priority_queue,
            self.standard_queue
        ]
        
        for queue in all_queues:
            for url, depth in list(queue):
                if self._is_url_ready_for_processing(url):
                    return True
        return False

    def _emergency_queue_reset(self):
        """Emergency reset for completely stuck queues"""
        reset_count = 0
        all_queues = [
            (self.high_priority_queue, 'high_priority'),
            (self.medium_priority_queue, 'medium_priority'),
            (self.low_priority_queue, 'low_priority'),
            (self.standard_queue, 'standard')
        ]

        for queue, queue_name in all_queues:
            for url, depth in list(queue):
                if url in self.url_states and self.url_states[url] != URLState.QUEUED:
                    self.url_states[url] = URLState.QUEUED
                    reset_count += 1
                    logging.debug(f"🔧 Emergency reset: {url} -> QUEUED")

        if reset_count > 0:
            logging.debug(f"🔧 Emergency queue reset completed: {reset_count} URLs recovered")
            self.stats['stuck_urls_recovered'] += reset_count

    def _cleanup_stuck_urls(self):
        """Clean up URLs stuck in processing state"""
        current_time = time.time()
        stuck_urls = []

        for url, start_time in self.processing_timeouts.items():
            if current_time - start_time > 300:  # 5 minutes timeout
                stuck_urls.append(url)
                logging.debug(f"🔧 Cleaning up stuck URL: {url}")

        for url in stuck_urls:
            if url in self.url_states and self.url_states[url] == URLState.PROCESSING:
                self.url_states[url] = URLState.STUCK
                del self.processing_timeouts[url]
                self.stats['stuck_urls_recovered'] += 1
                logging.debug(f"✅ Recovered stuck URL: {url}")

    def has_targets(self) -> bool:
        """Check if any queue has valid targets"""
        self._cleanup_stuck_urls()
        return (len(self.high_priority_queue) > 0 or
                len(self.medium_priority_queue) > 0 or
                len(self.low_priority_queue) > 0 or
                len(self.standard_queue) > 0)

    # ... (rest of the methods remain exactly the same - no changes needed below this line)

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
        """Get comprehensive statistics with enhanced tracking"""
        total_processed = self.stats['urls_processed_success'] + self.stats['urls_processed_failure']
        queue_stats = self.get_queue_stats()

        return {
            'urls_discovered': self.stats['urls_discovered'],
            'urls_queued': queue_stats['total_queued'],
            'urls_processed_success': self.stats['urls_processed_success'],
            'urls_processed_failure': self.stats['urls_processed_failure'],
            'urls_skipped': self.stats['urls_skipped'],
            'urls_retried': self.stats['urls_retried'],
            'urls_remaining': queue_stats['total_queued'],
            'total_processed': total_processed,
            'unique_domains': len(self.seen_domains),
            'discovered_subdomains': len(self.discovered_subdomains),
            'base_domain': self.base_domain,
            'state_conflicts': self.stats['state_conflicts'],
            'stuck_urls_recovered': self.stats['stuck_urls_recovered'],
            'currently_processing': len(self.processing_timeouts),
            'priority_queues': queue_stats,
            'error_breakdown': self.stats['error_breakdown'].copy(),
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
                        URLState.SSL_ERROR, URLState.HTTP_ERROR, URLState.STUCK]
        failed_urls = []
        for state in failed_states:
            failed_urls.extend(self.get_urls_by_state(state))
        return failed_urls

    def get_processing_urls(self) -> List[str]:
        """Get URLs currently being processed"""
        return list(self.processing_timeouts.keys())

    def get_skipped_urls(self) -> List[str]:
        """Get skipped URLs with reasons"""
        return self.get_urls_by_state(URLState.SKIPPED)

    def clear_queues(self):
        """Clear all queues while maintaining state tracking"""
        for queue in [self.high_priority_queue, self.medium_priority_queue,
                     self.low_priority_queue, self.standard_queue]:
            for url, depth in list(queue):
                self.url_states[url] = URLState.SKIPPED
                self.stats['urls_skipped'] += 1
            queue.clear()

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
        self.url_timestamps.clear()
        self.processing_timeouts.clear()
        self.seen_domains.clear()
        self.discovered_subdomains.clear()
        self.processed_urls.clear()

        self.stats = {
            'urls_discovered': 0,
            'urls_queued': 0,
            'urls_processed_success': 0,
            'urls_processed_failure': 0,
            'urls_skipped': 0,
            'urls_retried': 0,
            'domains_discovered': 0,
            'subdomains_discovered': 0,
            'state_conflicts': 0,
            'stuck_urls_recovered': 0,
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

        self.add_priority_target(self.base_domain, depth=0, score=100)

    def validate_state_consistency(self) -> List[str]:
        """Enhanced state consistency validation"""
        issues = []

        total_states = len(self.url_states)
        calculated_total = (self.stats['urls_processed_success'] +
                          self.stats['urls_processed_failure'] +
                          self.stats['urls_skipped'] +
                          self.stats['urls_queued'])

        if total_states != calculated_total:
            issues.append(f"State count mismatch: {total_states} states vs {calculated_total} calculated")

        total_queued = self.get_queue_stats()['total_queued']
        if total_queued != self.stats['urls_queued']:
            issues.append(f"Queue count mismatch: {total_queued} in queues vs {self.stats['urls_queued']} in stats")

        stuck_count = len(self.processing_timeouts)
        if stuck_count > 0:
            issues.append(f"Found {stuck_count} URLs stuck in processing state")

        return issues

    def export_state(self) -> Dict[str, any]:
        """Export complete state for debugging"""
        return {
            'base_domain': self.base_domain,
            'url_states': {url: state.value for url, state in self.url_states.items()},
            'url_depths': self.url_depths.copy(),
            'url_errors': self.url_errors.copy(),
            'url_scores': self.url_scores.copy(),
            'url_timestamps': self.url_timestamps.copy(),
            'processing_timeouts': self.processing_timeouts.copy(),
            'seen_domains': list(self.seen_domains),
            'discovered_subdomains': list(self.discovered_subdomains),
            'stats': self.stats.copy(),
            'queue_sizes': self.get_queue_stats(),
            'state_issues': self.validate_state_consistency()
        }

    def force_reset_all_processing(self):
        """Force reset all stuck URLs to queued state"""
        stuck_count = 0
        for url in list(self.url_states.keys()):
            if self.url_states[url] in [URLState.PROCESSING, URLState.STUCK]:
                self.url_states[url] = URLState.QUEUED
                if url in self.processing_timeouts:
                    del self.processing_timeouts[url]
                stuck_count += 1

        if stuck_count > 0:
            logging.debug(f"🔧 Force reset {stuck_count} stuck URLs")
            self.stats['stuck_urls_recovered'] += stuck_count

        return stuck_count