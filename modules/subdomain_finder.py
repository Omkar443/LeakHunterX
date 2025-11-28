#!/usr/bin/env python3
"""
LeakHunterX - Smart Subdomain Finder
Efficient discovery with progressive validation
"""

import logging
import subprocess
import asyncio
import aiohttp
import json
import dns.resolver
import dns.asyncresolver
from typing import List, Dict, Any, Set, Optional
import os
import re
from urllib.parse import urlparse
import time
import random
from dataclasses import dataclass
import socket

@dataclass
class DiscoveryResult:
    """Discovery result structure"""
    source: str
    subdomains: List[str]
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0

class SubdomainFinder:
    """
    Smart Subdomain Discovery with Progressive Validation
    """

    def __init__(self, domain: str, dm, use_subfinder: bool = True, aggressive: bool = False):
        self.logger = logging.getLogger("LeakHunterX.SubdomainFinder")
        self.domain = domain
        self.dm = dm
        self.use_subfinder = use_subfinder
        self.aggressive = aggressive
        self.found_subdomains: Set[str] = set()
        self.discovery_stats: Dict[str, Any] = {}
        self.session = None
        self.dns_resolver = None

        # Optimized configuration
        self.config = {
            'timeout': 30,
            'max_concurrent': 8,  # Reduced from 20
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'retry_attempts': 2,
            'rate_limit_delay': 0.2,
            'validation_timeout': 8,
            'max_initial_subs': 200  # Limit initial discovery
        }

        # Priority keywords for smart filtering
        self.priority_keywords = {
            'high': ['api', 'admin', 'auth', 'login', 'dashboard', 'secure'],
            'medium': ['app', 'web', 'dev', 'test', 'staging', 'portal'],
            'low': ['cdn', 'static', 'mail', 'blog', 'support']
        }

        # DNS resolvers
        self.dns_servers = [
            '8.8.8.8', '8.8.4.4',  # Google
            '1.1.1.1', '1.0.0.1',  # Cloudflare
        ]

    async def setup_session(self):
        """Initialize aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            connector = aiohttp.TCPConnector(
                limit=self.config['max_concurrent'],
                verify_ssl=False
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': self.config['user_agent']}
            )

    async def setup_dns_resolver(self):
        """Initialize async DNS resolver"""
        if self.dns_resolver is None:
            self.dns_resolver = dns.asyncresolver.Resolver()
            self.dns_resolver.nameservers = self.dns_servers
            self.dns_resolver.timeout = 5
            self.dns_resolver.lifetime = 8

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def find_subdomains(self) -> List[str]:
        """
        Smart subdomain discovery with progressive validation
        """
        self.logger.info(f"Starting subdomain discovery for {self.domain}")
        start_time = time.time()

        all_subdomains = set()
        discovery_results = {}

        try:
            await self.setup_session()
            await self.setup_dns_resolver()

            # Phase 1: Fast subfinder discovery
            if self.use_subfinder:
                result = await self._discover_with_subfinder()
                discovery_results['subfinder'] = result
                all_subdomains.update(result.subdomains)

            # Phase 2: Quick certificate transparency check
            result = await self._discover_with_crtsh()
            discovery_results['crtsh'] = result
            all_subdomains.update(result.subdomains)

            # Phase 3: Limited common subdomains
            result = await self._discover_with_common_brute()
            discovery_results['common_brute'] = result
            all_subdomains.update(result.subdomains)

            # Limit initial discovery to prevent overload
            initial_subs = list(all_subdomains)[:self.config['max_initial_subs']]

            # Phase 4: Fast DNS validation only
            validated_subs = await self._validate_with_dns_fast(initial_subs)

            # Add validated subdomains to domain manager
            for sub in validated_subs:
                self.dm.add_subdomain(sub)
                self.found_subdomains.add(sub)

            # Calculate statistics
            await self._calculate_discovery_stats(discovery_results, validated_subs, time.time() - start_time)

            self.logger.info(f"Discovery complete: {len(validated_subs)} validated subdomains")
            return validated_subs

        except Exception as e:
            self.logger.error(f"Discovery failed: {e}")
            return await self._fallback_discovery()
        finally:
            await self.close_session()

    async def _discover_with_subfinder(self) -> DiscoveryResult:
        """Fast subdomain discovery using subfinder"""
        start_time = time.time()
        try:
            # Optimized subfinder command with shorter timeout
            cmd = [
                "subfinder",
                "-d", self.domain,
                "-silent",
                "-timeout", "30",  # Reduced from 60
                "-max-time", "60"  # Reduced from 300
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # Overall timeout
            )

            subdomains = []
            if result.returncode == 0:
                subdomains = [s.strip() for s in result.stdout.split('\n') if s.strip()]
                subdomains = self._filter_valid_subdomains(subdomains)

            return DiscoveryResult(
                source="subfinder",
                subdomains=subdomains,
                success=result.returncode == 0,
                error=result.stderr if result.returncode != 0 else None,
                response_time=time.time() - start_time
            )

        except subprocess.TimeoutExpired:  # FIXED: Changed from TimeoutError to TimeoutExpired
            return DiscoveryResult(
                source="subfinder",
                subdomains=[],
                success=False,
                error="Timeout after 60 seconds",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return DiscoveryResult(
                source="subfinder",
                subdomains=[],
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )

    async def _discover_with_crtsh(self) -> DiscoveryResult:
        """Quick certificate transparency check"""
        start_time = time.time()
        try:
            url = f"https://crt.sh/?q=%25.{self.domain}&output=json"

            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    subdomains = set()

                    for cert in data[:100]:  # Limit results
                        common_name = cert.get('common_name', '')
                        if common_name and self._is_valid_subdomain(common_name):
                            subdomains.add(common_name.lower())

                    return DiscoveryResult(
                        source="crtsh",
                        subdomains=list(subdomains),
                        success=True,
                        response_time=time.time() - start_time
                    )
                else:
                    return DiscoveryResult(
                        source="crtsh",
                        subdomains=[],
                        success=False,
                        error=f"HTTP {response.status}",
                        response_time=time.time() - start_time
                    )

        except Exception as e:
            return DiscoveryResult(
                source="crtsh",
                subdomains=[],
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )

    async def _discover_with_common_brute(self) -> DiscoveryResult:
        """Limited common subdomain brute force"""
        start_time = time.time()
        try:
            common_subs = self._generate_optimized_subdomain_list()
            validated_subs = await self._validate_with_dns_fast(common_subs)

            return DiscoveryResult(
                source="common_brute",
                subdomains=validated_subs,
                success=True,
                response_time=time.time() - start_time
            )
        except Exception as e:
            return DiscoveryResult(
                source="common_brute",
                subdomains=[],
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )

    def _generate_optimized_subdomain_list(self) -> List[str]:
        """Generate optimized subdomain list (max 100)"""
        base_words = [
            # High priority
            'www', 'api', 'admin', 'auth', 'login', 'dashboard', 'secure',
            # Medium priority
            'app', 'web', 'dev', 'test', 'staging', 'portal', 'mobile',
            # Common infrastructure
            'mail', 'blog', 'cdn', 'static', 'assets', 'media', 'img',
            'ftp', 'cpanel', 'webmail', 'shop', 'store', 'payment'
        ]

        all_subs = set()

        # Single word subdomains only (no combinations)
        for word in base_words:
            all_subs.add(f"{word}.{self.domain}")

        # Add numbered variants for key subdomains
        key_words = ['api', 'admin', 'app', 'web', 'dev', 'test']
        for word in key_words:
            for i in range(3):  # Only 0,1,2
                all_subs.add(f"{word}{i}.{self.domain}")
                all_subs.add(f"{word}-{i}.{self.domain}")

        return list(all_subs)[:100]  # Hard limit

    async def _validate_with_dns_fast(self, subdomains: List[str]) -> List[str]:
        """Fast DNS validation with limited concurrency"""
        if not subdomains:
            return []

        semaphore = asyncio.Semaphore(5)  # Reduced concurrency
        validated = []

        async def validate_single(sub: str):
            async with semaphore:
                try:
                    # Small random delay to avoid rate limiting
                    await asyncio.sleep(random.uniform(0, 0.1))
                    answers = await self.dns_resolver.resolve(sub, 'A')
                    if answers:
                        return sub
                except Exception:
                    pass
                return None

        # Process in smaller batches
        batch_size = 20
        for i in range(0, len(subdomains), batch_size):
            if len(validated) >= self.config['max_initial_subs']:
                break

            batch = subdomains[i:i + batch_size]
            tasks = [validate_single(sub) for sub in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if result and isinstance(result, str):
                    validated.append(result)

        return validated

    def _is_valid_subdomain(self, hostname: str) -> bool:
        """Validate if hostname is a valid subdomain"""
        if not hostname or not isinstance(hostname, str):
            return False

        hostname = hostname.lower().strip()

        # Basic validation
        if not re.match(r'^[a-z0-9.-]+\.[a-z]{2,}$', hostname):
            return False

        # Must end with our target domain
        if not hostname.endswith(self.domain):
            return False

        # Must be a proper subdomain
        if hostname == self.domain:
            return False

        # Check for invalid patterns
        if '*' in hostname or ' ' in hostname:
            return False

        return True

    def _filter_valid_subdomains(self, subdomains: List[str]) -> List[str]:
        """Filter and validate subdomains"""
        valid_subs = []
        for sub in subdomains:
            if self._is_valid_subdomain(sub):
                valid_subs.append(sub.lower().strip())
        return list(set(valid_subs))

    async def _calculate_discovery_stats(self, results: Dict[str, DiscoveryResult], validated_subs: List[str], total_time: float):
        """Calculate discovery statistics"""
        successful_sources = [k for k, v in results.items() if v.success]
        total_discovered = sum(len(r.subdomains) for r in results.values())

        self.discovery_stats = {
            'domain': self.domain,
            'total_time_seconds': round(total_time, 2),
            'sources_used': list(results.keys()),
            'successful_sources': successful_sources,
            'total_discovered': total_discovered,
            'validated_subdomains': len(validated_subs),
            'validation_rate': round(len(validated_subs) / total_discovered * 100, 2) if total_discovered > 0 else 0
        }

    async def _fallback_discovery(self) -> List[str]:
        """Fallback discovery when primary methods fail"""
        self.logger.warning("Using fallback subdomain discovery")

        try:
            # Simple DNS check for common subdomains
            common_subs = [f"www.{self.domain}", f"api.{self.domain}", f"mail.{self.domain}"]
            validated = await self._validate_with_dns_fast(common_subs)

            for sub in validated:
                self.dm.add_subdomain(sub)
                self.found_subdomains.add(sub)

            return validated
        except Exception as e:
            self.logger.error(f"Fallback discovery failed: {e}")
            return []

    def filter_priority_subdomains(self, subdomains: List[str], max_count: int = 50) -> List[str]:
        """Priority-based subdomain filtering"""
        if len(subdomains) <= max_count:
            return subdomains

        priority_lists = {level: [] for level in ['high', 'medium', 'low']}
        priority_lists['other'] = []

        # Categorize subdomains by priority
        for sub in subdomains:
            sub_lower = sub.lower()
            assigned = False

            for level, keywords in self.priority_keywords.items():
                if any(keyword in sub_lower for keyword in keywords):
                    priority_lists[level].append(sub)
                    assigned = True
                    break

            if not assigned:
                priority_lists['other'].append(sub)

        # Build result in priority order
        result = []
        for level in ['high', 'medium', 'low']:
            result.extend(priority_lists[level])
            if len(result) >= max_count:
                break

        return result[:max_count]

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        return self.discovery_stats

    async def __aenter__(self):
        """Async context manager entry"""
        await self.setup_session()
        await self.setup_dns_resolver()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()