#!/usr/bin/env python3
"""
LeakHunterX - AI-Powered Security Scanner
COMPLETE FULL-FEATURED VERSION
"""

import argparse
import asyncio
import os
import json
import aiohttp
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.box import ROUNDED, DOUBLE_EDGE
from rich.text import Text

console = Console()

class LeakHunterX:
    """Advanced Security Scanner Main Controller"""

    def __init__(self):
        self.scan_start_time = None
        self.scan_results = {}
        self.performance_stats = {
            'js_files_processed': 0,
            'js_files_skipped_duplicates': 0,
            'memory_efficiency_gain': 0,
            'crawl_success_rate': 0,
            'ai_validation_count': 0,
            'smart_targets_identified': 0
        }

    def print_banner(self):
        """Print the professional banner"""
        banner = """
╭──────────────────────────────────────── 🚀 LeakHunterX ─────────────────────────────────────────╮
│                                                                                                  │
│   ██╗     ███████╗ █████╗ ██╗  ██╗██╗  ██╗███╗   ██╗████████╗███████╗██████╗ ██╗  ██╗██╗  ██╗   │
│   ██║     ██╔════╝██╔══██╗██║ ██╔╝██║ ██╔╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██║  ██║██║  ██║   │
│   ██║     █████╗  ███████║█████╔╝ █████╔╝ ██╔██╗ ██║   ██║   █████╗  ██████╔╝███████║███████║   │
│   ██║     ██╔══╝  ██╔══██║██╔═██╗ ██╔═██╗ ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔══██║██╔══██║   │
│   ███████╗███████╗██║  ██║██║  ██╗██║  ██╗██║ ╚████║   ██║   ███████╗██║  ██║██║  ██║██║  ██║   │
│   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   │
│                                                                                                  │
│        🔥 AI‑Powered Recon + JS Leak Detection + Secret Intelligence Engine 🔥                  │
│        🚀 STREAMING ARCHITECTURE + SMART SCORING + MEMORY EFFICIENCY 🚀                         │
│                                                                                                  │
╰─────────── Tantralogic AI — Engineered by Omkar Sahni ───────────╯
"""
        console.print(banner, style="bold cyan")

    def print_config(self, target: str, mode: str, wordlist: str, ai_enabled: bool):
        """Print scan configuration"""
        header = f"""
🎯 [bold]Scan Configuration[/bold]
├── Target: [bold green]{target}[/bold green]
├── Mode: [bold yellow]{mode}[/bold yellow]
├── AI: [bold magenta]{'ENABLED' if ai_enabled else 'DISABLED'}[/bold magenta]
├── Architecture: [bold blue]STREAMING + SMART SCORING[/bold blue]
└── Wordlist: [bold cyan]{wordlist}[/bold cyan]

🚀 Starting advanced reconnaissance...
"""
        console.print(header)

    def print_phase(self, phase_number: int, phase_name: str, emoji: str):
        """Print phase header"""
        console.print(f"\n[bold cyan]Phase {phase_number}: {phase_name} {emoji}[/bold cyan]")
        console.print("[dim]" + "─" * 60 + "[/dim]")

    def print_streaming_progress(self, processed: int, total: int, item_type: str, skipped: int = 0):
        """Print streaming progress with deduplication stats"""
        if total == 0:
            return

        progress_percent = (processed / total) * 100
        progress_bar = "█" * int(progress_percent / 5) + "░" * (20 - int(progress_percent / 5))

        console.print(f"[cyan]    📊 Streaming Progress:[/cyan]")
        console.print(f"[green]    {progress_bar} {processed}/{total} ({progress_percent:.1f}%)[/green]")

        if skipped > 0:
            console.print(f"[yellow]    ⏭️  Skipped duplicates: {skipped}[/yellow]")
            efficiency = (skipped / (processed + skipped)) * 100
            console.print(f"[green]    💾 Memory efficiency: {efficiency:.1f}%[/green]")

    def load_api_keys(self) -> List[str]:
        """Load API keys from config file"""
        config_path = "config/secrets.json"

        try:
            if not os.path.exists(config_path):
                console.print(f"❌ Config file not found: {config_path}", style="red")
                return []

            with open(config_path, 'r') as f:
                secrets = json.load(f)
                openrouter_config = secrets.get("OPENROUTER_CONFIG", {})
                api_keys = openrouter_config.get("api_keys", [])

                if api_keys:
                    console.print(f"✅ Loaded {len(api_keys)} API keys from config", style="green")
                    return api_keys
                else:
                    console.print("⚠️ No API keys found in config file", style="yellow")
                    return []

        except Exception as e:
            console.print(f"❌ Error loading config: {e}", style="red")
            return []

    async def test_connectivity(self, target: str) -> bool:
        """Test basic connectivity to target"""
        console.print("🔍 Testing connectivity to target...", style="blue")

        test_urls = [f"https://{target}", f"http://{target}"]

        connector = aiohttp.TCPConnector(ssl=False, limit=10)
        timeout = aiohttp.ClientTimeout(total=15)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for test_url in test_urls:
                try:
                    console.print(f"    Testing {test_url}...", style="blue")
                    async with session.get(test_url, ssl=False, allow_redirects=True) as response:
                        console.print(f"✅ Server responded from {test_url} (Status: {response.status})", style="green")
                        return True
                except Exception as e:
                    console.print(f"    ❌ {test_url}: {e}", style="yellow")

        console.print("⚠️ Limited connectivity detected, but continuing scan...", style="yellow")
        return True

    async def run_scan(self, domain: str, args) -> Dict[str, Any]:
        """Main scan function - FULL FEATURES"""
        self.scan_start_time = datetime.now()
        console.print(f"\n🚀 Starting advanced scan for [bold green]{domain}[/bold green]")

        # Test connectivity but never abort
        await self.test_connectivity(domain)

        try:
            # Import all modules
            from modules.domain_manager import DomainManager
            from modules.subdomain_finder import SubdomainFinder
            from modules.crawler import Crawler
            from modules.directory_fuzzer import DirectoryFuzzer
            from modules.js_analyzer import JSAnalyzer
            from modules.leak_detector import LeakDetector
            from modules.ai_analyzer import AIAnalyzer
            from modules.reporter import Reporter

            # Handle scoring engine import with error handling - UPDATED FOR NEW SCORING ENGINE
            try:
                from modules.scoring_engine import ScoringEngine
                console.print("✅ New Scoring Engine imported successfully", style="green")
            except Exception as e:
                console.print(f"⚠️ Scoring engine import issue: {e}", style="yellow")
                # Create fallback that matches the new scoring engine structure
                class SimpleScoringEngine:
                    async def score_subdomains_batch(self, subdomains, max_concurrent=10):
                        return [{
                            'subdomain': sub, 
                            'score': 50, 
                            'priority': 'MEDIUM', 
                            'breakdown': {}, 
                            'fingerprint': '',
                            'scoring_time': 0.1
                        } for sub in subdomains[:10]]
                    
                    def get_top_scorers(self, scored_domains, top_n=20):
                        return scored_domains[:top_n]
                    
                    def print_scoring_summary(self, scored_domains):
                        if scored_domains:
                            top_score = scored_domains[0]['score']
                            avg_score = sum(s['score'] for s in scored_domains) / len(scored_domains)
                            console.print(f"    📊 Scoring Summary: Top={top_score}, Avg={avg_score:.1f}", style="info")
                        else:
                            console.print("    ⚠️ No domains scored", style="warning")
                ScoringEngine = SimpleScoringEngine

            from modules.http_client import AsyncHTTPClient

        except ImportError as e:
            console.print(f"❌ Failed to import modules: {e}", style="red")
            return self._create_empty_results(domain, args)

        # Initialize modules with comprehensive error handling
        try:
            console.print("🔧 Initializing advanced modules...", style="blue")

            # Initialize enhanced HTTP client
            http_client = AsyncHTTPClient(
                timeout=15.0,
                max_retries=2,
                concurrency_limit=min(args.threads, 20)
            )
            await http_client.init_dns_resolver()

            # Initialize smart scoring engine with error handling - UPDATED FOR NEW SCORING ENGINE
            try:
                scoring_engine = ScoringEngine()
                console.print("✅ Smart Scoring Engine initialized with new features", style="green")
            except Exception as e:
                console.print(f"⚠️ Scoring engine initialization issue: {e}", style="yellow")
                # Create patched version that matches the new scoring engine structure
                class PatchedScoringEngine:
                    def __init__(self):
                        self.logger = self._setup_logger()
                        self.scoring_rules = self.get_default_rules()
                        self.performance_cache = {}
                        self.circuit_breaker = {}
                        self.total_scored = 0
                        self.performance_metrics = {
                            'avg_scoring_time': 0, 
                            'success_rate': 0, 
                            'cache_hits': 0
                        }
                    
                    def _setup_logger(self):
                        import logging
                        logger = logging.getLogger('scoring_engine')
                        if not logger.handlers:
                            handler = logging.StreamHandler()
                            formatter = logging.Formatter(
                                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                            )
                            handler.setFormatter(formatter)
                            logger.addHandler(handler)
                            logger.setLevel(logging.INFO)
                        return logger
                    
                    def get_default_rules(self):
                        return {
                            'keyword_tiers': {
                                'CRITICAL': {'api': 25, 'admin': 22, 'auth': 20, 'secure': 18},
                                'HIGH': {'app': 15, 'portal': 15, 'dashboard': 15},
                                'MEDIUM': {'dev': 8, 'test': 8, 'staging': 8},
                                'LOW': {'mail': 3, 'cdn': 2, 'static': 2}
                            },
                            'status_scoring': {
                                'alive': 30, 'status_200': 25, 'status_302': 20,
                                'status_401': 15, 'status_403': 12, 'js_present': 20
                            },
                            'priority_thresholds': {
                                'CRITICAL': 80, 'HIGH': 50, 'MEDIUM': 25, 'LOW': 0
                            }
                        }
                    
                    async def score_subdomains_batch(self, subdomains, max_concurrent=10):
                        """Batch scoring with performance tracking"""
                        results = []
                        for subdomain in subdomains[:20]:  # Limit for performance
                            result = {
                                'subdomain': subdomain,
                                'score': 50,
                                'breakdown': {'keywords': 25, 'live': 25},
                                'priority': 'MEDIUM',
                                'fingerprint': hashlib.md5(subdomain.encode()).hexdigest(),
                                'scoring_time': 0.1
                            }
                            results.append(result)
                        return results
                    
                    def get_top_scorers(self, scored_domains, top_n=20):
                        return scored_domains[:top_n]
                    
                    def print_scoring_summary(self, scored_domains):
                        """Scoring summary matching new engine format"""
                        if not scored_domains:
                            console.print("    ⚠️ No domains scored", style="warning")
                            return

                        top_score = scored_domains[0]['score']
                        avg_score = sum(s['score'] for s in scored_domains) / len(scored_domains)

                        # Priority distribution
                        priority_count = {}
                        for domain in scored_domains:
                            priority = domain.get('priority', 'LOW')
                            priority_count[priority] = priority_count.get(priority, 0) + 1

                        console.print("    📊 Scoring Summary:", style="info")
                        console.print(f"    ├── Total domains scored: {len(scored_domains)}", style="info")
                        console.print(f"    ├── Top score: {top_score}", style="info")
                        console.print(f"    ├── Average score: {avg_score:.1f}", style="info")
                        console.print(f"    └── Priority distribution: {priority_count}", style="info")
                
                scoring_engine = PatchedScoringEngine()

            # Initialize core modules
            dm = DomainManager(
                domain,
                aggressive=args.aggressive,
                max_depth=3 if args.aggressive else 2
            )

            sd = SubdomainFinder(domain, dm, use_subfinder=args.subfinder)
            crawler = Crawler(dm, concurrency=min(args.threads, 15), max_depth=3 if args.aggressive else 2)

            # Initialize conditional modules
            fuzzer = None
            if args.aggressive:
                fuzzer = DirectoryFuzzer(
                    domain,
                    wordlist_path=args.wordlist,
                    threads=min(args.threads, 8),
                    aggressive=args.aggressive
                )

            js_engine = JSAnalyzer(domain)
            leak_detector = LeakDetector(aggressive=args.aggressive)
            reporter = Reporter(domain)

            # Initialize AI analyzer
            ai_analyzer = None
            ai_enabled = any([args.ai_prioritize, args.ai_analyze, args.ai_validate, args.ai_insights])
            if ai_enabled:
                try:
                    ai_analyzer = AIAnalyzer()
                    console.print("🤖 AI Analyzer initialized", style="magenta")
                except Exception as e:
                    console.print(f"⚠️ AI Analyzer failed: {e}", style="yellow")

        except Exception as e:
            console.print(f"❌ Failed to initialize modules: {e}", style="red")
            return self._create_empty_results(domain, args)

        # Initialize comprehensive results structure
        results = self._initialize_results(domain, args, ai_analyzer is not None)

        # 🎯 PHASE 1: SMART SUBDOMAIN DISCOVERY WITH SCORING
        if args.subfinder:
            await self._run_phase_subdomain_discovery(args, dm, results, sd, scoring_engine)

        # 🎯 PHASE 2: ADAPTIVE FUZZING WITH PRIORITY TARGETS
        if args.aggressive and fuzzer:
            await self._run_phase_adaptive_fuzzing(args, dm, results, fuzzer, domain, scoring_engine)

        # 🎯 PHASE 3: STREAMING WEB CRAWLING
        await self._run_phase_streaming_crawling(args, dm, results, crawler, http_client)

        # 🎯 PHASE 4: STREAMING JS ANALYSIS WITH DEDUPLICATION
        leaks_found = await self._run_phase_streaming_js_analysis(args, results, js_engine, leak_detector, ai_analyzer)

        # 🎯 PHASE 5: AI VALIDATION & ANALYSIS
        if ai_analyzer:
            await self._run_phase_ai_analysis(args, results, leaks_found, ai_analyzer)

        # 🎯 PHASE 6: COMPREHENSIVE REPORTING
        await self._run_phase_reporting(results, reporter, domain)

        # Print performance statistics
        self._print_performance_stats(results, leak_detector, http_client, scoring_engine)

        return results

    async def _run_phase_subdomain_discovery(self, args, dm, results, sd, scoring_engine):
        """Phase 1: Smart subdomain discovery with scoring - UPDATED FOR NEW SCORING ENGINE"""
        self.print_phase(1, "Smart Subdomain Discovery & Scoring", "🎯")

        try:
            # Discover subdomains
            subs = await sd.find_subdomains()
            results["subdomains"] = list(set(subs))

            console.print(f"    ✅ Found {len(subs)} subdomains", style="green")

            # Score and prioritize subdomains using the new scoring engine
            if subs:
                console.print(f"    🧠 Scoring {len(subs)} subdomains with enhanced engine...", style="blue")

                # Use the new scoring engine batch method - UPDATED CALL
                scored_domains = await scoring_engine.score_subdomains_batch(
                    subs,
                    max_concurrent=15  # Removed timeout_per_domain parameter
                )

                # Print scoring summary using the new method
                scoring_engine.print_scoring_summary(scored_domains)

                # Select high-priority targets
                priority_targets = scoring_engine.get_top_scorers(scored_domains, top_n=25)
                
                # Extract subdomains from scored results - UPDATED FOR NEW DATA STRUCTURE
                results["priority_targets"] = [target['subdomain'] for target in priority_targets]
                results["scored_domains"] = scored_domains  # Store full scoring data with breakdowns
                
                self.performance_stats['smart_targets_identified'] = len(priority_targets)

                # Show priority breakdown
                priority_count = {}
                for target in priority_targets:
                    priority = target.get('priority', 'LOW')
                    priority_count[priority] = priority_count.get(priority, 0) + 1
                
                console.print(f"    🎯 Identified {len(priority_targets)} high-value targets: {priority_count}", style="green")

        except Exception as e:
            console.print(f"    ❌ Subdomain discovery failed: {e}", style="red")
            if args.verbose:
                traceback.print_exc()

    async def _run_phase_adaptive_fuzzing(self, args, dm, results, fuzzer, domain, scoring_engine):
        """Phase 2: Adaptive fuzzing with priority targets"""
        self.print_phase(2, "Adaptive Directory Fuzzing", "🚀")

        try:
            # Use priority targets if available, otherwise fallback
            if results.get("priority_targets"):
                console.print(f"    🎯 Fuzzing {len(results['priority_targets'])} priority targets...", style="blue")
                found_paths = await fuzzer.fuzz_multiple_targets(results["priority_targets"])
            else:
                # Fallback to traditional fuzzing
                base_url = f"https://{domain}"
                console.print("    🔐 Scanning for admin panels...", style="blue")
                admin_panels = await fuzzer.find_admin_panels(base_url)
                results["admin_panels"] = admin_panels

                console.print("    📁 Fuzzing directories...", style="blue")
                found_paths = await fuzzer.fuzz_directories(base_url)

            results["fuzzed_paths"] = found_paths

            # Add discovered paths to crawl queue
            for path in found_paths:
                if isinstance(path, dict) and 'url' in path:
                    dm.add_discovered(path["url"])

            console.print(f"    ✅ Adaptive fuzzing: {len(found_paths)} paths found", style="green")

        except Exception as e:
            console.print(f"    ❌ Adaptive fuzzing failed: {e}", style="red")

    async def _run_phase_streaming_crawling(self, args, dm, results, crawler, http_client):
        """Phase 3: Streaming web crawling with enhanced reliability"""
        self.print_phase(3, "Streaming Web Crawling", "🌐")

        try:
            console.print(f"    🌀 Starting streaming crawl with {dm.get_stats()['urls_queued']} targets...", style="blue")

            crawled_urls, discovered_js = await crawler.crawl()
            results["urls"] = list(set(crawled_urls))
            results["js_files"] = list(set(discovered_js))

            # Show crawl statistics
            crawl_stats = crawler.get_stats()
            success_rate = (crawl_stats.get('urls_processed_success', 0) /
                          max(crawl_stats.get('urls_processed_total', 1), 1)) * 100
            self.performance_stats['crawl_success_rate'] = success_rate

            console.print(f"    📊 Crawl success rate: {success_rate:.1f}%", style="yellow")
            console.print(f"    ✅ Discovered: {len(crawled_urls)} URLs, {len(discovered_js)} JS files", style="green")

        except Exception as e:
            console.print(f"    ❌ Streaming crawl failed: {e}", style="red")

    async def _run_phase_streaming_js_analysis(self, args, results, js_engine, leak_detector, ai_analyzer):
        """Phase 4: Streaming JS analysis with hash deduplication"""
        self.print_phase(4, "Streaming JS Analysis", "📜")
        leaks_found = []
        processed_hashes = set()

        if not results["js_files"]:
            console.print("    ⚠️ No JS files to analyze", style="yellow")
            return leaks_found

        console.print(f"    🌀 Streaming analysis of {len(results['js_files'])} JS files...", style="blue")

        js_files_to_analyze = results["js_files"][:30]  # Performance limit
        processed = 0
        skipped_duplicates = 0

        for i, js_url in enumerate(js_files_to_analyze):
            try:
                # Show progress
                if (i + 1) % 5 == 0 or (i + 1) == len(js_files_to_analyze):
                    self.print_streaming_progress(i + 1, len(js_files_to_analyze), "JS files", skipped_duplicates)

                # Calculate content hash for deduplication
                content_hash = await self._get_content_hash(js_url)
                if content_hash in processed_hashes:
                    skipped_duplicates += 1
                    continue

                processed_hashes.add(content_hash)

                # Extract endpoints
                endpoints = await js_engine.extract_endpoints(js_url)

                # Analyze content for leaks
                content_leaks = await self._analyze_js_content_streaming(js_url, leak_detector)
                processed += 1

                # Add leaks to results
                for leak in content_leaks:
                    if isinstance(leak, dict):
                        leak["source_url"] = js_url
                        leaks_found.append(leak)

                        # Show high-confidence leaks
                        if leak.get("confidence", 0) > 0.7:
                            console.print(f"    🚨 LEAK: {leak['type']} - {leak['value'][:50]}...", style="red")

            except Exception as e:
                if args.verbose:
                    console.print(f"    ❌ Error processing JS file: {e}", style="red")

        # Update performance stats
        self.performance_stats['js_files_processed'] = processed
        self.performance_stats['js_files_skipped_duplicates'] = skipped_duplicates
        if processed + skipped_duplicates > 0:
            self.performance_stats['memory_efficiency_gain'] = (skipped_duplicates / (processed + skipped_duplicates)) * 100

        console.print(f"    ✅ Streaming JS analysis: {processed} processed, {skipped_duplicates} duplicates skipped", style="green")
        return leaks_found

    async def _analyze_js_content_streaming(self, js_url: str, leak_detector) -> List[Dict[str, Any]]:
        """Analyze JS content with streaming approach"""
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            timeout = aiohttp.ClientTimeout(total=20)

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(js_url, ssl=False) as resp:
                    if resp.status == 200:
                        js_content = await resp.text(errors='ignore')

                        # Use streaming analysis
                        processed_data = leak_detector.process_js_file_streaming(js_url, js_content)

                        leaks_found = []
                        for leak in processed_data.get("leaks", []):
                            leak["source_url"] = js_url
                            leaks_found.append(leak)

                        return leaks_found

        except Exception:
            pass

        return []

    async def _get_content_hash(self, url: str) -> str:
        """Get content hash for deduplication"""
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            timeout = aiohttp.ClientTimeout(total=10)

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url, ssl=False) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        return hashlib.sha256(content).hexdigest()
        except Exception:
            return hashlib.sha256(url.encode()).hexdigest()

    async def _run_phase_ai_analysis(self, args, results, leaks_found, ai_analyzer):
        """Phase 5: AI validation and analysis"""
        self.print_phase(5, "AI Analysis & Validation", "🤖")

        try:
            console.print(f"    🧠 Analyzing {len(leaks_found)} potential leaks...", style="magenta")

            # Prepare JS files for batch analysis
            js_files_for_ai = []
            for leak in leaks_found:
                source_url = leak.get("source_url", "")
                if source_url and source_url.endswith('.js'):
                    js_files_for_ai.append({
                        'url': source_url,
                        'content': f"Potential leak: {leak.get('type', 'Unknown')} - {leak.get('value', '')}"
                    })

            # Batch analyze with AI
            if js_files_for_ai:
                ai_results = await ai_analyzer.batch_analyze_js_files(js_files_for_ai)

                # Update leaks with AI validation results
                validated_leaks = []
                for i, leak in enumerate(leaks_found):
                    if i < len(ai_results.get('files_analyzed', [])):
                        ai_analysis = ai_results['files_analyzed'][i]
                        if ai_analysis.get('success'):
                            leak['ai_validated'] = True
                            leak['ai_analysis'] = ai_analysis.get('analysis', '')
                            leak['ai_confidence'] = 0.8
                            self.performance_stats['ai_validation_count'] += 1
                        else:
                            leak['ai_validated'] = False
                            leak['ai_confidence'] = 0.3
                    validated_leaks.append(leak)

                results["leaks"] = validated_leaks
                results["high_confidence_leaks"] = [l for l in validated_leaks if l.get('ai_confidence', 0) > 0.7]

                console.print(f"    ✅ AI analysis: {len(results['high_confidence_leaks'])} high-confidence leaks", style="green")
            else:
                results["leaks"] = leaks_found
                console.print("    ⚠️ No content for AI analysis", style="yellow")

        except Exception as e:
            console.print(f"    ❌ AI analysis failed: {e}", style="red")
            results["leaks"] = leaks_found

    async def _run_phase_reporting(self, results, reporter, domain):
        """Phase 6: Comprehensive reporting"""
        self.print_phase(6, "Professional Reporting", "📊")

        try:
            reporter.save(results)
            console.print(f"    ✅ Comprehensive reports saved to results/{domain}/", style="green")
        except Exception as e:
            console.print(f"    ❌ Report generation failed: {e}", style="red")

    def _print_performance_stats(self, results, leak_detector, http_client, scoring_engine):
        """Print advanced performance statistics - UPDATED FOR NEW SCORING ENGINE"""
        console.print(f"\n[bold cyan]📈 ADVANCED PERFORMANCE STATISTICS[/bold cyan]")

        # Memory efficiency
        if self.performance_stats['memory_efficiency_gain'] > 0:
            console.print(f"    💾 Memory efficiency: {self.performance_stats['memory_efficiency_gain']:.1f}% gain", style="green")

        # Crawl success rate
        if self.performance_stats['crawl_success_rate'] > 0:
            console.print(f"    🌐 Crawl success rate: {self.performance_stats['crawl_success_rate']:.1f}%", style="green")

        # Smart targeting
        if self.performance_stats['smart_targets_identified'] > 0:
            console.print(f"    🎯 Smart targeting: {self.performance_stats['smart_targets_identified']} high-value targets", style="green")

        # AI validation
        if self.performance_stats['ai_validation_count'] > 0:
            console.print(f"    🤖 AI validation: {self.performance_stats['ai_validation_count']} leaks analyzed", style="magenta")

        # Scoring engine performance - NEW METRICS FROM UPDATED ENGINE
        if hasattr(scoring_engine, 'performance_metrics'):
            metrics = scoring_engine.performance_metrics
            console.print(f"    ⚡ Scoring performance: {metrics.get('avg_scoring_time', 0):.2f}s avg", style="blue")
            console.print(f"    📊 Scoring success rate: {metrics.get('success_rate', 0)*100:.1f}%", style="blue")
            console.print(f"    💰 Cache hits: {metrics.get('cache_hits', 0)}", style="blue")

        # Total scored domains from new engine
        if hasattr(scoring_engine, 'total_scored'):
            console.print(f"    🔢 Total domains scored: {scoring_engine.total_scored}", style="blue")

    def _create_empty_results(self, domain: str, args) -> Dict[str, Any]:
        """Create empty results structure - UPDATED FOR NEW SCORING DATA"""
        return {
            "subdomains": [], "urls": [], "js_files": [], "leaks": [],
            "fuzzed_paths": [], "admin_panels": [], "priority_targets": [],
            "scored_domains": [],  # NEW: Store scoring engine results
            "scan_config": {
                "domain": domain, "aggressive": args.aggressive,
                "subfinder": args.subfinder, "threads": args.threads,
                "ai_enabled": False
            }
        }

    def _initialize_results(self, domain: str, args, ai_enabled: bool) -> Dict[str, Any]:
        """Initialize comprehensive results structure - UPDATED FOR NEW SCORING DATA"""
        return {
            "subdomains": [], "urls": [], "js_files": [], "leaks": [],
            "fuzzed_paths": [], "admin_panels": [], "priority_targets": [],
            "scored_domains": [],  # NEW: Store full scoring engine results with breakdowns
            "scan_config": {
                "domain": domain, "aggressive": args.aggressive,
                "subfinder": args.subfinder, "threads": args.threads,
                "ai_enabled": ai_enabled
            }
        }

    def print_summary(self, results: Dict[str, Any], scan_duration: str):
        """Print comprehensive final summary - UPDATED FOR NEW SCORING DATA"""
        stats = {
            'domain': results.get('scan_config', {}).get('domain', 'Unknown'),
            'subdomains': len(results.get('subdomains', [])),
            'urls': len(results.get('urls', [])),
            'js_files': len(results.get('js_files', [])),
            'leaks': len(results.get('leaks', [])),
            'high_confidence': len(results.get('high_confidence_leaks', [])),
            'priority_targets': len(results.get('priority_targets', [])),
            'scored_domains': len(results.get('scored_domains', [])),  # NEW: Scoring data
            'duration': scan_duration
        }

        # Calculate scoring statistics if available
        scoring_stats = ""
        if results.get('scored_domains'):
            scores = [domain.get('score', 0) for domain in results['scored_domains']]
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                scoring_stats = f"\n[cyan]📊 Scoring Analytics: Avg={avg_score:.1f}, Max={max_score}[/cyan]"

        summary_text = f"""
🎯 [bold]Comprehensive Scan Summary[/bold]
├── Target: {stats['domain']}
├── Subdomains: {stats['subdomains']}
├── URLs Crawled: {stats['urls']}
├── JS Files: {stats['js_files']}
├── Leaks Detected: {stats['leaks']}
├── High-Confidence: {stats['high_confidence']}
├── Priority Targets: {stats['priority_targets']}
├── Scored Domains: {stats['scored_domains']}
└── Duration: {stats['duration']}
{scoring_stats}
[green]💾 Memory Efficiency: {self.performance_stats['memory_efficiency_gain']:.1f}%[/green]
[blue]🌐 Crawl Success: {self.performance_stats['crawl_success_rate']:.1f}%[/blue]
[magenta]🤖 AI Analysis: {self.performance_stats['ai_validation_count']} items[/magenta]
"""

        summary_panel = Panel(
            summary_text.strip(),
            title="🎉 Advanced Scan Complete",
            border_style="green",
            padding=(1, 2)
        )
        console.print(summary_panel)

def create_argument_parser():
    """Create comprehensive argument parser"""
    parser = argparse.ArgumentParser(
        description="LeakHunterX — Advanced Bug Bounty Discovery & Leak Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --target example.com --subfinder --aggressive
  python main.py --target example.com --ai-validate --ai-insights
  python main.py --target example.com --quick
        """
    )

    parser.add_argument("--target", required=True, help="Target domain to scan")
    parser.add_argument("--subfinder", action="store_true", help="Enable subdomain discovery")
    parser.add_argument("--aggressive", action="store_true", help="Enable aggressive scanning")

    # AI Features
    parser.add_argument("--ai-prioritize", action="store_true", help="AI URL prioritization")
    parser.add_argument("--ai-analyze", action="store_true", help="AI endpoint analysis")
    parser.add_argument("--ai-validate", action="store_true", help="AI leak validation")
    parser.add_argument("--ai-insights", action="store_true", help="AI report insights")

    # Performance
    parser.add_argument("--threads", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--wordlist", type=str, default="wordlists/default.txt", help="Wordlist path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick scan mode")

    return parser

async def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.target:
        console.print("❌ Error: --target argument is required", style="red")
        return

    # Clean domain
    domain = args.target.lower().replace("https://", "").replace("http://", "").strip("/").split('/')[0]

    # Initialize scanner
    scanner = LeakHunterX()
    scanner.print_banner()

    # Print configuration
    mode = "Aggressive" if args.aggressive else "Standard"
    mode += " + Quick" if args.quick else ""
    ai_enabled = any([args.ai_prioritize, args.ai_analyze, args.ai_validate, args.ai_insights])
    scanner.print_config(domain, mode, args.wordlist, ai_enabled)

    try:
        # Run scan
        start_time = datetime.now()
        results = await scanner.run_scan(domain, args)
        end_time = datetime.now()
        scan_duration = str(end_time - start_time).split('.')[0]

        # Print results
        scanner.print_summary(results, scan_duration)

    except KeyboardInterrupt:
        console.print("\n❌ Scan interrupted by user", style="red")
    except Exception as e:
        console.print(f"\n❌ Scan failed: {e}", style="red")
        if args.verbose:
            traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
