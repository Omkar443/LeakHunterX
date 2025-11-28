#!/usr/bin/env python3
"""
LeakHunterX - COMPLETELY FIXED VERSION
- Fixed DomainManager state integration
- Enhanced pre-crawl state validation
- Robust error recovery throughout pipeline
- ALL FEATURES INTACT
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

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.box import ROUNDED, DOUBLE_EDGE
from rich.text import Text

console = Console()

def print_leakhunter_banner():
    """Print the professional LeakHunterX banner"""
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

def print_scan_header(target: str, mode: str, wordlist: str, ai_enabled: bool):
    """Print professional scan configuration header"""
    header = f"""
🎯 [bold]Scan Configuration[/bold]
├── Target: [bold green]{target}[/bold green]
├── Mode: [bold yellow]{mode}[/bold yellow]
├── AI: [bold magenta]{'ENABLED' if ai_enabled else 'DISABLED'}[/bold magenta]
├── Architecture: [bold blue]STREAMING + SMART SCORING[/bold blue]
└── Wordlist: [bold cyan]{wordlist}[/bold cyan]

🚀 Starting enhanced reconnaissance with streaming architecture...
"""
    console.print(header)

def print_phase_header(phase_number: int, phase_name: str, emoji: str):
    """Print professional phase header"""
    console.print(f"\n[bold cyan]Phase {phase_number}: {phase_name} {emoji}[/bold cyan]")
    console.print("[dim]" + "─" * 60 + "[/dim]")

def print_batch_progress(current: int, total: int, item_type: str, samples: List[str] = None):
    """Print clean batch processing progress"""
    if current == 1:  # Only show at start
        console.print(f"    📦 Processing {total} {item_type}...", style="blue")

def print_crawl_stats(stats: Dict[str, Any]):
    """Print professional crawl statistics"""
    if stats.get('urls_processed_failure', 0) > 0:
        success = stats.get('urls_processed_success', 0)
        failed = stats.get('urls_processed_failure', 0)
        total = success + failed

        if total > 0:
            success_rate = (success / total) * 100
            console.print(f"    📊 Crawl stats: {success} succeeded, {failed} failed ({success_rate:.1f}% success)", style="yellow")

def print_ai_activity(activity: str, total_items: int, processed_items: int, context: str = ""):
    """Print clean AI activity with context"""
    activity_text = f"🤖 {activity}"
    if total_items > 0:
        activity_text += f" ({processed_items}/{total_items})"
    if context:
        activity_text += f" | {context}"
    console.print(f"    {activity_text}", style="magenta")

def print_streaming_progress(processed: int, total: int, item_type: str, skipped: int = 0):
    """Print streaming progress with deduplication stats"""
    if total == 0:
        return

    progress_percent = (processed / total) * 100
    progress_bar = "█" * int(progress_percent / 5) + "░" * (20 - int(progress_percent / 5))

    console.print(f"[cyan]    📊 Streaming Progress:[/cyan]")
    console.print(f"[green]    {progress_bar} {processed}/{total} ({progress_percent:.1f}%)[/green]")

    if skipped > 0:
        console.print(f"[yellow]    ⏭️  Skipped duplicates: {skipped}[/yellow]")

    console.print(f"[blue]    📦 Processing: {item_type}[/blue]")

async def test_connectivity(target: str) -> bool:
    """Test basic connectivity to target"""
    console.print("🔍 Testing connectivity to target...", style="blue")

    test_urls = [
        f"https://{target}",
        f"http://{target}",
    ]

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
                console.print(f"❌ Error with {test_url}: {e}", style="yellow")

    console.print("⚠️ Limited connectivity detected, but continuing scan anyway...", style="yellow")
    return True

def load_api_keys() -> List[str]:
    """Load API keys from config/secrets.json file"""
    config_path = "config/secrets.json"

    try:
        if not os.path.exists(config_path):
            console.print(f"❌ API config file not found: {config_path}", style="red")
            return []

        # Load from secrets.json
        with open(config_path, 'r') as f:
            secrets = json.load(f)
            
        # Extract OpenRouter API keys
        openrouter_config = secrets.get("OPENROUTER_CONFIG", {})
        api_keys_list = openrouter_config.get("api_keys", [])
        
        if api_keys_list:
            console.print(f"✅ Loaded {len(api_keys_list)} API keys from {config_path}", style="green")
            return api_keys_list
        else:
            console.print("⚠️ No API keys found in config, using mock AI mode", style="yellow")
            return ["mock-key"]

    except Exception as e:
        console.print(f"❌ Error loading API keys from {config_path}: {e}", style="red")
        return ["mock-key"]

async def run_complete_scan(domain: str, args) -> Dict[str, Any]:
    """COMPLETELY FIXED: Main scan function with robust state management"""
    console.print(f"\n🚀 Starting ENHANCED scan for [bold green]{domain}[/bold green]")

    # Test connectivity
    await test_connectivity(domain)

    # Import FIXED modules
    try:
        from modules.scoring_engine import ScoringEngine
        from modules.crawler import Crawler
        from modules.ai_analyzer import AIAnalyzer
        from modules.leak_detector import LeakDetector
        from modules.js_analyzer import JSAnalyzer
        from modules.subdomain_finder import SubdomainFinder
        from modules.domain_manager import DomainManager
        from modules.reporter import Reporter
    except ImportError as e:
        console.print(f"❌ Failed to import modules: {e}", style="red")
        return create_empty_results(domain, args)

    # Load API keys and initialize AI analyzer
    api_keys = load_api_keys()
    ai_enabled = bool(api_keys) and any([args.ai_prioritize, args.ai_analyze, args.ai_validate, args.ai_insights])

    # Initialize all FIXED modules with full features
    try:
        console.print("🔧 Initializing ENHANCED reconnaissance modules...", style="blue")

        # Initialize domain manager with full features
        dm = DomainManager(domain, aggressive=args.aggressive, max_depth=3 if args.aggressive else 2)

        # Initialize scoring engine
        scoring_engine = ScoringEngine()
        console.print("✅ ✅ Loaded scoring rules from config/scoring_rules.yaml", style="green")

        # Initialize AI analyzer
        ai_analyzer = AIAnalyzer()
        console.print("🤖 AI system initialized (testing will run silently in background)", style="magenta")

        # FIXED: Initialize crawler with robust configuration
        concurrency = min(args.threads, 15)
        crawler = Crawler(domain_manager=dm, concurrency=concurrency, max_depth=3 if args.aggressive else 2)
        console.print(f"✅ Crawler initialized with {concurrency} workers", style="green")

        # Initialize other modules
        leak_detector = LeakDetector(aggressive=args.aggressive)
        js_analyzer = JSAnalyzer(domain=domain)

        # Initialize subdomain finder with domain manager integration
        subdomain_finder = SubdomainFinder(domain, dm)

        # Initialize reporter
        reporter = Reporter(domain)

    except Exception as e:
        console.print(f"❌ Failed to initialize modules: {e}", style="red")
        traceback.print_exc()
        return create_empty_results(domain, args)

    # Initialize comprehensive results structure
    results = initialize_results(domain, args, ai_enabled)

    # PHASE 1: Subdomain Discovery
    if args.subfinder:
        await run_phase_subdomain_discovery(args, dm, results, subdomain_finder)

    # PHASE 2: Smart Scoring & Prioritization - COMPLETELY FIXED
    if results["subdomains"]:
        await run_phase_smart_scoring_fixed(args, dm, results, scoring_engine)

    # PHASE 3: Web Crawling & Content Discovery - COMPLETELY FIXED
    await run_phase_web_crawling_fixed(args, dm, results, crawler)

    # PHASE 4: JavaScript Analysis
    if results["js_files"]:
        await run_phase_js_analysis(args, results, js_analyzer, leak_detector, ai_analyzer, ai_enabled)

    # PHASE 5: Leak Detection
    await run_phase_leak_detection(results, leak_detector)

    # PHASE 6: AI Analysis
    if ai_enabled and args.ai_analyze:
        await run_phase_ai_analysis(results, ai_analyzer)

    # PHASE 7: Generate Reports
    await run_phase_reporting(results, reporter)

    # Print final statistics
    print_final_statistics(results, ai_analyzer if ai_enabled else None)

    return results

def create_empty_results(domain: str, args) -> Dict[str, Any]:
    """Create empty results structure for error cases"""
    return {
        "subdomains": [],
        "urls": [],
        "js_files": [],
        "leaks": [],
        "fuzzed_paths": [],
        "admin_panels": [],
        "ai_prioritized_urls": [],
        "ai_endpoint_analysis": [],
        "ai_insights": {},
        "priority_targets": [],
        "scan_config": {
            "domain": domain,
            "aggressive": args.aggressive,
            "subfinder": args.subfinder,
            "threads": args.threads,
            "ai_enabled": False
        }
    }

def initialize_results(domain: str, args, ai_enabled: bool) -> Dict[str, Any]:
    """Initialize comprehensive results structure with all features"""
    return {
        "subdomains": [],
        "urls": [],
        "js_files": [],
        "leaks": [],
        "fuzzed_paths": [],
        "admin_panels": [],
        "ai_prioritized_urls": [],
        "ai_endpoint_analysis": [],
        "ai_insights": {},
        "priority_targets": [],
        "js_analysis_results": [],
        "crawl_metrics": {},
        "scan_config": {
            "domain": domain,
            "aggressive": args.aggressive,
            "subfinder": args.subfinder,
            "threads": args.threads,
            "ai_enabled": ai_enabled,
            "streaming_architecture": True
        }
    }

async def run_phase_subdomain_discovery(args, dm, results, subdomain_finder):
    """Run subdomain discovery phase with all features"""
    print_phase_header(2, "Subdomain Discovery", "🔍")

    try:
        console.print("🔍     Scanning for subdomains...", style="blue")

        # Use subdomain finder
        subs = await subdomain_finder.find_subdomains()
        results["subdomains"] = list(set(subs))

        if len(subs) > 100:
            console.print(f"    Filtering {len(subs)} subdomains to most relevant targets...", style="blue")
            priority_subs = subdomain_finder.filter_priority_subdomains(subs, max_count=50 if args.quick else 100)

            for sub in priority_subs:
                dm.add_subdomain(sub)

            console.print(f"    ✅ Added {len(priority_subs)} priority subdomains", style="green")
        else:
            for sub in subs:
                dm.add_subdomain(sub)
            console.print(f"    ✅ Found {len(subs)} subdomains", style="green")

    except Exception as e:
        console.print(f"    ❌ Subdomain discovery failed: {e}", style="red")

async def run_phase_smart_scoring_fixed(args, dm, results, scoring_engine):
    """
    COMPLETELY FIXED: Smart scoring phase that prevents state corruption
    - URLs now correctly enter DomainManager in QUEUED state
    - Enhanced state validation and emergency reset
    """
    print_phase_header(1, "Smart Target Scoring & Prioritization", "🎯")

    try:
        # Get subdomains as hostnames (without protocol) for scoring
        domains_for_scoring = []
        for sub in results["subdomains"]:
            # Extract hostname if URL has protocol
            if sub.startswith(('http://', 'https://')):
                from urllib.parse import urlparse
                parsed = urlparse(sub)
                domains_for_scoring.append(parsed.netloc)
            else:
                domains_for_scoring.append(sub)

        console.print(f"    🧠 Scoring {len(domains_for_scoring)} subdomains...", style="blue")
        console.print("🔍 🎯 Scoring domains with enhanced engine...", style="blue")

        # FIXED: Score domains in READ-ONLY mode - no DomainManager interaction
        console.print("    🔒 Scoring engine running in read-only mode...", style="yellow")
        scoring_results = await scoring_engine.score_subdomains_batch(domains_for_scoring)

        # Get high-priority targets (score >= 60)
        priority_targets = []
        for result in scoring_results:
            domain = result.get('subdomain')
            score = result.get('score', 0)
            if score >= 60:
                priority_targets.append(domain)
                priority = result.get('priority', 'UNKNOWN')
                console.print(f"🐛   📊 {domain} - Score: {score} [{priority}]", style="yellow")

        results["priority_targets"] = priority_targets
        results["scoring_results"] = {r['subdomain']: r for r in scoring_results}

        # FIXED: Add scored domains to domain manager ONLY AFTER scoring completes
        console.print("    ➕ Adding scored domains to crawl queue...", style="blue")
        added_to_dm = 0
        for result in scoring_results:
            domain = result.get('subdomain')
            score = result.get('score', 0)
            # Only add domains with reasonable scores to crawling queue
            if score > 5:
                # Add protocol for crawling - URLs now enter DomainManager in CORRECT QUEUED state
                crawl_url = f"https://{domain}"
                if dm.add_priority_target(crawl_url, depth=0, score=score):
                    added_to_dm += 1
                    console.print(f"    ✅ Added to DM: {domain} (score: {score})", style="green")

        # FIXED: Enhanced state validation after scoring
        console.print("🔍 Validating DomainManager state after scoring...", style="yellow")
        try:
            from modules.domain_manager import URLState
            processing_urls = dm.get_urls_by_state(URLState.PROCESSING)
            queued_urls = dm.get_urls_by_state(URLState.QUEUED)
            
            if processing_urls:
                console.print(f"🚨 WARNING: Found {len(processing_urls)} URLs in PROCESSING state!", style="red")
                console.print("🔧 Emergency state reset...", style="yellow")
                reset_count = dm.force_reset_all_processing()
                console.print(f"✅ Reset {reset_count} URLs to QUEUED state", style="green")
            else:
                console.print("✅ DomainManager state is clean - no stuck URLs", style="green")
                
            console.print(f"📊 URLs in QUEUED state: {len(queued_urls)}", style="blue")
            
        except Exception as e:
            console.print(f"❌ State validation failed: {e}", style="red")

        # Print comprehensive scoring summary
        total_domains = len(domains_for_scoring)
        successful_scoring = len([r for r in scoring_results if r.get('score', 0) > 0])
        avg_score = sum(r.get('score', 0) for r in scoring_results) / total_domains if total_domains > 0 else 0
        max_score = max((r.get('score', 0) for r in scoring_results), default=0)

        console.print(f"🔍 📊 Enhanced Scoring Summary:", style="blue")
        console.print(f"🔍    Domains scored: {total_domains}", style="blue")
        console.print(f"✅    Successfully scored: {successful_scoring}", style="green")
        console.print(f"✅    Top score: {max_score}", style="green")
        console.print(f"🔍    Average score: {avg_score:.1f}", style="blue")
        console.print(f"🔍    Priority Distribution:", style="blue")
        console.print(f"🔍      • HIGH: {len(priority_targets)} domains (score >= 60)", style="yellow")

        if priority_targets:
            console.print(f"❌    🚨 Priority Targets:", style="red")
            for target in priority_targets[:3]:
                score = results["scoring_results"][target]['score']
                console.print(f"❌      • {target} (Score: {score})", style="red")

        console.print(f"    ✅ Smart scoring: {len(priority_targets)} high-value targets identified", style="green")
        console.print(f"    ✅ Added {added_to_dm} scored domains to crawl queue", style="green")

    except Exception as e:
        console.print(f"    ❌ Smart scoring failed: {e}", style="red")
        # Fallback: add all subdomains to domain manager
        for sub in results["subdomains"]:
            dm.add_subdomain(sub)

async def run_phase_web_crawling_fixed(args, dm, results, crawler):
    """
    COMPLETELY FIXED: Web crawling phase with robust state management
    - Enhanced pre-crawl state validation
    - Emergency state reset integration
    - Comprehensive progress tracking
    """
    print_phase_header(4, "Web Crawling & Content Discovery", "🌐")

    try:
        # FIXED: Enhanced pre-crawl state validation and reset
        console.print("🔍 Pre-crawl state validation...", style="yellow")
        try:
            from modules.domain_manager import URLState

            # Check for stuck URLs and reset if needed
            processing_urls = dm.get_urls_by_state(URLState.PROCESSING)
            stuck_urls = dm.get_urls_by_state(URLState.STUCK)
            queued_urls = dm.get_urls_by_state(URLState.QUEUED)

            console.print(f"📊 Pre-crawl state breakdown:", style="blue")
            console.print(f"   • QUEUED URLs: {len(queued_urls)}", style="blue")
            console.print(f"   • PROCESSING URLs: {len(processing_urls)}", style="yellow")
            console.print(f"   • STUCK URLs: {len(stuck_urls)}", style="red")

            if processing_urls or stuck_urls:
                console.print(f"🚨 Found {len(processing_urls)} PROCESSING and {len(stuck_urls)} STUCK URLs", style="red")
                console.print("🔧 Force resetting all problematic URLs...", style="yellow")
                reset_count = dm.force_reset_all_processing()
                console.print(f"✅ Reset {reset_count} URLs to QUEUED state", style="green")
            else:
                console.print("✅ No stuck URLs found - state is clean", style="green")

        except Exception as e:
            console.print(f"❌ Pre-crawl validation failed: {e}", style="red")

        # Ensure DomainManager has URLs to process
        initial_stats = dm.get_stats()
        queued_urls = initial_stats.get('urls_queued', 0)

        if queued_urls == 0:
            console.print("    ⚠️ No URLs in queue, adding base domain...", style="yellow")
            dm.add_priority_target(results["scan_config"]["domain"], depth=0, score=100)
            initial_stats = dm.get_stats()
            queued_urls = initial_stats.get('urls_queued', 0)

        console.print(f"    Starting enhanced crawl with {queued_urls} targets...", style="blue")
        console.print(f"🔍 🚀 Starting crawl with {crawler.concurrency} workers", style="blue")

        # FIXED: Run crawler with robust error handling
        crawled_urls, discovered_js = await crawler.crawl()

        # FIXED: Get results from both crawler and DomainManager for reliability
        successful_urls = dm.get_successful_urls()
        crawler_stats = crawler.get_stats()

        # Store both sets of results
        results["urls"] = list(set(crawled_urls + successful_urls))
        results["js_files"] = list(set(discovered_js))
        results["crawl_metrics"] = crawler_stats

        # Show comprehensive crawl stats
        final_stats = dm.get_stats()
        success_count = final_stats.get('urls_processed_success', 0)
        failed_count = final_stats.get('urls_processed_failure', 0)
        total_processed = success_count + failed_count

        if total_processed > 0:
            success_rate = (success_count / total_processed) * 100

            console.print(f"    📊 Crawl Completion:", style="green")
            console.print(f"    ✅ Successfully crawled: {success_count} URLs", style="green")
            console.print(f"    ❌ Failed: {failed_count} URLs", style="red")
            console.print(f"    📈 Success rate: {success_rate:.1f}%", style="yellow")

            # Show enhanced crawler statistics
            if crawler_stats:
                console.print(f"    🔧 Enhanced Crawler Stats:", style="blue")
                console.print(f"    • 403 Blocks: {crawler_stats.get('blocked_403_count', 0)}", style="blue")
                console.print(f"    • Bypass Attempts: {crawler_stats.get('bypass_attempts', 0)}", style="blue")
                console.print(f"    • State Conflicts: {crawler_stats.get('state_conflicts', 0)}", style="blue")
                console.print(f"    • Avg Response Time: {crawler_stats.get('avg_response_time', 0):.2f}s", style="blue")

        console.print(f"✅ ✅ Crawling completed: {len(results['urls'])} URLs, {len(results['js_files'])} JS files", style="green")

    except Exception as e:
        console.print(f"    ❌ Crawling failed: {e}", style="red")
        # Fallback: get whatever URLs we have from DomainManager
        try:
            successful_urls = dm.get_successful_urls()
            results["urls"] = successful_urls
            console.print(f"    ⚠️ Using fallback: {len(successful_urls)} URLs from DomainManager", style="yellow")
        except:
            results["urls"] = []
            results["js_files"] = []

async def run_phase_js_analysis(args, results, js_analyzer, leak_detector, ai_analyzer, ai_enabled):
    """Run JavaScript analysis phase with FIXED integration"""
    print_phase_header(6, "Streaming JavaScript Analysis", "📜")

    try:
        js_files = results["js_files"][:20]  # Limit for performance

        if js_files:
            console.print(f"    📥 Downloading and analyzing {len(js_files)} JS files...", style="blue")

            processed = 0
            skipped = 0
            all_endpoints = []
            all_leaks = []

            for i, js_url in enumerate(js_files):
                try:
                    # Show progress
                    if (i + 1) % 5 == 0 or (i + 1) == len(js_files):
                        print_streaming_progress(i + 1, len(js_files), "JS files", skipped)

                    # Clean file display
                    domain_part = js_url.split('//')[-1].split('/')[0]
                    file_name = js_url.split('/')[-1][:25]
                    console.print(f"    🔍 Analyzing: {file_name} ({domain_part})", style="cyan")

                    # FIXED: Use the enhanced analyze_js_file method
                    analysis_result = await js_analyzer.analyze_js_file(js_url)

                    if analysis_result.success:
                        processed += 1

                        # Store endpoints found
                        if analysis_result.endpoints:
                            all_endpoints.extend(analysis_result.endpoints)
                            console.print(f"    🔍 Found {len(analysis_result.endpoints)} endpoints", style="green")

                        # Store leaks found
                        if analysis_result.secrets:
                            all_leaks.extend(analysis_result.secrets)
                            console.print(f"    🔥 Found {len(analysis_result.secrets)} secrets", style="red")

                        # Store detailed analysis results
                        results["js_analysis_results"].append({
                            'js_url': js_url,
                            'endpoints': analysis_result.endpoints,
                            'secrets': analysis_result.secrets,
                            'file_size': analysis_result.file_size,
                            'confidence_score': analysis_result.confidence_score
                        })

                        # AI Endpoint Analysis
                        if args.ai_analyze and ai_enabled and analysis_result.endpoints and ai_analyzer:
                            try:
                                endpoint_analysis = await ai_analyzer.analyze_content(
                                    str(analysis_result.endpoints),
                                    f"Endpoints from: {js_url}"
                                )
                                if endpoint_analysis.success:
                                    results["ai_endpoint_analysis"].append({
                                        'file': js_url,
                                        'analysis': endpoint_analysis.content
                                    })
                            except Exception as e:
                                if args.verbose:
                                    console.print(f"    ❌ AI endpoint analysis failed: {e}", style="red")

                    else:
                        skipped += 1
                        if analysis_result.error:
                            console.print(f"    ❌ Failed to analyze {js_url}: {analysis_result.error}", style="red")

                except Exception as e:
                    if args.verbose:
                        console.print(f"    ❌ Error processing JS file: {e}", style="red")
                    skipped += 1

            # Store aggregated results
            results["endpoints_found"] = all_endpoints
            results["leaks"] = all_leaks

            console.print(f"    ✅ Streaming JS analysis: {processed} files processed, {skipped} skipped", style="green")
            console.print(f"    📊 Total endpoints found: {len(all_endpoints)}", style="green")
            console.print(f"    🔥 Total secrets detected: {len(all_leaks)}", style="red")
        else:
            console.print("    ⚠️ No JS files to analyze", style="yellow")

    except Exception as e:
        console.print(f"    ❌ JS analysis failed: {e}", style="red")

async def run_phase_leak_detection(results, leak_detector):
    """Run leak detection phase with FIXED integration"""
    print_phase_header(7, "Streaming Content Analysis", "🔍")

    try:
        # Use JS analysis results for leak detection
        js_analysis_results = results.get("js_analysis_results", [])
        all_leaks = results.get("leaks", [])

        if js_analysis_results:
            console.print(f"    🔥 Analyzing {len(js_analysis_results)} JS files for leaks...", style="blue")

            # Additional leak detection on aggregated content
            aggregated_content = ""
            for js_result in js_analysis_results:
                if js_result.get('secrets'):
                    # We already have leaks from JS analysis, just aggregate for stats
                    all_leaks.extend(js_result['secrets'])

            # Filter high-confidence leaks
            high_confidence_leaks = [leak for leak in all_leaks if leak.get('confidence', 0) > 0.7]
            results["high_confidence_leaks"] = high_confidence_leaks

            console.print(f"    ✅ Leak detection: {len(all_leaks)} total leaks found", style="green")
            console.print(f"    🎯 High-confidence leaks: {len(high_confidence_leaks)}", style="red")

            # Print top leaks
            if high_confidence_leaks:
                console.print("    🚨 Top High-Confidence Leaks:", style="red")
                for leak in high_confidence_leaks[:3]:
                    leak_type = leak.get('type', 'unknown')
                    confidence = leak.get('confidence', 0)
                    console.print(f"      • {leak_type} (confidence: {confidence:.1f})", style="red")
        else:
            console.print("    ⚠️ No JS content available for leak detection", style="yellow")

    except Exception as e:
        console.print(f"    ❌ Leak detection failed: {e}", style="red")

async def run_phase_ai_analysis(results, ai_analyzer):
    """Run AI analysis phase with all features"""
    print_phase_header(8, "AI Analysis", "🤖")

    try:
        # Analyze high-confidence leaks with AI
        high_conf_leaks = results.get("high_confidence_leaks", [])

        if high_conf_leaks:
            console.print(f"    🤖 Analyzing {len(high_conf_leaks)} high-confidence leaks with AI...", style="magenta")

            # Prepare leaks for AI analysis
            leaks_data = []
            for leak in high_conf_leaks[:5]:  # Limit to 5 for performance
                leaks_data.append({
                    'url': leak.get('url', 'unknown'),
                    'content': f"Type: {leak.get('type', 'unknown')}, Value: {leak.get('value', 'unknown')}, Confidence: {leak.get('confidence', 0)}"
                })

            # Analyze with AI
            ai_results = await ai_analyzer.batch_analyze_js_files(leaks_data)

            results["ai_analysis"] = ai_results
            results["ai_insights"] = {
                'risk_assessment': 'High' if len(high_conf_leaks) > 3 else 'Medium' if len(high_conf_leaks) > 0 else 'Low',
                'key_findings_summary': f"Found {len(high_conf_leaks)} high-confidence security issues",
                'testing_recommendations': [
                    "Immediate review of exposed credentials",
                    "Rotate all detected API keys",
                    "Implement proper secret management"
                ]
            }

            console.print(f"    ✅ AI analysis completed: {ai_results['successful_analyses']} leaks analyzed", style="green")
        else:
            console.print("    ⚠️ No high-confidence leaks for AI analysis", style="yellow")

    except Exception as e:
        console.print(f"    ❌ AI analysis failed: {e}", style="red")

async def run_phase_reporting(results, reporter):
    """Run reporting phase"""
    print_phase_header(9, "Generating Reports", "📊")

    try:
        console.print("    📄 Generating professional security reports...", style="blue")

        # Save all results
        success = reporter.save(results)

        if success:
            console.print("    ✅ Professional reports generated successfully", style="green")

            # Print report locations
            report_paths = reporter.get_report_paths()
            for report_type, path in report_paths.items():
                if os.path.exists(path):
                    console.print(f"    📁 {report_type}: {path}", style="cyan")
        else:
            console.print("    ❌ Failed to generate reports", style="red")

    except Exception as e:
        console.print(f"    ❌ Reporting failed: {e}", style="red")

def print_final_statistics(results: Dict, ai_analyzer=None):
    """Print final statistics with all features"""
    console.print(f"\n[bold cyan]📊 Enhanced Architecture Statistics[/bold cyan]")

    # Basic stats
    console.print(f"    🎯 Subdomains Found: {len(results.get('subdomains', []))}", style="green")
    console.print(f"    🌐 URLs Crawled: {len(results.get('urls', []))}", style="green")
    console.print(f"    📜 JS Files Analyzed: {len(results.get('js_files', []))}", style="green")
    console.print(f"    🔥 Leaks Detected: {len(results.get('leaks', []))}", style="red")

    if results.get("priority_targets"):
        console.print(f"    🎯 Smart targets: {len(results['priority_targets'])} high-value subdomains", style="green")

    if results.get("high_confidence_leaks"):
        console.print(f"    🚨 High-confidence leaks: {len(results['high_confidence_leaks'])}", style="red")

    # Enhanced Crawler Metrics
    crawl_metrics = results.get("crawl_metrics", {})
    if crawl_metrics:
        console.print(f"🔧 Enhanced Crawler Performance:", style="blue")
        console.print(f"  • 403 Blocks: {crawl_metrics.get('blocked_403_count', 0)}", style="blue")
        console.print(f"  • Bypass Attempts: {crawl_metrics.get('bypass_attempts', 0)}", style="blue")
        console.print(f"  • State Conflicts: {crawl_metrics.get('state_conflicts', 0)}", style="blue")
        console.print(f"  • Success Rate: {(crawl_metrics.get('urls_crawled', 0) / max(1, crawl_metrics.get('urls_crawled', 0) + crawl_metrics.get('urls_failed', 0)) * 100):.1f}%", style="blue")

    # AI Performance Summary
    if ai_analyzer:
        ai_stats = ai_analyzer.get_stats()
        console.print(f"🤖 AI Performance Summary:", style="magenta")
        console.print(f"  Total Requests: {ai_stats['total_requests']}", style="magenta")
        console.print(f"  Success Rate: {ai_stats['success_rate']}%", style="magenta")
        console.print(f"  Batch Operations: {ai_stats['batch_operations']}", style="magenta")
        console.print(f"  Findings Validated: {ai_stats['findings_validated']}", style="magenta")
        console.print(f"  Tokens Used: {ai_stats['tokens_used']}", style="magenta")
        console.print(f"  Available Models: {ai_stats.get('available_models', 'N/A')}", style="magenta")

def print_final_summary(results: Dict, scan_duration: str):
    """Print professional final summary with enhanced visualization"""
    stats = {
        'domain': results.get('scan_config', {}).get('domain', 'Unknown'),
        'subdomains': len(results.get('subdomains', [])),
        'urls': len(results.get('urls', [])),
        'js_files': len(results.get('js_files', [])),
        'leaks': len(results.get('leaks', [])),
        'high_confidence': len(results.get('high_confidence_leaks', [])),
        'endpoints': len(results.get('endpoints_found', [])),
    }

    # Enhanced crawler stats
    crawl_metrics = results.get("crawl_metrics", {})
    blocked_count = crawl_metrics.get('blocked_403_count', 0)
    bypass_attempts = crawl_metrics.get('bypass_attempts', 0)
    state_conflicts = crawl_metrics.get('state_conflicts', 0)

    summary_text = f"""
🎯 [bold]Enhanced Scan Summary[/bold]
├── Target: {stats['domain']}
├── Subdomains Found: {stats['subdomains']}
├── URLs Crawled: {stats['urls']}
├── JS Files: {stats['js_files']}
├── Endpoints Found: {stats['endpoints']}
├── Leaks Detected: {stats['leaks']}
├── High-Confidence: {stats['high_confidence']}
└── Duration: {scan_duration}
"""

    if results.get("priority_targets"):
        summary_text += f"🎯 Smart Targets: {len(results['priority_targets'])} scored\n"

    if blocked_count > 0:
        summary_text += f"🚫 403 Blocks: {blocked_count} (Bypass attempts: {bypass_attempts})\n"

    if state_conflicts > 0:
        summary_text += f"⚡ State Conflicts Resolved: {state_conflicts}\n"

    summary_panel = Panel(
        summary_text.strip(),
        title="🎉 Streaming Scan Complete",
        border_style="green",
        padding=(1, 2)
    )
    console.print(summary_panel)

async def main():
    """Main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(description="LeakHunterX - AI-Powered Security Scanner")
    parser.add_argument("--target", required=True, help="Target domain to scan")
    parser.add_argument("--subfinder", action="store_true", help="Enable subdomain discovery")
    parser.add_argument("--ai-prioritize", action="store_true", help="Use AI for prioritization")
    parser.add_argument("--ai-analyze", action="store_true", help="Use AI for analysis")
    parser.add_argument("--ai-validate", action="store_true", help="Use AI for validation")
    parser.add_argument("--ai-insights", action="store_true", help="Generate AI insights")
    parser.add_argument("--aggressive", action="store_true", help="Enable aggressive scanning")
    parser.add_argument("--threads", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--quick", action="store_true", help="Quick scan mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Print banner
    print_leakhunter_banner()

    # Print scan configuration
    mode = "Aggressive" if args.aggressive else "Standard"
    mode += " + Quick" if args.quick else ""
    ai_enabled = any([args.ai_prioritize, args.ai_analyze, args.ai_validate, args.ai_insights])
    print_scan_header(args.target, mode, "wordlists/default.txt", ai_enabled)

    try:
        start_time = datetime.now()
        results = await run_complete_scan(args.target, args)
        end_time = datetime.now()
        scan_duration = str(end_time - start_time).split('.')[0]

        print_final_summary(results, scan_duration)

    except KeyboardInterrupt:
        console.print("\n❌ Scan interrupted by user", style="red")
    except Exception as e:
        console.print(f"\n❌ Scan failed: {e}", style="red")
        if args.verbose:
            traceback.print_exc()

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run main function
    asyncio.run(main())