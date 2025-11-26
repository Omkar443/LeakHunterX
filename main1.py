#!/usr/bin/env python3
"""
LeakHunterX - Industry-Grade AI-Powered Security Scanner
ENHANCED Main Controller with Streaming Architecture & Smart Scoring
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

def print_phase_header_main(phase_number: int, phase_name: str, emoji: str):
    """Print professional phase header - MAIN VERSION"""
    console.print(f"\n[bold cyan]Phase {phase_number}: {phase_name} {emoji}[/bold cyan]")
    console.print("[dim]" + "─" * 60 + "[/dim]")

def print_batch_progress_main(current: int, total: int, item_type: str, samples: List[str] = None):
    """Print clean batch processing progress"""
    if current == 1:  # Only show at start
        console.print(f"    📦 Processing {total} {item_type}...", style="blue")

def print_crawl_stats_main(stats: Dict[str, Any]):
    """Print professional crawl statistics - ENHANCED with exact state tracking"""
    if stats.get('urls_processed_failure', 0) > 0:
        success = stats.get('urls_processed_success', 0)
        failed = stats.get('urls_processed_failure', 0)
        total = success + failed
        
        if total > 0:
            success_rate = (success / total) * 100
            console.print(f"    📊 Crawl stats: {success} succeeded, {failed} failed ({success_rate:.1f}% success)", style="yellow")

def print_ai_activity_main(activity: str, total_items: int, processed_items: int, context: str = ""):
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

def print_final_summary_main(results: Dict[str, Any], scan_duration: str):
    """Print professional final summary - ENHANCED with streaming metrics"""
    stats = {
        'domain': results.get('scan_config', {}).get('domain', 'Unknown'),
        'subdomains': len(results.get('subdomains', [])),
        'urls': len(results.get('urls', [])),
        'js_files': len(results.get('js_files', [])),
        'leaks': len(results.get('leaks', [])),
        'high_confidence_leaks': len(results.get('high_confidence_leaks', [])),
        'duration': scan_duration,
        'ai_prioritized': len(results.get('ai_prioritized_urls', [])),
        'priority_targets': len(results.get('priority_targets', []))
    }

    # Enhanced risk assessment
    critical_leaks = len([l for l in results.get('leaks', []) if l.get('severity') == 'CRITICAL'])
    high_leaks = len([l for l in results.get('leaks', []) if l.get('severity') == 'HIGH'])
    validated_leaks = len([l for l in results.get('leaks', []) if l.get('ai_validated', False)])

    summary_text = f"""
🎯 [bold]Enhanced Scan Summary[/bold]
├── Target: {stats['domain']}
├── Subdomains Found: {stats['subdomains']}
├── URLs Crawled: {stats['urls']}
├── JS Files: {stats['js_files']}
├── Leaks Detected: {stats['leaks']}
├── High-Confidence: {stats['high_confidence_leaks']}
├── Critical Issues: {critical_leaks}
├── High Priority: {high_leaks}
└── Duration: {stats['duration']}
"""

    if stats['ai_prioritized'] > 0:
        summary_text += f"📈 AI-Prioritized: {stats['ai_prioritized']} targets\n"
    if stats['priority_targets'] > 0:
        summary_text += f"🎯 Smart Targets: {stats['priority_targets']} scored\n"

    summary_panel = Panel(
        summary_text.strip(),
        title="🎉 Streaming Scan Complete",
        border_style="green",
        padding=(1, 2)
    )
    console.print(summary_panel)

def create_enhanced_results_table():
    """Create enhanced results table"""
    results_table = Table(
        title="🔍 LeakHunterX - Enhanced Scan Results",
        title_style="bold red",
        header_style="bold cyan",
        box=DOUBLE_EDGE,
        show_header=True,
        show_lines=True
    )

    results_table.add_column("#", style="dim", width=4)
    results_table.add_column("Subdomain", style="cyan", width=22)
    results_table.add_column("Endpoint / URL", style="green", width=28)
    results_table.add_column("Status", justify="center", width=8)
    results_table.add_column("Score", justify="center", width=10)
    results_table.add_column("Priority", justify="center", width=12)
    results_table.add_column("Confidence", justify="center", width=12)

    return results_table

def load_api_keys() -> List[str]:
    """Load OpenRouter API keys from config file with enhanced error handling"""
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

    except json.JSONDecodeError as e:
        console.print(f"❌ Invalid JSON in config file: {e}", style="red")
        return []
    except Exception as e:
        console.print(f"❌ Error loading config: {e}", style="red")
        return []

async def test_connectivity(target: str) -> bool:
    """Test basic connectivity to target with ROBUST error handling"""
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
                    # ANY response means server is reachable - even errors
                    console.print(f"✅ Server responded from {test_url} (Status: {response.status})", style="green")
                    return True
            except asyncio.TimeoutError:
                console.print(f"⏰ Timeout connecting to {test_url}", style="yellow")
            except aiohttp.ClientConnectorError as e:
                console.print(f"🔌 Connection failed to {test_url}: {e}", style="yellow")
            except aiohttp.ClientError as e:
                console.print(f"🌐 Network error with {test_url}: {e}", style="yellow")
            except Exception as e:
                console.print(f"❌ Unexpected error with {test_url}: {e}", style="red")

    console.print("⚠️ Limited connectivity detected, but continuing scan anyway...", style="yellow")
    return True  # Always continue scan

async def analyze_js_content_streaming(js_url: str, leak_detector, ai) -> List[Dict[str, Any]]:
    """ENHANCED: Analyze JavaScript content with streaming approach"""
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        timeout = aiohttp.ClientTimeout(total=20)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(js_url, ssl=False) as resp:
                if resp.status == 200:
                    js_content = await resp.text(errors='ignore')
                    
                    # Use streaming analysis (process and clear memory)
                    processed_data = leak_detector.process_js_file_streaming(js_url, js_content)
                    
                    leaks_found = []
                    for leak in processed_data.get("leaks", []):
                        leak["source_url"] = js_url
                        scored = ai.score(leak) if ai else leak
                        leaks_found.append(scored)
                    
                    return leaks_found

    except asyncio.TimeoutError:
        # Silent timeout - don't show individual timeouts
        pass
    except Exception as e:
        # Silent failure - don't show individual errors
        pass

    return []

async def analyze_url_content_streaming(url: str, leak_detector, ai) -> List[Dict[str, Any]]:
    """ENHANCED: Analyze URL content with streaming approach"""
    try:
        if not url.endswith(('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg')):
            connector = aiohttp.TCPConnector(ssl=False)
            timeout = aiohttp.ClientTimeout(total=15)

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url, ssl=False) as resp:
                    if resp.status == 200:
                        content = await resp.text(errors='ignore')
                        
                        # Use streaming content analysis
                        content_leaks = leak_detector.check_content(content, f"URL:{url}")

                        leaks_found = []
                        for leak in content_leaks:
                            leak["source_url"] = url
                            scored = ai.score(leak) if ai else leak
                            leaks_found.append(scored)
                        return leaks_found

    except asyncio.TimeoutError:
        pass  # Silent timeout for URL analysis
    except Exception:
        pass  # Silent failure for URL analysis

    return []

async def run_scan(domain: str, args) -> Dict[str, Any]:
    """ENHANCED: Main scan function with streaming architecture and smart scoring"""
    console.print(f"\n🚀 Starting ENHANCED scan for [bold green]{domain}[/bold green]")

    # Test connectivity but NEVER abort - always continue
    await test_connectivity(domain)

    # Import modules here to avoid circular imports
    try:
        from modules.domain_manager import DomainManager
        from modules.subdomain_finder import SubdomainFinder
        from modules.crawler import Crawler
        from modules.directory_fuzzer import DirectoryFuzzer
        from modules.js_analyzer import JSAnalyzer
        from modules.leak_detector import LeakDetector
        from modules.ai_analyzer import AIScoring
        from modules.reporter import Reporter
        from modules.utils import ensure_dir
        from modules.scoring_engine import ScoringEngine, PriorityLevel  # NEW: Smart scoring
        from modules.http_client import AsyncHTTPClient  # NEW: Enhanced HTTP client
    except ImportError as e:
        console.print(f"❌ Failed to import modules: {e}", style="red")
        return create_empty_results(domain, args)

    # Load API keys and initialize AI analyzer EARLY for background testing
    api_keys = load_api_keys()
    ai_enabled = bool(api_keys) and any([args.ai_prioritize, args.ai_analyze, args.ai_validate, args.ai_insights])

    if not api_keys and ai_enabled:
        console.print("⚠️ AI features requested but no API keys found", style="yellow")
        ai_enabled = False

    # Initialize AI analyzer EARLY but DON'T start testing yet
    ai_analyzer = None
    ai_testing_task = None

    if api_keys:
        console.print("🤖 AI system initialized (testing will run silently in background)", style="magenta")
        ai_analyzer = AIScoring(api_keys=api_keys)
        # Start background AI testing immediately
        ai_testing_task = ai_analyzer.start_silent_background_testing()

    # Initialize other modules with error handling
    try:
        console.print("🔧 Initializing ENHANCED reconnaissance modules...", style="blue")

        # NEW: Initialize enhanced HTTP client
        http_client = AsyncHTTPClient(timeout=15.0, max_retries=2, concurrency_limit=20)
        await http_client.init_dns_resolver()

        # NEW: Initialize smart scoring engine
        scoring_engine = ScoringEngine()

        # Initialize all modules while AI testing runs silently in background
        dm = DomainManager(domain, aggressive=args.aggressive, max_depth=3 if args.aggressive else 2)
        sd = SubdomainFinder(domain, dm, use_subfinder=args.subfinder)
        crawler = Crawler(dm, concurrency=min(args.threads, 10), max_depth=3 if args.aggressive else 2)

        # Initialize other modules only if needed
        fuzzer = None
        if args.aggressive:
            fuzzer = DirectoryFuzzer(domain, wordlist_path=args.wordlist, threads=min(args.threads, 5), aggressive=args.aggressive)

        js_engine = JSAnalyzer(domain)
        leak_detector = LeakDetector(aggressive=args.aggressive)
        reporter = Reporter(domain)

        # If no API keys but AI was requested, create a minimal AI analyzer
        if not ai_analyzer and ai_enabled:
            ai_analyzer = AIScoring(api_keys=[])

    except Exception as e:
        console.print(f"❌ Failed to initialize modules: {e}", style="red")
        return create_empty_results(domain, args)

    # Initialize results structure
    results = initialize_results(domain, args, ai_enabled)

    # STEP 1: SUBDOMAIN DISCOVERY - RUNS IMMEDIATELY WHILE AI TESTING CONTINUES
    if args.subfinder:
        await run_phase_1_subdomain_discovery(args, dm, results, sd)

    # NEW STEP: SMART SCORING & PRIORITIZATION
    if results["subdomains"]:
        await run_phase_smart_scoring(args, results, scoring_engine, fuzzer)

    # STEP 2: AGGRESSIVE FUZZING - NOW WITH SMART TARGETS
    if args.aggressive and fuzzer and results.get("priority_targets"):
        await run_phase_2_smart_fuzzing(args, dm, results, fuzzer, domain)

    # STEP 3: WEB CRAWLING & CONTENT DISCOVERY - WITH ENHANCED RELIABILITY
    await run_phase_3_web_crawling(args, dm, results, crawler)

    # Now check if AI testing is complete (it should be by this point)
    if ai_testing_task and not ai_testing_task.done():
        console.print("⏳ AI connection testing almost complete...", style="yellow")
        try:
            await asyncio.wait_for(ai_testing_task, timeout=2.0)
        except asyncio.TimeoutError:
            pass  # Don't care if it's not done, we'll proceed anyway

    # STEP 4: AI-POWERED PRIORITIZATION - NOW AI SHOULD BE READY
    if args.ai_prioritize and ai_enabled and results["urls"] and ai_analyzer:
        await run_phase_4_ai_prioritization(args, results, ai_analyzer, ai_enabled)

    # STEP 5: JAVASCRIPT ANALYSIS & SECRET DETECTION - WITH STREAMING
    leaks_found = await run_phase_5_js_analysis_streaming(args, results, js_engine, leak_detector, ai_analyzer, ai_enabled)

    # STEP 6: CONTENT ANALYSIS - WITH STREAMING
    await run_phase_6_content_analysis_streaming(results, leak_detector, ai_analyzer)

    # STEP 7: AI LEAK VALIDATION - WITH BATCH PROCESSING
    if ai_analyzer:
        await run_phase_7_ai_validation_enhanced(args, results, leaks_found, ai_analyzer, ai_enabled)

    # STEP 8: AI REPORT INSIGHTS - WITH COMPREHENSIVE REPORTING
    if args.ai_insights and ai_enabled and ai_analyzer:
        await run_phase_8_ai_insights_enhanced(results, ai_analyzer)

    # STEP 9: REPORTING
    await run_phase_9_reporting(results, reporter, domain)

    # Print enhanced statistics
    print_enhanced_stats(results, leak_detector, http_client, scoring_engine)

    # Print AI usage summary
    if ai_enabled and ai_analyzer:
        ai_analyzer.print_ai_summary()

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
        "priority_targets": [],  # NEW: Smart scoring results
        "scan_config": {
            "domain": domain,
            "aggressive": args.aggressive,
            "subfinder": args.subfinder,
            "threads": args.threads,
            "ai_enabled": False
        }
    }

def initialize_results(domain: str, args, ai_enabled: bool) -> Dict[str, Any]:
    """Initialize comprehensive results structure with streaming support"""
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
        "priority_targets": [],  # NEW: For smart scoring results
        "scan_config": {
            "domain": domain,
            "aggressive": args.aggressive,
            "subfinder": args.subfinder,
            "threads": args.threads,
            "ai_enabled": ai_enabled,
            "streaming_architecture": True  # NEW: Streaming mode enabled
        }
    }

async def run_phase_smart_scoring(args, results, scoring_engine, fuzzer):
    """NEW PHASE: Smart scoring and target prioritization"""
    print_phase_header_main(1, "Smart Target Scoring & Prioritization", "🎯")
    
    try:
        if results["subdomains"]:
            console.print(f"    🧠 Scoring {len(results['subdomains'])} subdomains...", style="blue")
            
            # Score all subdomains using the scoring engine
            scored_domains = await scoring_engine.score_subdomains_batch(
                results["subdomains"],
                max_concurrent=15,
                timeout_per_domain=10
            )
            
            # Get priority distribution
            priority_dist = scoring_engine.get_priority_distribution(scored_domains)
            
            # Select high-priority targets for fuzzing
            priority_targets = scoring_engine.select_priority_targets(scored_domains, max_targets=20)
            
            results["priority_targets"] = [target.subdomain for target in priority_targets]
            
            # Print scoring summary
            scoring_engine.print_enhanced_scoring_summary(scored_domains)
            
            console.print(f"    ✅ Smart scoring: {len(priority_targets)} high-value targets identified", style="green")
            
    except Exception as e:
        console.print(f"    ❌ Smart scoring failed: {e}", style="red")
        # Fallback: use all subdomains
        results["priority_targets"] = results["subdomains"][:20]

async def run_phase_1_subdomain_discovery(args, dm, results, sd):
    """Run subdomain discovery phase"""
    print_phase_header_main(2, "Subdomain Discovery", "🔍")
    try:
        subs = await sd.find_subdomains()
        results["subdomains"] = list(set(subs))

        if len(subs) > 100:
            console.print(f"    Filtering {len(subs)} subdomains to most relevant targets...", style="blue")
            priority_keywords = ['api', 'admin', 'app', 'web', 'www', 'mail', 'login', 'auth', 'dashboard']
            priority_subs = [s for s in subs if any(kw in s.lower() for kw in priority_keywords)]

            max_subs = 30 if args.quick else 50
            priority_subs = priority_subs[:max_subs]

            for sub in priority_subs:
                dm.add_subdomain(sub)

            console.print(f"    ✅ Added {len(priority_subs)} priority subdomains", style="green")
        else:
            for sub in subs:
                dm.add_subdomain(sub)
            console.print(f"    ✅ Found {len(subs)} subdomains", style="green")

    except Exception as e:
        console.print(f"    ❌ Subdomain discovery failed: {e}", style="red")

async def run_phase_2_smart_fuzzing(args, dm, results, fuzzer, domain):
    """ENHANCED: Run aggressive fuzzing with smart targets"""
    print_phase_header_main(3, "Smart Directory Fuzzing", "🚀")
    try:
        # Use priority targets for fuzzing
        if results.get("priority_targets"):
            console.print(f"    🎯 Fuzzing {len(results['priority_targets'])} priority targets...", style="blue")
            found_paths = await fuzzer.fuzz_multiple_targets(results["priority_targets"])
        else:
            # Fallback to single domain fuzzing
            base_url = f"https://{domain}"
            console.print("    🔐 Scanning for admin panels...", style="blue")
            admin_panels = await fuzzer.find_admin_panels(base_url)
            results["admin_panels"] = admin_panels

            console.print("    📁 Fuzzing directories...", style="blue")
            try:
                found_paths = await asyncio.wait_for(
                    fuzzer.fuzz_directories(base_url),
                    timeout=45.0
                )
            except asyncio.TimeoutError:
                console.print("    ⏰ Directory fuzzing timed out, continuing...", style="yellow")
                found_paths = []

        results["fuzzed_paths"] = found_paths

        # Add found paths to crawl queue
        for path in results["fuzzed_paths"]:
            if isinstance(path, dict) and 'url' in path:
                dm.add_discovered(path["url"])
        for panel in results["admin_panels"]:
            if isinstance(panel, dict) and 'url' in panel:
                dm.add_discovered(panel["url"])

        console.print(f"    ✅ Smart fuzzing completed: {len(results['fuzzed_paths'])} paths found", style="green")

    except Exception as e:
        console.print(f"    ❌ Smart fuzzing failed: {e}", style="red")

async def run_phase_3_web_crawling(args, dm, results, crawler):
    """Run web crawling phase with enhanced reliability"""
    print_phase_header_main(4, "Web Crawling & Content Discovery", "🌐")
    try:
        console.print(f"    Starting enhanced crawl with {dm.get_stats()['urls_queued']} targets...", style="blue")
        crawled_urls, discovered_js = await crawler.crawl()
        results["urls"] = list(set(crawled_urls))
        results["js_files"] = list(set(discovered_js))

        # Show professional crawl stats
        crawl_stats = crawler.get_stats()
        print_crawl_stats_main(crawl_stats)

        console.print(f"    ✅ Discovered: {len(crawled_urls)} URLs, {len(discovered_js)} JS files", style="green")

    except Exception as e:
        console.print(f"    ❌ Crawling failed: {e}", style="red")

async def run_phase_4_ai_prioritization(args, results, ai_analyzer, ai_enabled):
    """Run AI prioritization phase"""
    print_phase_header_main(5, "AI-Powered Target Prioritization", "🤖")
    try:
        # Check if AI is ready (background testing should be complete by now)
        console.print("    🔄 Checking AI connection status...", style="magenta")

        # This will be fast if background testing already completed
        if not await ai_analyzer.ensure_ai_working():
            console.print("    ❌ AI system not available, using fallback prioritization", style="yellow")
            all_urls = results["urls"] + results["js_files"] + [p["url"] for p in results["fuzzed_paths"] if isinstance(p, dict)]
            results["ai_prioritized_urls"] = ai_analyzer._fallback_prioritization(all_urls)
            return

        all_urls = results["urls"] + results["js_files"] + [p["url"] for p in results["fuzzed_paths"] if isinstance(p, dict)]

        if all_urls:
            print_ai_activity_main(
                "Analyzing URLs for security priorities",
                len(all_urls[:15]),
                0,
                f"Processing {len(all_urls[:15])} targets"
            )

            prioritized_urls = await ai_analyzer.prioritize_urls(all_urls, results["scan_config"]["domain"])

            # Ensure we have a list and handle the response properly
            if prioritized_urls and isinstance(prioritized_urls, list):
                results["ai_prioritized_urls"] = prioritized_urls

                if prioritized_urls:
                    high_priority = [p for p in prioritized_urls if isinstance(p, dict) and p.get('priority') == 'HIGH']
                    print_ai_activity_main(
                        "URL Prioritization Complete",
                        len(all_urls[:15]),
                        len(prioritized_urls),
                        f"High priority: {len(high_priority)} targets"
                    )
            else:
                console.print("    ⚠️ AI prioritization returned invalid data, using fallback", style="yellow")
                results["ai_prioritized_urls"] = ai_analyzer._fallback_prioritization(all_urls)

    except Exception as e:
        console.print(f"    ❌ AI prioritization failed: {e}", style="red")
        console.print("    🔄 Using fallback prioritization", style="yellow")
        all_urls = results["urls"] + results["js_files"] + [p["url"] for p in results["fuzzed_paths"] if isinstance(p, dict)]
        results["ai_prioritized_urls"] = ai_analyzer._fallback_prioritization(all_urls)

async def run_phase_5_js_analysis_streaming(args, results, js_engine, leak_detector, ai_analyzer, ai_enabled):
    """ENHANCED: Run JavaScript analysis with streaming approach"""
    print_phase_header_main(6, "Streaming JavaScript Analysis", "📜")
    leaks_found = []

    if results["js_files"]:
        # Show streaming progress
        console.print(f"    🌀 Streaming analysis of {len(results['js_files'])} JS files...", style="blue")

        js_files_to_analyze = results["js_files"][:20]  # Limit for performance
        processed = 0
        skipped = 0

        for i, js_url in enumerate(js_files_to_analyze):
            try:
                # Show progress
                if (i + 1) % 5 == 0 or (i + 1) == len(js_files_to_analyze):
                    print_streaming_progress(i + 1, len(js_files_to_analyze), "JS files", skipped)

                # Clean file display
                domain_part = js_url.split('//')[-1].split('/')[0]
                file_name = js_url.split('/')[-1][:25]
                console.print(f"    🔍 Analyzing: {file_name} ({domain_part})", style="cyan")

                # Use streaming JS analysis
                endpoints = await js_engine.extract_endpoints(js_url)

                if endpoints and i < 3:
                    console.print(f"    🔍 Found {len(endpoints)} endpoints", style="green")

                # AI Endpoint Analysis
                if args.ai_analyze and ai_enabled and endpoints and ai_analyzer:
                    try:
                        endpoint_analysis = await ai_analyzer.analyze_endpoints(endpoints, f"From: {js_url}")
                        if endpoint_analysis and isinstance(endpoint_analysis, list):
                            results["ai_endpoint_analysis"].extend(endpoint_analysis)
                    except Exception as e:
                        if args.verbose:
                            console.print(f"    ❌ AI endpoint analysis failed: {e}", style="red")

                # Use streaming leak detection
                content_leaks = await analyze_js_content_streaming(js_url, leak_detector, ai_analyzer)
                processed += 1

                # Score and add leaks
                for leak in content_leaks:
                    if isinstance(leak, dict):
                        leak["source_url"] = js_url
                        scored = ai_analyzer.score(leak) if ai_analyzer else leak
                        leaks_found.append(scored)

                        if scored.get("confidence", 0) > 0.7:
                            console.print(f"    🚨 LEAK: {scored['type']} - {scored['value'][:50]}...", style="red")

            except Exception as e:
                if args.verbose:  # Only show errors in verbose mode
                    console.print(f"    ❌ Error processing JS file: {e}", style="red")
                skipped += 1

        # Print streaming efficiency stats
        leak_stats = leak_detector.get_stats()
        if leak_stats.get('duplicates_skipped', 0) > 0:
            efficiency = (leak_stats['duplicates_skipped'] / 
                         (leak_stats['content_processed'] + leak_stats['duplicates_skipped'])) * 100
            console.print(f"    💾 Streaming efficiency: {efficiency:.1f}% duplicates skipped", style="green")

        console.print(f"    ✅ Streaming JS analysis: {processed} files processed, {skipped} skipped", style="green")

    return leaks_found

async def run_phase_6_content_analysis_streaming(results, leak_detector, ai_analyzer):
    """ENHANCED: Run content analysis with streaming approach"""
    print_phase_header_main(7, "Streaming Content Analysis", "🔍")
    try:
        analysis_tasks = []
        for url in results["urls"][:15]:  # Limit for performance
            task = analyze_url_content_streaming(url, leak_detector, ai_analyzer)
            analysis_tasks.append(task)

        if analysis_tasks:
            content_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            valid_results = [r for r in content_results if isinstance(r, list)]
            total_leaks = sum(len(r) for r in valid_results)

            console.print(f"    ✅ Streaming content analysis: {total_leaks} potential leaks found", style="green")
    except Exception as e:
        console.print(f"    ❌ Content analysis failed: {e}", style="red")

async def run_phase_7_ai_validation_enhanced(args, results, leaks_found, ai_analyzer, ai_enabled):
    """ENHANCED: Run AI validation with batch processing"""
    if args.ai_validate and ai_enabled and leaks_found and ai_analyzer:
        print_phase_header_main(8, "AI Batch Leak Validation", "🤖")
        try:
            print_ai_activity_main(
                "Batch validating detected leaks",
                len(leaks_found),
                0,
                f"Processing {len(leaks_found)} potential leaks"
            )

            # Use batch validation for efficiency
            validated_leaks = await ai_analyzer.batch_validate_findings(leaks_found)
            results["leaks"] = validated_leaks

            if validated_leaks:
                valid_count = len(validated_leaks)
                print_ai_activity_main(
                    "Batch Validation Complete",
                    len(leaks_found),
                    len(validated_leaks),
                    f"Validated: {valid_count} real leaks"
                )

        except Exception as e:
            console.print(f"    ❌ AI batch validation failed: {e}", style="red")
            results["leaks"] = ai_analyzer.rank_findings(leaks_found)
    else:
        # Use basic scoring if AI is not available
        if ai_analyzer:
            results["leaks"] = ai_analyzer.rank_findings(leaks_found)
        else:
            results["leaks"] = leaks_found

    # Filter high-confidence leaks
    if ai_analyzer:
        high_confidence_leaks = ai_analyzer.filter_high_confidence(results["leaks"])
        results["high_confidence_leaks"] = high_confidence_leaks
    else:
        results["high_confidence_leaks"] = [l for l in results["leaks"] if l.get("confidence", 0) >= 0.7]

async def run_phase_8_ai_insights_enhanced(results, ai_analyzer):
    """ENHANCED: Run AI insights with comprehensive reporting"""
    print_phase_header_main(9, "AI Comprehensive Report Insights", "🧠")
    try:
        print_ai_activity_main("Generating comprehensive security insights", 0, 0, "Analyzing scan results")
        ai_insights = await ai_analyzer.generate_comprehensive_report(results)
        results["ai_insights"] = ai_insights
        console.print(f"    ✅ AI comprehensive insights generated", style="green")
    except Exception as e:
        console.print(f"    ❌ AI insights generation failed: {e}", style="red")

async def run_phase_9_reporting(results, reporter, domain):
    """Run reporting phase"""
    print_phase_header_main(10, "Generating Professional Reports", "📊")
    try:
        reporter.save(results)
        console.print(f"    ✅ Professional reports saved to results/{domain}/", style="green")
    except Exception as e:
        console.print(f"    ❌ Report generation failed: {e}", style="red")

def print_enhanced_stats(results, leak_detector, http_client, scoring_engine):
    """Print enhanced statistics for streaming architecture"""
    console.print(f"\n[bold cyan]📊 Enhanced Architecture Statistics[/bold cyan]")
    
    # Leak detection stats
    leak_stats = leak_detector.get_stats()
    if leak_stats.get('duplicates_skipped', 0) > 0:
        efficiency = (leak_stats['duplicates_skipped'] / 
                     (leak_stats['content_processed'] + leak_stats['duplicates_skipped'])) * 100
        console.print(f"    💾 Memory efficiency: {efficiency:.1f}% duplicates skipped", style="green")
    
    # HTTP client stats
    http_stats = http_client.get_stats()
    if http_stats.get('success_rate_percent', 0) > 0:
        console.print(f"    🌐 HTTP success rate: {http_stats['success_rate_percent']}%", style="green")
    
    # Smart scoring stats
    if results.get('priority_targets'):
        console.print(f"    🎯 Smart targets: {len(results['priority_targets'])} high-value subdomains", style="green")

def print_final_results(results: Dict[str, Any], scan_duration: str):
    """Print professional final results with enhanced visualization"""
    print_final_summary_main(results, scan_duration)

    # Enhanced results table for AI-prioritized URLs
    if results.get('ai_prioritized_urls'):
        print_ai_prioritized_results(results)

    # Critical findings with better sorting
    if results.get('high_confidence_leaks'):
        print_critical_findings(results)

    # Enhanced AI insights
    if results.get('ai_insights') and isinstance(results['ai_insights'], dict):
        print_ai_insights(results)

def print_ai_prioritized_results(results: Dict[str, Any]):
    """Print AI-prioritized results in enhanced table"""
    results_table = create_enhanced_results_table()

    for i, item in enumerate(results['ai_prioritized_urls'][:15]):
        if not isinstance(item, dict):
            continue

        url = item.get('url', '')
        subdomain = url.split('//')[-1].split('/')[0] if '//' in url else url
        endpoint = '/' + '/'.join(url.split('/')[3:]) if len(url.split('/')) > 3 else '/'

        priority = item.get('priority', 'MEDIUM')
        priority_color = {
            'HIGH': 'red',
            'MEDIUM': 'yellow',
            'LOW': 'green'
        }.get(priority, 'white')

        score = item.get('score', 'N/A')
        confidence = f"{item.get('confidence', 0)*100:.0f}%"

        results_table.add_row(
            str(i + 1),
            subdomain[:21],
            endpoint[:27],
            "200",
            str(score),
            f"[{priority_color}]{priority}[/{priority_color}]",
            confidence
        )

    console.print(results_table)

def print_critical_findings(results: Dict[str, Any]):
    """Print critical security findings"""
    critical_leaks = sorted(
        results['high_confidence_leaks'],
        key=lambda x: (
            {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x.get('severity', 'LOW'), 0),
            x.get('confidence', 0)
        ),
        reverse=True
    )

    findings_table = Table(
        title="🚨 Critical Security Findings",
        title_style="bold red",
        header_style="bold yellow",
        box=ROUNDED,
        show_header=True,
        show_lines=True
    )

    findings_table.add_column("#", style="dim", width=4)
    findings_table.add_column("Type", style="cyan", width=16)
    findings_table.add_column("Value", style="white", width=35)
    findings_table.add_column("Severity", justify="center", width=10)
    findings_table.add_column("Confidence", justify="center", width=12)
    findings_table.add_column("Source", style="blue", width=20)

    for i, leak in enumerate(critical_leaks[:10]):
        severity = leak.get('severity', 'MEDIUM')
        severity_color = {
            'CRITICAL': 'red',
            'HIGH': 'bright_red',
            'MEDIUM': 'yellow',
            'LOW': 'green'
        }.get(severity, 'white')

        findings_table.add_row(
            str(i + 1),
            leak.get('type', 'Unknown')[:15],
            leak.get('value', '')[:34],
            f"[{severity_color}]{severity}[/{severity_color}]",
            f"{leak.get('confidence', 0)*100:.0f}%",
            leak.get('source_url', 'Unknown')[:19]
        )

    console.print(findings_table)

def print_ai_insights(results: Dict[str, Any]):
    """Print AI insights in enhanced panel"""
    insights = results['ai_insights']

    risk_color = {
        'CRITICAL': 'red',
        'HIGH': 'bright_red',
        'MEDIUM': 'yellow',
        'LOW': 'green'
    }.get(insights.get('risk_assessment', 'LOW'), 'white')

    insights_text = f"""
🧠 [bold]AI Security Assessment[/bold]
├── Risk Level: [{risk_color}]{insights.get('risk_assessment', 'N/A')}[/{risk_color}]
├── Key Findings: {insights.get('key_findings_summary', 'No significant findings')}
└── Recommendations: {', '.join(insights.get('testing_recommendations', ['Continue manual testing'])[:2])}
"""

    insights_panel = Panel(
        insights_text.strip(),
        title="🤖 AI Analysis",
        border_style="magenta",
        padding=(1, 2)
    )
    console.print(insights_panel)

# REST OF THE CODE REMAINS THE SAME (argument parsing, validation, etc.)
# Only the main scan function and related phases have been enhanced

async def main():
    """Main function with comprehensive error handling"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    if not validate_arguments(args):
        return

    # Apply quick mode settings
    if args.quick:
        apply_quick_mode(args)

    # Validate and clean domain
    domain = clean_domain(args.target)

    # Print banner and header
    print_leakhunter_banner()
    print_scan_configuration(domain, args)

    # Ensure wordlist exists
    if args.aggressive and not os.path.exists(args.wordlist):
        handle_missing_wordlist(args)

    try:
        start_time = datetime.now()
        results = await run_scan(domain, args)
        end_time = datetime.now()
        scan_duration = str(end_time - start_time).split('.')[0]  # Remove microseconds

        print_final_results(results, scan_duration)

    except KeyboardInterrupt:
        console.print("\n❌ Scan interrupted by user", style="red")
    except Exception as e:
        console.print(f"\n❌ Scan failed: {e}", style="red")
        if args.verbose:
            traceback.print_exc()

def create_argument_parser():
    """Create comprehensive argument parser"""
    parser = argparse.ArgumentParser(
        description="LeakHunterX — Advanced Bug Bounty Discovery & Leak Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --target example.com --subfinder
  python main.py --target example.com --aggressive --ai-prioritize
  python main.py --target example.com --quick --verbose
        """
    )

    # Target options
    parser.add_argument("--target", required=True, help="Single target domain to scan")
    parser.add_argument("--targets", help="File containing list of domains")

    # Discovery options - FIXED: default=False instead of True
    parser.add_argument("--subfinder", action="store_true", default=False,
                       help="Enable subdomain discovery (default: False)")
    parser.add_argument("--no-subfinder", action="store_false", dest="subfinder",
                       help="Disable subdomain discovery")
    parser.add_argument("--aggressive", action="store_true",
                       help="Enable aggressive scanning with directory fuzzing")

    # AI Features
    parser.add_argument("--ai-prioritize", action="store_true",
                       help="Use AI to prioritize URLs for fuzzing")
    parser.add_argument("--ai-analyze", action="store_true",
                       help="Use AI to analyze discovered endpoints")
    parser.add_argument("--ai-validate", action="store_true",
                       help="Use AI to validate detected leaks")
    parser.add_argument("--ai-insights", action="store_true",
                       help="Generate AI insights for final report")

    # Performance options
    parser.add_argument("--threads", type=int, default=10,
                       help="Concurrent workers (default: 10)")
    parser.add_argument("--timeout", type=int, default=15,
                       help="Request timeout in seconds (default: 15)")

    # Wordlist options
    parser.add_argument("--wordlist", type=str, default="wordlists/default.txt",
                       help="Path to wordlist file for fuzzing")

    # Output options
    parser.add_argument("--output", type=str, help="Custom output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick scan mode (limited scope)")

    return parser

def validate_arguments(args) -> bool:
    """Validate command line arguments"""
    if not args.target:
        console.print("❌ Error: --target argument is required", style="red")
        return False

    if args.targets and not os.path.exists(args.targets):
        console.print(f"❌ Error: Targets file not found: {args.targets}", style="red")
        return False

    return True

def apply_quick_mode(args):
    """Apply quick mode settings but preserve explicit AI flags - FIXED"""
    args.threads = 5
    args.aggressive = False

    # Only disable AI features if they weren't explicitly requested
    if not any([args.ai_prioritize, args.ai_analyze, args.ai_validate, args.ai_insights]):
        args.ai_prioritize = False
        args.ai_analyze = False
        args.ai_validate = False
        args.ai_insights = False
        console.print("⚡ Quick scan mode enabled (AI disabled)", style="yellow")
    else:
        console.print("⚡ Quick scan mode enabled (AI features preserved)", style="yellow")

def clean_domain(target: str) -> str:
    """Clean and validate domain"""
    domain = target.lower().replace("https://", "").replace("http://", "").strip("/")

    if '/' in domain:
        domain = domain.split('/')[0]

    return domain

def print_scan_configuration(domain: str, args):
    """Print scan configuration - FIXED"""
    mode = "Aggressive" if args.aggressive else "Standard"
    mode += " + Quick" if args.quick else ""

    # Show AI as ENABLED if any AI flags are explicitly provided
    ai_explicitly_requested = any([args.ai_prioritize, args.ai_analyze, args.ai_validate, args.ai_insights])

    print_scan_header(domain, mode, args.wordlist, ai_explicitly_requested)

def handle_missing_wordlist(args):
    """Handle missing wordlist file"""
    console.print(f"⚠️ Wordlist not found: {args.wordlist}", style="yellow")
    console.print("🔧 Using built-in common directories instead", style="blue")
    args.wordlist = None

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run main function
    asyncio.run(main())
