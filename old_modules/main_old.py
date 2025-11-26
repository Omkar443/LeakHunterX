#!/usr/bin/env python3
"""
LeakHunterX - Industry-Grade AI-Powered Security Scanner
Complete Main Controller with Clean Professional Interface
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
└── Wordlist: [bold blue]{wordlist}[/bold blue]

🚀 Starting automated reconnaissance...
"""
    console.print(header)

def print_phase_header_main(phase_number: int, phase_name: str, emoji: str):
    """Print professional phase header - MAIN VERSION (avoid conflict)"""
    console.print(f"\n[bold cyan]Phase {phase_number}: {phase_name} {emoji}[/bold cyan]")
    console.print("[dim]" + "─" * 60 + "[/dim]")

def print_batch_progress_main(current: int, total: int, item_type: str, samples: List[str] = None):
    """Print clean batch processing progress"""
    if current == 1:  # Only show at start
        console.print(f"    📦 Processing {total} {item_type}...", style="blue")

def print_crawl_stats_main(stats: Dict[str, Any]):
    """Print professional crawl statistics - CLEAN VERSION"""
    if stats.get('urls_failed', 0) > 0:
        success = stats.get('urls_crawled', 0)
        failed = stats.get('urls_failed', 0)
        console.print(f"    📊 Crawl stats: {success} succeeded, {failed} failed", style="yellow")

def print_ai_activity_main(activity: str, total_items: int, processed_items: int, context: str = ""):
    """Print clean AI activity with context"""
    activity_text = f"🤖 {activity}"
    if total_items > 0:
        activity_text += f" ({processed_items}/{total_items})"
    if context:
        activity_text += f" | {context}"
    console.print(f"    {activity_text}", style="magenta")

def print_final_summary_main(results: Dict[str, Any], scan_duration: str):
    """Print professional final summary - MAIN VERSION"""
    stats = {
        'domain': results.get('scan_config', {}).get('domain', 'Unknown'),
        'subdomains': len(results.get('subdomains', [])),
        'urls': len(results.get('urls', [])),
        'js_files': len(results.get('js_files', [])),
        'leaks': len(results.get('leaks', [])),
        'high_confidence_leaks': len(results.get('high_confidence_leaks', [])),
        'duration': scan_duration,
        'ai_prioritized': len(results.get('ai_prioritized_urls', []))
    }

    summary_text = f"""
🎯 [bold]Scan Summary[/bold]
├── Target: {stats['domain']}
├── Subdomains Found: {stats['subdomains']}
├── URLs Crawled: {stats['urls']}
├── JS Files: {stats['js_files']}
├── Leaks Detected: {stats['leaks']}
├── High-Confidence: {stats['high_confidence_leaks']}
└── Duration: {stats['duration']}
"""

    if stats['ai_prioritized'] > 0:
        summary_text += f"📈 AI-Prioritized: {stats['ai_prioritized']} targets\n"

    summary_panel = Panel(
        summary_text.strip(),
        title="🎉 Scan Complete",
        border_style="green",
        padding=(1, 2)
    )
    console.print(summary_panel)

def create_enhanced_results_table():
    """Create enhanced results table"""
    results_table = Table(
        title="🔍 LeakHunterX - Scan Results",
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

async def analyze_js_content(js_url: str, leak_detector, ai) -> List[Dict[str, Any]]:
    """Analyze JavaScript content for leaks with clean error handling"""
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        timeout = aiohttp.ClientTimeout(total=20)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(js_url, ssl=False) as resp:
                if resp.status == 200:
                    js_content = await resp.text(errors='ignore')
                    content_leaks = leak_detector.check_content(js_content)

                    leaks_found = []
                    for leak in content_leaks:
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

async def analyze_url_content(url: str, leak_detector, ai) -> List[Dict[str, Any]]:
    """Analyze URL content for leaks with enhanced error handling"""
    try:
        if not url.endswith(('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg')):
            connector = aiohttp.TCPConnector(ssl=False)
            timeout = aiohttp.ClientTimeout(total=15)

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url, ssl=False) as resp:
                    if resp.status == 200:
                        content = await resp.text(errors='ignore')
                        content_leaks = leak_detector.check_content(content)

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
    """Main scan function with robust error handling"""
    console.print(f"\n🚀 Starting comprehensive scan for [bold green]{domain}[/bold green]")

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
        console.print("🔧 Initializing reconnaissance modules...", style="blue")

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

    # STEP 2: AGGRESSIVE FUZZING - RUNS IMMEDIATELY WHILE AI TESTING CONTINUES
    if args.aggressive and fuzzer:
        await run_phase_2_aggressive_fuzzing(args, dm, results, fuzzer, domain)

    # STEP 3: WEB CRAWLING & CONTENT DISCOVERY - RUNS IMMEDIATELY WHILE AI TESTING CONTINUES
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

    # STEP 5: JAVASCRIPT ANALYSIS & SECRET DETECTION
    leaks_found = await run_phase_5_js_analysis(args, results, js_engine, leak_detector, ai_analyzer, ai_enabled)

    # STEP 6: CONTENT ANALYSIS
    await run_phase_6_content_analysis(results, leak_detector, ai_analyzer)

    # STEP 7: AI LEAK VALIDATION
    if ai_analyzer:
        await run_phase_7_ai_validation(args, results, leaks_found, ai_analyzer, ai_enabled)

    # STEP 8: AI REPORT INSIGHTS
    if args.ai_insights and ai_enabled and ai_analyzer:
        await run_phase_8_ai_insights(results, ai_analyzer)

    # STEP 9: REPORTING
    await run_phase_9_reporting(results, reporter, domain)

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
        "scan_config": {
            "domain": domain,
            "aggressive": args.aggressive,
            "subfinder": args.subfinder,
            "threads": args.threads,
            "ai_enabled": False
        }
    }

def initialize_results(domain: str, args, ai_enabled: bool) -> Dict[str, Any]:
    """Initialize comprehensive results structure"""
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
        "scan_config": {
            "domain": domain,
            "aggressive": args.aggressive,
            "subfinder": args.subfinder,
            "threads": args.threads,
            "ai_enabled": ai_enabled
        }
    }

async def run_phase_1_subdomain_discovery(args, dm, results, sd):
    """Run subdomain discovery phase"""
    print_phase_header_main(1, "Subdomain Discovery", "🎯")
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

async def run_phase_2_aggressive_fuzzing(args, dm, results, fuzzer, domain):
    """Run aggressive fuzzing phase"""
    print_phase_header_main(2, "Aggressive Directory Fuzzing", "🚀")
    try:
        base_url = f"https://{domain}"

        # Quick admin panel discovery
        console.print("    🔐 Scanning for admin panels...", style="blue")
        admin_panels = await fuzzer.find_admin_panels(base_url)
        results["admin_panels"] = admin_panels

        # Directory fuzzing with timeout
        console.print("    📁 Fuzzing directories...", style="blue")
        try:
            found_paths = await asyncio.wait_for(
                fuzzer.fuzz_directories(base_url),
                timeout=45.0
            )
            results["fuzzed_paths"] = found_paths
        except asyncio.TimeoutError:
            console.print("    ⏰ Directory fuzzing timed out, continuing...", style="yellow")

        # Add found paths to crawl queue
        for path in results["fuzzed_paths"]:
            dm.add_discovered(path["url"])
        for panel in results["admin_panels"]:
            dm.add_discovered(panel["url"])

        console.print(f"    ✅ Fuzzing completed: {len(results['fuzzed_paths'])} paths found", style="green")

    except Exception as e:
        console.print(f"    ❌ Fuzzing failed: {e}", style="red")

async def run_phase_3_web_crawling(args, dm, results, crawler):
    """Run web crawling phase"""
    print_phase_header_main(3, "Web Crawling & Content Discovery", "🌐")
    try:
        console.print(f"    Starting crawl with {dm.get_stats()['targets_remaining']} targets...", style="blue")
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
    print_phase_header_main(4, "AI-Powered Target Prioritization", "🤖")
    try:
        # Check if AI is ready (background testing should be complete by now)
        console.print("    🔄 Checking AI connection status...", style="magenta")

        # This will be fast if background testing already completed
        if not await ai_analyzer.ensure_ai_working():
            console.print("    ❌ AI system not available, using fallback prioritization", style="yellow")
            all_urls = results["urls"] + results["js_files"] + [p["url"] for p in results["fuzzed_paths"]]
            results["ai_prioritized_urls"] = ai_analyzer._fallback_prioritization(all_urls)
            return

        all_urls = results["urls"] + results["js_files"] + [p["url"] for p in results["fuzzed_paths"]]

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
        all_urls = results["urls"] + results["js_files"] + [p["url"] for p in results["fuzzed_paths"]]
        results["ai_prioritized_urls"] = ai_analyzer._fallback_prioritization(all_urls)

async def run_phase_5_js_analysis(args, results, js_engine, leak_detector, ai_analyzer, ai_enabled):
    """Run JavaScript analysis phase"""
    print_phase_header_main(5, "JavaScript Analysis & Secret Detection", "📜")
    leaks_found = []

    if results["js_files"]:
        # Clean batch progress
        console.print(f"    📦 Processing {len(results['js_files'])} JS files...", style="blue")

        js_files_to_analyze = results["js_files"][:20]  # Limit for performance

        for i, js_url in enumerate(js_files_to_analyze):
            try:
                # Clean file display - remove the [cyan] tag issue
                domain_part = js_url.split('//')[-1].split('/')[0]
                file_name = js_url.split('/')[-1][:25]
                console.print(f"    🔍 Analyzing: {file_name} ({domain_part})", style="cyan")

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

                # Check endpoints for leaks
                endpoints_text = "\n".join(endpoints)
                potential_leaks = leak_detector.check_content(endpoints_text)

                # Check JS content directly
                content_leaks = await analyze_js_content(js_url, leak_detector, ai_analyzer)
                potential_leaks.extend(content_leaks)

                # Score and add leaks
                for leak in potential_leaks:
                    if isinstance(leak, dict):
                        leak["source_url"] = js_url
                        scored = ai_analyzer.score(leak) if ai_analyzer else leak
                        leaks_found.append(scored)

                        if scored.get("confidence", 0) > 0.7:
                            console.print(f"    🚨 LEAK: {scored['type']} - {scored['value'][:50]}...", style="red")

            except Exception as e:
                if args.verbose:  # Only show errors in verbose mode
                    console.print(f"    ❌ Error processing JS file: {e}", style="red")

        console.print(f"    ✅ JS analysis completed: {len(js_files_to_analyze)} files processed", style="green")

    return leaks_found

async def run_phase_6_content_analysis(results, leak_detector, ai_analyzer):
    """Run content analysis phase"""
    print_phase_header_main(6, "Content Analysis", "🔍")
    try:
        analysis_tasks = []
        for url in results["urls"][:15]:  # Limit for performance
            task = analyze_url_content(url, leak_detector, ai_analyzer)
            analysis_tasks.append(task)

        if analysis_tasks:
            content_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            valid_results = [r for r in content_results if isinstance(r, list)]
            total_leaks = sum(len(r) for r in valid_results)

            console.print(f"    ✅ Content analysis completed: {total_leaks} potential leaks found", style="green")
    except Exception as e:
        console.print(f"    ❌ Content analysis failed: {e}", style="red")

async def run_phase_7_ai_validation(args, results, leaks_found, ai_analyzer, ai_enabled):
    """Run AI validation phase"""
    if args.ai_validate and ai_enabled and leaks_found and ai_analyzer:
        print_phase_header_main(7, "AI Leak Validation", "🤖")
        try:
            print_ai_activity_main(
                "Validating detected leaks",
                len(leaks_found),
                0,
                f"Processing {len(leaks_found)} potential leaks"
            )

            validated_leaks = await ai_analyzer.validate_leaks(leaks_found)
            results["leaks"] = validated_leaks

            if validated_leaks:
                valid_count = sum(1 for v in validated_leaks if v.get('is_valid', False))
                print_ai_activity_main(
                    "Leak Validation Complete",
                    len(leaks_found),
                    len(validated_leaks),
                    f"Validated: {valid_count} real leaks"
                )

        except Exception as e:
            console.print(f"    ❌ AI validation failed: {e}", style="red")
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

async def run_phase_8_ai_insights(results, ai_analyzer):
    """Run AI insights phase"""
    print_phase_header_main(8, "AI Report Insights", "🧠")
    try:
        print_ai_activity_main("Generating security insights", 0, 0, "Analyzing scan results")
        ai_insights = await ai_analyzer.generate_report_insights(results)
        results["ai_insights"] = ai_insights
        console.print(f"    ✅ AI insights generated", style="green")
    except Exception as e:
        console.print(f"    ❌ AI insights generation failed: {e}", style="red")

async def run_phase_9_reporting(results, reporter, domain):
    """Run reporting phase"""
    print_phase_header_main(9, "Generating Reports", "📊")
    try:
        reporter.save(results)
        console.print(f"    ✅ Reports saved to results/{domain}/", style="green")
    except Exception as e:
        console.print(f"    ❌ Report generation failed: {e}", style="red")

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
