# modules/utils.py
import os
import json
import math
import re
from urllib.parse import urlparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.table import Table
from rich.box import ROUNDED, DOUBLE_EDGE

console = Console()

def normalize_url(url: str) -> str:
    url = url.strip()
    if url.endswith('/'):
        url = url[:-1]
    return url

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except:
        return False

def entropy_score(string: str) -> float:
    if not string:
        return 0.0
    prob = [float(string.count(c)) / len(string) for c in set(string)]
    return -sum(p * math.log2(p) for p in prob)

def compute_severity(name: str, entropy: float) -> str:
    """
    Return severity based on type + entropy
    """
    name_lower = name.lower()

    if name_lower in ["aws_access_key", "aws_secret_key", "stripe_key", "twilio_key", "github_token"]:
        return "CRITICAL"
    elif name_lower in ["google_api_key", "bearer_token", "jwt_token", "slack_token", "firebase_key"]:
        return "HIGH"
    elif name_lower in ["email_password", "s3_bucket", "mongodb_uri", "mysql_connection", "admin_endpoint"]:
        return "MEDIUM"
    elif entropy > 4.0:
        return "HIGH"
    elif entropy > 3.0:
        return "MEDIUM"
    else:
        return "LOW"

def save_json(data, filepath):
    """Save data as JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_to_file(data, filepath):
    """Save data to text file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding='utf-8') as f:
        if isinstance(data, list):
            for line in data:
                f.write(f"{line}\n")
        else:
            f.write(str(data))

def print_banner(text):
    """Print enhanced LeakHunterX banner"""
    banner_text = Text()
    banner_text.append("""
╦  ╔═╗╔═╗╔═╗╦ ╦╔═╗╦ ╦╔╦╗╦╔═╗╦  ╔═╗
║  ║ ║║ ║║ ║║ ║╠═╝║ ║ ║ ║║  ║  ╚═╗
╩═╝╚═╝╚═╝╚═╝╚═╝╩  ╚═╝ ╩ ╩╚═╝╩═╝╚═╝
""", style="bold red")

    banner = Panel(
        banner_text,
        title="🚀 LeakHunterX",
        subtitle=f"🔥 {text}",
        border_style="bright_red",
        padding=(1, 2)
    )

    console.print(banner)

    # Feature highlights
    features = [
        "[red]•[/red] [white]Advanced Secret Scanning[/white]",
        "[red]•[/red] [white]JavaScript Analysis[/white]",
        "[red]•[/red] [white]Subdomain Discovery[/white]",
        "[red]•[/red] [white]AI-Powered Triage[/white]",
        "[red]•[/red] [white]Professional Reporting[/white]"
    ]

    feature_columns = Columns(features, equal=True, expand=True)
    console.print(feature_columns)
    console.print()

def print_status(message, status="info"):
    """Print status messages with consistent styling"""
    status_colors = {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "debug": "blue"
    }

    icon = {
        "info": "🔍",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "debug": "🐛"
    }

    color = status_colors.get(status, "white")
    emoji = icon.get(status, "•")

    console.print(f"[{color}]{emoji} {message}[/{color}]")

def print_summary(stats: dict):
    """Print scan summary"""
    summary_panel = Panel(
        f"""[bold white]Scan Complete![/bold white]

📊 [cyan]Summary:[/cyan]
   • [green]Subdomains Found:[/green] {stats.get('subdomains', 0)}
   • [green]URLs Crawled:[/green] {stats.get('urls', 0)}
   • [green]JS Files Analyzed:[/green] {stats.get('js_files', 0)}
   • [red]Leaks Detected:[/green] {stats.get('leaks', 0)}

📁 [cyan]Reports saved in:[/cyan] [white]{stats.get('output_dir', 'results/')}[/white]""",
        title="🎯 Scan Results",
        border_style="green"
    )
    console.print(summary_panel)

def validate_domain(domain: str) -> bool:
    """Validate domain format"""
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    return bool(re.match(domain_pattern, domain))

def clean_domain(domain: str) -> str:
    """Clean and normalize domain input"""
    domain = domain.lower().strip()
    # Remove protocol and path
    domain = domain.replace('https://', '').replace('http://', '').split('/')[0]
    # Remove port if present
    domain = domain.split(':')[0]
    return domain

def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def is_js_file(url: str) -> bool:
    """Check if URL points to a JavaScript file"""
    js_extensions = ['.js', '.mjs', '.cjs']
    return any(url.lower().endswith(ext) for ext in js_extensions) or '/js/' in url.lower()

def ensure_dir(path):
    """Ensure the directory exists, create if not"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# COMPATIBILITY FUNCTIONS - For modules that still use old function names
def color_text(message, color="white"):
    """Compatibility function - use print_status instead"""
    return f"[{color}]{message}[/{color}]"

def print_phase_header(phase_number: int, phase_name: str, emoji: str = "🔍"):
    """Print professional phase headers"""
    console.print(f"\n[bold cyan]{emoji} Phase {phase_number}: {phase_name}[/bold cyan]")

def print_batch_progress(current: int, total: int, item_type: str, samples: list = None):
    """Print batch progress with samples"""
    if total > 10 and current <= 10:
        # Show first 10 items with samples
        sample_text = ", ".join(samples[:3]) if samples else ""
        if len(samples) > 3:
            sample_text += f" ... (+{len(samples) - 3} more)"
        console.print(f"[blue]    Analyzing {total} {item_type}: {sample_text}[/blue]")
    elif current == total:
        console.print(f"[green]    ✅ Completed analysis of {total} {item_type}[/green]")

def print_crawl_stats(stats: dict):
    """Print beautiful crawl statistics"""
    if not stats:
        return
    
    success_count = stats.get('urls_crawled', 0)
    failed_count = stats.get('urls_failed', 0)
    total = success_count + failed_count
    
    if total == 0:
        return
    
    success_percent = (success_count / total) * 100
    failed_percent = (failed_count / total) * 100
    
    # Create progress bars
    success_bar = "█" * int(success_percent / 10)
    failed_bar = "█" * int(failed_percent / 10)
    
    console.print(f"[cyan]    📊 Crawl Statistics:[/cyan]")
    console.print(f"[green]    Success: {success_bar} {success_count}/{total} ({success_percent:.1f}%)[/green]")
    console.print(f"[red]    Failed : {failed_bar} {failed_count}/{total} ({failed_percent:.1f}%)[/red]")
    
    # Error breakdown
    if stats.get('dns_failures', 0) > 0:
        console.print(f"[yellow]    🌐 DNS Failures: {stats.get('dns_failures', 0)}[/yellow]")
    if stats.get('timeouts', 0) > 0:
        console.print(f"[yellow]    ⏰ Timeouts: {stats.get('timeouts', 0)}[/yellow]")

def create_enhanced_results_table():
    """Create professional results table with better formatting"""
    table = Table(
        title="🎯 Scan Results - Priority Targets",
        title_style="bold red",
        header_style="bold cyan",
        box=DOUBLE_EDGE,
        show_header=True,
        show_lines=True
    )
    
    table.add_column("#", style="dim", width=4)
    table.add_column("Target", style="cyan", width=22)
    table.add_column("Endpoint", style="green", width=28)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Priority", justify="center", width=10)
    table.add_column("Confidence", justify="center", width=12)
    
    return table

def print_ai_activity(operation: str, sent_count: int, returned_count: int, details: str = ""):
    """Print AI activity with clear metrics"""
    console.print(f"[magenta]    🤖 {operation}[/magenta]")
    if sent_count > 0:
        console.print(f"[blue]    📤 Sent: {sent_count} items[/blue]")
    if returned_count > 0:
        console.print(f"[green]    📥 Returned: {returned_count} items[/green]")
    if details:
        console.print(f"[cyan]    📊 {details}[/cyan]")

def print_final_summary(results: dict, duration: str):
    """Print comprehensive final summary"""
    stats = {
        'domain': results.get('scan_config', {}).get('domain', 'Unknown'),
        'subdomains': len(results.get('subdomains', [])),
        'urls': len(results.get('urls', [])),
        'js_files': len(results.get('js_files', [])),
        'leaks': len(results.get('leaks', [])),
        'high_confidence_leaks': len(results.get('high_confidence_leaks', [])),
        'duration': duration,
        'ai_prioritized': len(results.get('ai_prioritized_urls', []))
    }
    
    # Risk assessment
    critical_leaks = len([l for l in results.get('leaks', []) if l.get('severity') == 'CRITICAL'])
    high_leaks = len([l for l in results.get('leaks', []) if l.get('severity') == 'HIGH'])
    
    console.print(f"\n[bold green]🎯 SCAN COMPLETE[/bold green]")
    console.print(f"[cyan]┌{'─' * 50}┐[/cyan]")
    console.print(f"[cyan]│ 📊 Executive Summary{' ' * 29}│[/cyan]")
    console.print(f"[cyan]├{'─' * 50}┤[/cyan]")
    console.print(f"[cyan]│ 🔍 Target: {stats['domain']:<36} │[/cyan]")
    console.print(f"[cyan]│ 🎯 Subdomains: {stats['subdomains']:<35} │[/cyan]")
    console.print(f"[cyan]│ 🌐 URLs Crawled: {stats['urls']:<33} │[/cyan]")
    console.print(f"[cyan]│ 📜 JS Files: {stats['js_files']:<36} │[/cyan]")
    console.print(f"[cyan]│ 🔐 Total Leaks: {stats['leaks']:<35} │[/cyan]")
    
    if critical_leaks > 0:
        console.print(f"[cyan]│ 🚨 Critical Leaks: [red]{critical_leaks}[/red]{' ' * 28} │[/cyan]")
    if high_leaks > 0:
        console.print(f"[cyan]│ ⚠️  High Leaks: [yellow]{high_leaks}[/yellow]{' ' * 31} │[/cyan]")
        
    console.print(f"[cyan]│ ✅ High-Confidence: {stats['high_confidence_leaks']:<30} │[/cyan]")
    console.print(f"[cyan]│ ⏱️  Duration: {stats['duration']:<35} │[/cyan]")
    
    if stats['ai_prioritized'] > 0:
        console.print(f"[cyan]│ 🤖 AI-Prioritized: {stats['ai_prioritized']:<31} │[/cyan]")
    
    console.print(f"[cyan]└{'─' * 50}┘[/cyan]")
