# modules/utils.py
import os
import json
import math
import re
import hashlib
from urllib.parse import urlparse
from typing import List, Dict, Any, Set, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.table import Table
from rich.box import ROUNDED, DOUBLE_EDGE

console = Console()

# HASH-BASED DEDUPLICATION FUNCTIONS

def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content for deduplication"""
    return hashlib.sha256(content.encode('utf-8', errors='ignore')).hexdigest()

def calculate_finding_signature(leak_type: str, value: str) -> str:
    """Calculate unique signature for a finding to avoid duplicates"""
    normalized_value = value.lower().strip()[:50]  # Use first 50 chars for signature
    return f"{leak_type}:{normalized_value}"

def is_duplicate_finding(finding: Dict[str, Any], processed_signatures: Set[str]) -> bool:
    """Check if a finding is a duplicate based on signature"""
    signature = calculate_finding_signature(
        finding.get('type', 'unknown'),
        finding.get('value', '')
    )
    return signature in processed_signatures

def add_finding_signature(finding: Dict[str, Any], processed_signatures: Set[str]) -> str:
    """Add finding signature to processed set and return the signature"""
    signature = calculate_finding_signature(
        finding.get('type', 'unknown'),
        finding.get('value', '')
    )
    processed_signatures.add(signature)
    return signature

# STREAMING PROCESSING UTILITIES

class StreamProcessor:
    """Helper class for streaming content processing"""
    
    def __init__(self):
        self.processed_hashes: Set[str] = set()
        self.processed_signatures: Set[str] = set()
        self.stats = {
            'content_processed': 0,
            'duplicates_skipped': 0,
            'unique_findings': 0
        }
    
    def process_content_streaming(self, content: str, processor_func) -> List[Dict[str, Any]]:
        """Process content with streaming deduplication"""
        if not content:
            return []
            
        # Check for duplicate content
        content_hash = calculate_content_hash(content)
        if content_hash in self.processed_hashes:
            self.stats['duplicates_skipped'] += 1
            return []
            
        # Process content
        self.processed_hashes.add(content_hash)
        self.stats['content_processed'] += 1
        
        findings = processor_func(content)
        
        # Filter duplicate findings
        unique_findings = []
        for finding in findings:
            if not is_duplicate_finding(finding, self.processed_signatures):
                add_finding_signature(finding, self.processed_signatures)
                unique_findings.append(finding)
                self.stats['unique_findings'] += 1
                
        return unique_findings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming processor statistics"""
        return self.stats.copy()

# ENHANCED URL AND CONTENT UTILITIES

def normalize_url(url: str) -> str:
    """Enhanced URL normalization with consistency"""
    url = url.strip()
    if url.endswith('/'):
        url = url[:-1]
    
    # Ensure consistent scheme
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
        
    return url

def is_valid_url(url: str) -> bool:
    """Enhanced URL validation"""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except:
        return False

def extract_domain(url: str) -> str:
    """Extract domain from URL consistently"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ""

def should_skip_static_content(url: str) -> bool:
    """Check if URL points to static content that can be skipped"""
    static_extensions = [
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.css', '.woff', 
        '.woff2', '.ttf', '.eot', '.pdf', '.zip', '.tar', '.gz'
    ]
    
    static_patterns = [
        '/cdn-cgi/', '/_next/static/', '/static/', '/assets/', '/images/',
        '/fonts/', '/css/', '/js/libs/', '/wp-content/themes/'
    ]
    
    url_lower = url.lower()
    
    # Check extensions
    if any(url_lower.endswith(ext) for ext in static_extensions):
        return True
        
    # Check patterns
    if any(pattern in url_lower for pattern in static_patterns):
        return True
        
    return False

# ENHANCED SECURITY UTILITIES

def entropy_score(string: str) -> float:
    """Calculate Shannon entropy for a string"""
    if not string:
        return 0.0
    prob = [float(string.count(c)) / len(string) for c in set(string)]
    return -sum(p * math.log2(p) for p in prob)

def compute_severity(name: str, entropy: float) -> str:
    """
    Enhanced severity computation with streaming context
    """
    name_lower = name.lower()

    critical_types = {
        "aws_access_key", "aws_secret_key", "stripe_key", "twilio_key", 
        "github_token", "private_key", "ssh_private_key", "credit_card"
    }
    
    high_types = {
        "google_api_key", "bearer_token", "jwt_token", "slack_token", 
        "firebase_key", "mailgun_key", "mongodb_uri", "mysql_connection",
        "postgres_connection", "redis_connection"
    }
    
    medium_types = {
        "email_password", "s3_bucket", "admin_endpoint", "api_endpoint",
        "generic_api_key", "basic_auth", "internal_domain"
    }

    if name_lower in critical_types:
        return "CRITICAL"
    elif name_lower in high_types:
        return "HIGH"
    elif name_lower in medium_types:
        return "MEDIUM"
    elif entropy > 4.0:
        return "HIGH"
    elif entropy > 3.0:
        return "MEDIUM"
    else:
        return "LOW"

# FILE AND STORAGE UTILITIES

def save_json(data, filepath):
    """Save data as JSON file with streaming-friendly format"""
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

def save_findings_streaming(findings: List[Dict], filepath: str):
    """Save findings in streaming-friendly format (append mode)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert findings to JSON lines format for streaming
    with open(filepath, "a", encoding='utf-8') as f:
        for finding in findings:
            f.write(json.dumps(finding, ensure_ascii=False) + '\n')

def load_findings_streaming(filepath: str) -> List[Dict]:
    """Load findings from streaming JSON lines format"""
    findings = []
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    findings.append(json.loads(line.strip()))
    except FileNotFoundError:
        pass
    return findings

# ENHANCED UI AND DISPLAY UTILITIES

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
        title="🚀 LeakHunterX - Streaming Edition",
        subtitle=f"🔥 {text}",
        border_style="bright_red",
        padding=(1, 2)
    )

    console.print(banner)

    # Enhanced feature highlights
    features = [
        "[red]•[/red] [white]Streaming Analysis[/white]",
        "[red]•[/red] [white]Hash-Based Deduplication[/white]",
        "[red]•[/red] [white]Memory-Efficient Processing[/white]",
        "[red]•[/red] [white]AI-Powered Validation[/white]",
        "[red]•[/red] [white]Zero Storage Bloat[/white]"
    ]

    feature_columns = Columns(features, equal=True, expand=True)
    console.print(feature_columns)
    console.print()

def print_status(message, status="info"):
    """Enhanced status messages with streaming context"""
    status_colors = {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "debug": "blue",
        "streaming": "magenta"
    }

    icon = {
        "info": "🔍",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "debug": "🐛",
        "streaming": "🔄"
    }

    color = status_colors.get(status, "white")
    emoji = icon.get(status, "•")

    console.print(f"[{color}]{emoji} {message}[/{color}]")

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

def print_memory_efficiency(stats: Dict[str, Any]):
    """Print memory efficiency statistics"""
    if stats.get('duplicates_skipped', 0) > 0:
        efficiency = (stats['duplicates_skipped'] / 
                     (stats['content_processed'] + stats['duplicates_skipped'])) * 100
        console.print(f"[green]    💾 Memory Efficiency: {efficiency:.1f}% duplicates skipped[/green]")

# ENHANCED VALIDATION UTILITIES

def validate_domain(domain: str) -> bool:
    """Enhanced domain validation"""
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    return bool(re.match(domain_pattern, domain))

def clean_domain(domain: str) -> str:
    """Enhanced domain cleaning"""
    domain = domain.lower().strip()
    domain = domain.replace('https://', '').replace('http://', '').split('/')[0]
    domain = domain.split(':')[0]
    return domain

def is_js_file(url: str) -> bool:
    """Enhanced JS file detection"""
    js_extensions = ['.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx']
    js_patterns = ['/js/', '/static/js/', '/assets/js/', '/dist/js/']
    
    url_lower = url.lower()
    
    # Check extensions
    if any(url_lower.endswith(ext) for ext in js_extensions):
        return True
        
    # Check patterns
    if any(pattern in url_lower for pattern in js_patterns):
        return True
        
    # Check common JS file patterns
    if re.search(r'/[^/]+\.js(?:\?|$|/)', url_lower):
        return True
        
    return False

# COMPATIBILITY FUNCTIONS

def ensure_dir(path):
    """Ensure the directory exists, create if not"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def color_text(message, color="white"):
    """Compatibility function"""
    return f"[{color}]{message}[/{color}]"

def print_phase_header(phase_number: int, phase_name: str, emoji: str = "🔍"):
    """Enhanced phase headers with streaming context"""
    console.print(f"\n[bold cyan]{emoji} Phase {phase_number}: {phase_name}[/bold cyan]")
    console.print("[dim]" + "─" * 60 + "[/dim]")

def print_batch_progress(current: int, total: int, item_type: str, samples: list = None):
    """Enhanced batch progress with streaming info"""
    if total > 10 and current <= 10:
        sample_text = ", ".join(samples[:3]) if samples else ""
        if len(samples) > 3:
            sample_text += f" ... (+{len(samples) - 3} more)"
        console.print(f"[blue]    Streaming {total} {item_type}: {sample_text}[/blue]")
    elif current == total:
        console.print(f"[green]    ✅ Stream processed {total} {item_type}[/green]")

def print_crawl_stats(stats: dict):
    """Enhanced crawl statistics with exact counts"""
    if not stats:
        return

    # Use exact state tracking if available
    success_count = stats.get('urls_processed_success', stats.get('urls_crawled', 0))
    failed_count = stats.get('urls_processed_failure', stats.get('urls_failed', 0))
    total = success_count + failed_count

    if total == 0:
        return

    success_percent = (success_count / total) * 100
    failed_percent = (failed_count / total) * 100

    # Create progress bars
    success_bar = "█" * int(success_percent / 10)
    failed_bar = "█" * int(failed_percent / 10)

    console.print(f"[cyan]    📊 Crawl Statistics (Exact):[/cyan]")
    console.print(f"[green]    Success: {success_bar} {success_count}/{total} ({success_percent:.1f}%)[/green]")
    console.print(f"[red]    Failed : {failed_bar} {failed_count}/{total} ({failed_percent:.1f}%)[/red]")

    # Enhanced error breakdown
    error_breakdown = stats.get('error_breakdown', {})
    if error_breakdown:
        console.print(f"[yellow]    🐛 Error Breakdown:[/yellow]")
        for error_type, count in error_breakdown.items():
            if count > 0:
                console.print(f"[yellow]      • {error_type}: {count}[/yellow]")

def create_enhanced_results_table():
    """Create professional results table"""
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
    """Enhanced AI activity with batch context"""
    console.print(f"[magenta]    🤖 {operation}[/magenta]")
    if sent_count > 0:
        console.print(f"[blue]    📤 Sent: {sent_count} items[/blue]")
    if returned_count > 0:
        console.print(f"[green]    📥 Returned: {returned_count} items[/green]")
    if details:
        console.print(f"[cyan]    📊 {details}[/cyan]")

def print_final_summary(results: dict, duration: str):
    """Enhanced final summary with streaming metrics"""
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

    # Enhanced risk assessment
    critical_leaks = len([l for l in results.get('leaks', []) if l.get('severity') == 'CRITICAL'])
    high_leaks = len([l for l in results.get('leaks', []) if l.get('severity') == 'HIGH'])
    validated_leaks = len([l for l in results.get('leaks', []) if l.get('ai_validated', False)])

    console.print(f"\n[bold green]🎯 STREAMING SCAN COMPLETE[/bold green]")
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
    if validated_leaks > 0:
        console.print(f"[cyan]│ ✅ AI-Validated: {validated_leaks:<32} │[/cyan]")

    console.print(f"[cyan]│ ✅ High-Confidence: {stats['high_confidence_leaks']:<30} │[/cyan]")
    console.print(f"[cyan]│ ⏱️  Duration: {stats['duration']:<35} │[/cyan]")

    if stats['ai_prioritized'] > 0:
        console.print(f"[cyan]│ 🤖 AI-Prioritized: {stats['ai_prioritized']:<31} │[/cyan]")

    console.print(f"[cyan]└{'─' * 50}┘[/cyan]")

# STREAMING-SPECIFIC UTILITIES

def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def estimate_memory_savings(duplicates_skipped: int, avg_content_size: int = 50000) -> str:
    """Estimate memory savings from deduplication"""
    memory_saved = duplicates_skipped * avg_content_size
    return format_size(memory_saved)

def print_deduplication_stats(processed: int, skipped: int):
    """Print deduplication statistics"""
    if skipped > 0:
        efficiency = (skipped / (processed + skipped)) * 100
        memory_saved = estimate_memory_savings(skipped)
        console.print(f"[green]    💾 Deduplication: {efficiency:.1f}% efficiency, ~{memory_saved} saved[/green]")
