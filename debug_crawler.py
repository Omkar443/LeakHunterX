# debug_crawler.py
import asyncio
import aiohttp
from urllib.parse import urlparse
from rich.console import Console

console = Console()

async def test_individual_urls():
    """Test individual URLs to see what's actually happening"""
    console.print("\n🔍 [bold red]DEBUG: Testing individual URLs[/bold red]")
    
    test_urls = [
        "https://tesla.com",
        "https://www.tesla.com", 
        "https://diner-webapp.tesla.com",
        "https://akamai-apigateway-stg-warpdashboardapi.tesla.com",
        "https://google.com"  # Control test
    ]
    
    from modules.http_client import AsyncHTTPClient
    
    http_client = AsyncHTTPClient(timeout=10.0, max_retries=1)
    await http_client.init_dns_resolver()
    
    for url in test_urls:
        console.print(f"\n🎯 Testing: {url}", style="bold cyan")
        
        try:
            # Test DNS first
            parsed = urlparse(url)
            domain = parsed.netloc
            
            console.print(f"🔍 Testing DNS for: {domain}", style="blue")
            dns_works = await http_client.validate_dns_async(domain)
            console.print(f"🌐 DNS: {'✅ WORKS' if dns_works else '❌ FAILED'}", 
                         style="green" if dns_works else "red")
            
            if not dns_works:
                continue
                
            # Test HTTP request
            console.print(f"🌐 Testing HTTP request...", style="blue")
            fetched_url, content, status_code, state = await http_client.fetch_with_fallback(url)
            
            if status_code == 200:
                console.print(f"✅ HTTP 200 SUCCESS: {url}", style="green")
                console.print(f"   Content length: {len(content) if content else 0} bytes", style="dim")
            elif status_code > 0:
                console.print(f"⚠️ HTTP {status_code}: {url}", style="yellow")
                console.print(f"   State: {state}", style="dim")
            else:
                console.print(f"❌ FAILED: {url}", style="red")
                console.print(f"   State: {state}", style="dim")
                
        except Exception as e:
            console.print(f"💥 EXCEPTION: {url} - {e}", style="red")
    
    await http_client.close()

async def test_crawler_with_fixed_state():
    """Test the crawler with state reset"""
    console.print("\n🔍 [bold red]DEBUG: Testing crawler with fixed state[/bold red]")
    
    from modules.domain_manager import DomainManager, URLState
    from modules.crawler import EnterpriseCrawler
    
    # Create domain manager
    dm = DomainManager("tesla.com")
    
    # Add test URLs
    test_urls = [
        "https://tesla.com",
        "https://www.tesla.com",
        "https://google.com"  # Control
    ]
    
    for url in test_urls:
        dm.add_priority_target(url, depth=0, score=100)
    
    # FIX: Reset any stuck URLs before starting
    console.print("🔧 Resetting any stuck URLs...", style="yellow")
    reset_count = dm.force_reset_all_processing()
    console.print(f"✅ Reset {reset_count} stuck URLs", style="green")
    
    # Verify state
    queued_urls = dm.get_urls_by_state(URLState.QUEUED)
    console.print(f"📊 URLs in QUEUED state: {len(queued_urls)}", style="blue")
    
    # Create crawler
    crawler = EnterpriseCrawler(domain_manager=dm, concurrency=2, max_depth=1)
    
    console.print(f"📊 Starting with {dm.get_stats()['urls_queued']} URLs", style="blue")
    
    # Run crawl
    urls, js_files = await crawler.crawl()
    
    console.print(f"📊 Results: {len(urls)} URLs, {len(js_files)} JS files", style="blue")
    
    return urls, js_files

async def test_direct_http_requests():
    """Test HTTP requests directly without crawler"""
    console.print("\n🔍 [bold red]DEBUG: Direct HTTP requests[/bold red]")
    
    test_urls = [
        "https://tesla.com",
        "https://www.tesla.com",
        "https://google.com"
    ]
    
    for url in test_urls:
        console.print(f"\n🎯 Direct test: {url}", style="bold cyan")
        
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers=headers
            ) as session:
                
                async with session.get(url, ssl=False, allow_redirects=True) as response:
                    console.print(f"🌐 Status: {response.status}", 
                                style="green" if response.status == 200 else "yellow")
                    console.print(f"📏 Content length: {len(await response.text())} bytes", style="dim")
                    
        except Exception as e:
            console.print(f"💥 Exception: {e}", style="red")

async def main():
    """Run comprehensive debugging"""
    console.print("🚀 [bold magenta]LEAKHUNTERX CRAWLER DEBUG - FIXED[/bold magenta]")
    console.print("=" * 60, style="bold blue")
    
    # Test 1: Individual URL testing with HTTP client
    await test_individual_urls()
    
    console.print("\n" + "=" * 60, style="bold blue")
    
    # Test 2: Direct HTTP requests
    await test_direct_http_requests()
    
    console.print("\n" + "=" * 60, style="bold blue")
    
    # Test 3: Crawler with fixed state
    urls, js_files = await test_crawler_with_fixed_state()
    
    console.print("\n" + "=" * 60, style="bold blue")
    console.print("📊 [bold green]DEBUG COMPLETE[/bold green]")
    console.print(f"🎯 Found {len(urls)} URLs and {len(js_files)} JS files", style="bold")

if __name__ == "__main__":
    asyncio.run(main())
