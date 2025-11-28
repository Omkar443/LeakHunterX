#!/usr/bin/env python3
"""
LeakHunterX - Connectivity Test Script
Diagnose why crawler is failing to fetch URLs
"""

import asyncio
import aiohttp
import socket
from urllib.parse import urlparse
from rich.console import Console

console = Console()

async def test_dns_resolution(domain: str) -> bool:
    """Test DNS resolution for a domain"""
    try:
        console.print(f"🔍 Testing DNS for: {domain}", style="blue")
        
        # Test IPv4
        try:
            socket.getaddrinfo(domain, 443, family=socket.AF_INET, timeout=5)
            console.print(f"✅ IPv4 DNS works: {domain}", style="green")
            return True
        except socket.gaierror:
            pass
        
        # Test IPv6
        try:
            socket.getaddrinfo(domain, 443, family=socket.AF_INET6, timeout=5)
            console.print(f"✅ IPv6 DNS works: {domain}", style="green")
            return True
        except socket.gaierror:
            pass
            
        console.print(f"❌ DNS failed: {domain}", style="red")
        return False
        
    except Exception as e:
        console.print(f"💥 DNS error for {domain}: {e}", style="red")
        return False

async def test_http_request(url: str, timeout: float = 10.0) -> dict:
    """Test HTTP request with detailed diagnostics"""
    result = {
        'url': url,
        'dns_works': False,
        'status_code': 0,
        'response_time': 0,
        'error': None,
        'headers': {},
        'final_url': url
    }
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Test DNS first
        result['dns_works'] = await test_dns_resolution(domain)
        if not result['dns_works']:
            result['error'] = "DNS resolution failed"
            return result
        
        # Test HTTP request
        console.print(f"🌐 Testing HTTP: {url}", style="blue")
        
        connector = aiohttp.TCPConnector(ssl=False, limit=1)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout_obj,
            headers=headers
        ) as session:
            
            async with session.get(url, ssl=False, allow_redirects=True) as response:
                result['status_code'] = response.status
                result['headers'] = dict(response.headers)
                result['final_url'] = str(response.url)
                
                # Try to read a small part of content
                try:
                    content_sample = await response.text()[:100]
                    result['content_sample'] = content_sample
                except:
                    result['content_sample'] = "Unable to read content"
        
        result['response_time'] = asyncio.get_event_loop().time() - start_time
        
        if result['status_code'] == 200:
            console.print(f"✅ HTTP 200: {url} ({result['response_time']:.2f}s)", style="green")
        else:
            console.print(f"⚠️ HTTP {result['status_code']}: {url}", style="yellow")
            
    except asyncio.TimeoutError:
        result['error'] = f"Timeout after {timeout}s"
        console.print(f"⏰ Timeout: {url}", style="red")
    except aiohttp.ClientConnectorError as e:
        result['error'] = f"Connection error: {e}"
        console.print(f"🔌 Connection error: {url} - {e}", style="red")
    except aiohttp.ClientResponseError as e:
        result['error'] = f"Response error: {e}"
        result['status_code'] = e.status
        console.print(f"📡 Response error: {url} - {e.status}", style="red")
    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
        console.print(f"💥 Unexpected error: {url} - {e}", style="red")
    
    return result

async def test_protocol_fallback(url: str) -> dict:
    """Test both HTTPS and HTTP protocols"""
    results = {}
    
    # Test HTTPS first
    if url.startswith('https://'):
        results['https'] = await test_http_request(url)
        
        # Test HTTP fallback
        http_url = url.replace('https://', 'http://')
        results['http'] = await test_http_request(http_url)
    else:
        results['http'] = await test_http_request(url)
        
        # Test HTTPS
        https_url = url.replace('http://', 'https://')
        results['https'] = await test_http_request(https_url)
    
    return results

async def comprehensive_connectivity_test():
    """Run comprehensive connectivity tests"""
    console.print("\n" + "="*60, style="bold blue")
    console.print("🔧 LEAKHUNTERX CONNECTIVITY DIAGNOSTICS", style="bold blue")
    console.print("="*60, style="bold blue")
    
    # Test domains - mix of base domain and subdomains
    test_urls = [
        "https://tesla.com",
        "https://www.tesla.com",
        "https://static.tesla.com",
        "https://akamai-apigateway-stg-warpdashboardapi.tesla.com",
        "https://apf-api.eng.vn.cloud.tesla.com",
        "https://digitalassets-accounts.tesla.com",
        "https://origin-bolt.tesla.com",
        "https://origin-finplat-stg.tesla.com"
    ]
    
    all_results = {}
    
    for url in test_urls:
        console.print(f"\n🎯 Testing: {url}", style="bold cyan")
        all_results[url] = await test_protocol_fallback(url)
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Print summary
    console.print("\n" + "="*60, style="bold blue")
    console.print("📊 CONNECTIVITY TEST SUMMARY", style="bold blue")
    console.print("="*60, style="bold blue")
    
    successful_dns = 0
    successful_http = 0
    blocked_403 = 0
    timeouts = 0
    connection_errors = 0
    
    for url, protocols in all_results.items():
        console.print(f"\n🔗 {url}", style="bold")
        
        for protocol, result in protocols.items():
            status_emoji = "✅" if result.get('status_code') == 200 else "❌"
            dns_emoji = "✅" if result.get('dns_works') else "❌"
            
            if result.get('dns_works'):
                successful_dns += 1
            
            if result.get('status_code') == 200:
                successful_http += 1
            elif result.get('status_code') == 403:
                blocked_403 += 1
            
            if result.get('error'):
                if 'Timeout' in result['error']:
                    timeouts += 1
                elif 'Connection' in result['error']:
                    connection_errors += 1
            
            console.print(f"  {protocol.upper():6} | DNS: {dns_emoji} | HTTP: {status_emoji} {result.get('status_code', 'N/A'):3} | {result.get('error', 'Success')}")
    
    # Overall statistics
    console.print("\n" + "="*60, style="bold blue")
    console.print("📈 OVERALL STATISTICS", style="bold blue")
    console.print("="*60, style="bold blue")
    
    total_tests = len(test_urls) * 2  # Each URL tested with HTTP and HTTPS
    
    console.print(f"✅ Successful DNS: {successful_dns}/{total_tests}")
    console.print(f"✅ Successful HTTP 200: {successful_http}/{total_tests}")
    console.print(f"🚫 403 Blocks: {blocked_403}/{total_tests}")
    console.print(f"⏰ Timeouts: {timeouts}/{total_tests}")
    console.print(f"🔌 Connection Errors: {connection_errors}/{total_tests}")
    
    # Recommendations based on results
    console.print("\n" + "="*60, style="bold blue")
    console.print("💡 RECOMMENDATIONS", style="bold blue")
    console.print("="*60, style="bold blue")
    
    if blocked_403 > total_tests * 0.5:
        console.print("🔧 ISSUE: High rate of 403 blocks")
        console.print("   → Implement better User-Agent rotation")
        console.print("   → Add request headers randomization")
        console.print("   → Consider using proxies")
        
    if timeouts > total_tests * 0.3:
        console.print("🔧 ISSUE: Frequent timeouts")
        console.print("   → Increase HTTP timeout values")
        console.print("   → Add retry logic with exponential backoff")
        
    if connection_errors > total_tests * 0.3:
        console.print("🔧 ISSUE: Connection errors")
        console.print("   → Check network connectivity")
        console.print("   → Verify DNS resolver configuration")
        
    if successful_http == 0:
        console.print("🔧 ISSUE: No successful HTTP requests")
        console.print("   → Target might be aggressively blocking")
        console.print("   → Consider using residential proxies")
        console.print("   → Add delays between requests")
    else:
        console.print("✅ Some requests successful - crawler should work with adjustments")

async def test_crawler_components():
    """Test individual crawler components"""
    console.print("\n" + "="*60, style="bold green")
    console.print("🕷️ CRAWLER COMPONENT TESTS", style="bold green")
    console.print("="*60, style="bold green")
    
    try:
        # Test DomainManager
        console.print("🔧 Testing DomainManager...", style="blue")
        from modules.domain_manager import DomainManager
        
        dm = DomainManager("tesla.com")
        dm.add_priority_target("https://tesla.com", depth=0, score=100)
        dm.add_priority_target("https://www.tesla.com", depth=0, score=90)
        
        # Get next targets
        url1, depth1 = dm.get_next_target()
        url2, depth2 = dm.get_next_target()
        
        console.print(f"✅ DomainManager: Got targets - {url1}, {url2}", style="green")
        
        # Mark as complete
        dm.mark_url_complete(url1, success=True)
        dm.mark_url_complete(url2, success=False, error_type="http_error")
        
        stats = dm.get_stats()
        console.print(f"✅ DomainManager stats: {stats['urls_processed_success']} success, {stats['urls_processed_failure']} failed", style="green")
        
    except Exception as e:
        console.print(f"❌ DomainManager test failed: {e}", style="red")
    
    try:
        # Test HTTP Client
        console.print("\n🔧 Testing HTTP Client...", style="blue")
        from modules.http_client import AsyncHTTPClient
        
        http_client = AsyncHTTPClient(timeout=10.0, max_retries=1)
        await http_client.init_dns_resolver()
        
        test_url = "https://tesla.com"
        fetched_url, content, status_code, state = await http_client.fetch_with_fallback(test_url)
        
        console.print(f"✅ HTTP Client: {test_url} → Status: {status_code}, State: {state}", style="green")
        
        await http_client.close()
        
    except Exception as e:
        console.print(f"❌ HTTP Client test failed: {e}", style="red")

async def main():
    """Main diagnostic function"""
    console.print("🚀 LeakHunterX Connectivity Diagnostics", style="bold magenta")
    console.print("This will identify why the crawler is failing...", style="dim")
    
    # Run comprehensive connectivity tests
    await comprehensive_connectivity_test()
    
    # Test crawler components
    await test_crawler_components()
    
    console.print("\n🎉 Diagnostics complete! Check recommendations above.", style="bold green")

if __name__ == "__main__":
    # Run diagnostics
    asyncio.run(main())
