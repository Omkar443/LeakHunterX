# modules/subdomain_finder.py
import asyncio
import subprocess
import logging
from .domain_manager import DomainManager
from .utils import print_status

class SubdomainFinder:
    def __init__(self, domain: str, dm: DomainManager, use_subfinder: bool = True):
        self.domain = domain
        self.dm = dm
        self.use_subfinder = use_subfinder
        self.discovered_subs = set()

    async def _run_subfinder(self):
        """Async wrapper for subfinder CLI"""
        cmd = ["subfinder", "-silent", "-d", self.domain]
        try:
            print_status(f"    Running: {' '.join(cmd)}", "info")
            
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if stdout:
                domains = stdout.decode().splitlines()
                for d in domains:
                    cleaned = d.strip()
                    if cleaned and cleaned not in self.discovered_subs:
                        self.discovered_subs.add(cleaned)
                        # Use add_subdomain instead of add_discovered
                        self.dm.add_subdomain(cleaned)
                print_status(f"    ✅ Found {len(domains)} subdomains via Subfinder", "success")
                
            if stderr:
                error_output = stderr.decode().strip()
                if error_output:
                    logging.warning(f"Subfinder stderr: {error_output}")
                    
        except FileNotFoundError:
            print_status("    ⚠️  Subfinder not installed or not in PATH", "warning")
            print_status("    💡 Install from: https://github.com/projectdiscovery/subfinder", "info")
        except Exception as e:
            logging.error(f"Subfinder error: {e}")

    async def _brute_force_subdomains(self):
        """Fallback method if subfinder is not available"""
        common_subs = ['www', 'api', 'admin', 'test', 'dev', 'staging', 'mail', 'ftp', 'blog', 'shop']
        discovered = []
        
        # This would be enhanced with actual DNS lookups in production
        for sub in common_subs:
            potential_domain = f"{sub}.{self.domain}"
            if potential_domain not in self.discovered_subs:
                self.discovered_subs.add(potential_domain)
                self.dm.add_subdomain(potential_domain)
                discovered.append(potential_domain)
        
        if discovered:
            print_status(f"    ✅ Found {len(discovered)} common subdomains", "success")

    async def find_subdomains(self):
        """Public method called by main scanner"""
        print_status(f"    Scanning for subdomains of {self.domain}...", "info")
        
        if self.use_subfinder:
            await self._run_subfinder()
        
        # If no subdomains found or subfinder not available, try fallback
        if not self.discovered_subs:
            await self._brute_force_subdomains()
        
        # Add main domain to discovered if not already there
        if self.domain not in self.discovered_subs:
            self.discovered_subs.add(self.domain)
            self.dm.add_subdomain(self.domain)
        
        print_status(f"    📊 Total unique domains: {len(self.discovered_subs)}", "info")
        return list(self.discovered_subs)

    def get_stats(self):
        """Get subdomain finding statistics"""
        return {
            'total_subdomains': len(self.discovered_subs),
            'using_subfinder': self.use_subfinder
        }
