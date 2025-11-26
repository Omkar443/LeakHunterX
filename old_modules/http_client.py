import aiohttp
import asyncio
import random
from .constants import USER_AGENTS, DEFAULT_TIMEOUT, MAX_RETRIES
import logging

class AsyncHTTPClient:
    def __init__(self, timeout=DEFAULT_TIMEOUT, max_retries=MAX_RETRIES):
        self.timeout = timeout
        self.max_retries = max_retries
        self.sem = asyncio.Semaphore(50)  # concurrency limiter

    async def fetch(self, url):
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        for attempt in range(self.max_retries):
            try:
                async with self.sem:
                    async with aiohttp.ClientSession(headers=headers) as session:
                        async with session.get(url, timeout=self.timeout, ssl=False) as resp:
                            text = await resp.text(errors="ignore")
                            return url, text, resp.status
            except Exception as e:
                logging.debug(f"Retry {attempt+1}/{self.max_retries} for {url}: {e}")
                await asyncio.sleep(0.5)
        return url, None, None
