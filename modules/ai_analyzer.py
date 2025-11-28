#!/usr/bin/env python3
"""
LeakHunterX - Advanced AI Analyzer
Professional Grade with OpenRouter Support + Intelligent API Key Management
"""

import logging
import asyncio
import aiohttp
import json
import yaml
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class APIKeyStatus:
    """API key status tracking"""
    key: str
    provider: str
    is_active: bool = False
    last_used: float = 0
    error_count: int = 0
    success_count: int = 0
    tokens_used: int = 0

@dataclass
class AIRequest:
    """AI request data structure"""
    prompt: str
    model: str
    max_tokens: int = 1500
    temperature: float = 0.1

@dataclass
class AIResponse:
    """AI response data structure"""
    content: str
    model: str
    tokens_used: int
    success: bool
    error: Optional[str] = None
    provider: Optional[str] = None
    api_key_index: Optional[int] = None

class AIAnalyzer:
    """
    Advanced AI Analyzer with Intelligent API Key Management
    Robust error handling, key rotation, and comprehensive features
    """

    def __init__(self, config_file: str = "config/secrets.json"):
        self.logger = logging.getLogger("LeakHunterX.AIAnalyzer")
        self.config_file = config_file
        self.api_keys_status: Dict[str, List[APIKeyStatus]] = {}
        
        # Initialize model priorities FIRST
        self.model_priority = {
            'openrouter': [
                'deepseek/deepseek-r1',
                'meta-llama/llama-3-70b-instruct', 
                'google/gemini-pro',
                'anthropic/claude-3-sonnet'
            ],
            'openai': [
                'gpt-4-turbo',
                'gpt-4',
                'gpt-3.5-turbo'
            ],
            'google': [
                'gemini-pro',
                'models/gemini-pro'
            ],
            'anthropic': [
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ],
            'mock': ['mock-ai']
        }
        
        # Now initialize other attributes
        self.available_models = self._discover_available_models()
        self.session = None
        self.initialized = False
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'tokens_used': 0,
            'batch_operations': 0,
            'findings_validated': 0,
            'api_key_rotations': 0,
            'connectivity_tests': 0
        }

        self.logger.info(f"AI Analyzer initialized with {len(self.available_models)} available providers")

    def _load_api_keys(self) -> Dict[str, List[APIKeyStatus]]:
        """Load and initialize API keys from configuration file"""
        api_keys_status = {
            provider: [] for provider in ['openrouter', 'openai', 'google', 'anthropic', 'mock']
        }

        try:
            if not os.path.exists(self.config_file):
                self.logger.warning(f"Config file not found: {self.config_file}")
                return api_keys_status

            with open(self.config_file, 'r') as f:
                # Handle JSON format (secrets.json)
                if self.config_file.endswith('.json'):
                    secrets = json.load(f)
                    openrouter_config = secrets.get("OPENROUTER_CONFIG", {})
                    api_keys_list = openrouter_config.get("api_keys", [])
                    
                    if api_keys_list:
                        for key in api_keys_list:
                            if key and key.strip():
                                api_keys_status['openrouter'].append(
                                    APIKeyStatus(key=key.strip(), provider='openrouter')
                                )
                        self.logger.info(f"Loaded {len(api_keys_list)} OpenRouter API keys")

                # Handle YAML format (api_keys.yaml)
                elif self.config_file.endswith(('.yaml', '.yml')):
                    loaded_keys = yaml.safe_load(f) or {}
                    for provider_name, keys in loaded_keys.items():
                        if provider_name in api_keys_status and isinstance(keys, list):
                            for key in keys:
                                if key and key.strip():
                                    api_keys_status[provider_name].append(
                                        APIKeyStatus(key=key.strip(), provider=provider_name)
                                    )
                            self.logger.info(f"Loaded {len(keys)} {provider_name} API keys")

            # Log summary
            active_providers = []
            for provider, keys in api_keys_status.items():
                if keys and provider != 'mock':
                    active_providers.append(f"{provider}({len(keys)})")
            
            if active_providers:
                self.logger.info(f"API Providers loaded: {', '.join(active_providers)}")
            else:
                self.logger.warning("No API keys found - will use mock mode")

        except Exception as e:
            self.logger.error(f"Error loading API keys from {self.config_file}: {e}")

        return api_keys_status

    def _discover_available_models(self) -> Dict[str, List[str]]:
        """Discover which AI models are available based on API keys"""
        available_models = {}
        self.api_keys_status = self._load_api_keys()

        # Check each provider for available keys
        for provider in ['openrouter', 'openai', 'google', 'anthropic']:
            if self.api_keys_status.get(provider):
                available_models[provider] = self.model_priority.get(provider, [])
                self.logger.info(f"{provider.capitalize()} models available")

        # Fallback to mock mode if no API keys
        if not available_models or all(not keys for keys in self.api_keys_status.values()):
            available_models['mock'] = self.model_priority['mock']
            self.logger.warning("No valid API keys found - using mock AI mode")

        return available_models

    async def setup_session(self):
        """Initialize aiohttp session with proper configuration"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=45)
            connector = aiohttp.TCPConnector(
                limit=10, 
                limit_per_host=5, 
                verify_ssl=False,
                force_close=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout, 
                connector=connector,
                headers={'User-Agent': 'LeakHunterX/1.0'}
            )

    async def close_session(self):
        """Close aiohttp session properly"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.debug("AI analyzer session closed")

    async def test_api_key_connectivity(self, key_status: APIKeyStatus) -> bool:
        """Test if an API key is active and working"""
        self.stats['connectivity_tests'] += 1
        
        test_prompt = "Respond with exactly: CONNECTIVITY_TEST_OK"
        
        try:
            if key_status.provider == 'openrouter':
                response = await self._make_openrouter_request(test_prompt, "deepseek/deepseek-r1", key_status)
            elif key_status.provider == 'openai':
                response = await self._make_openai_request(test_prompt, "gpt-3.5-turbo", key_status)
            else:
                # For other providers, assume active for now
                return True

            key_status.is_active = response.success
            if response.success:
                key_status.success_count += 1
                self.logger.info(f"✅ API key test PASSED for {key_status.provider}")
            else:
                key_status.error_count += 1
                self.logger.warning(f"❌ API key test FAILED for {key_status.provider}: {response.error}")

            return response.success

        except Exception as e:
            self.logger.error(f"API key connectivity test failed: {e}")
            key_status.is_active = False
            key_status.error_count += 1
            return False

    async def initialize_connectivity(self):
        """Initialize and test all API keys"""
        if self.initialized:
            return

        await self.setup_session()
        self.logger.info("Testing API key connectivity...")

        # Test all API keys
        connectivity_tasks = []
        for provider, keys in self.api_keys_status.items():
            for key_status in keys:
                if provider != 'mock':
                    connectivity_tasks.append(self.test_api_key_connectivity(key_status))

        if connectivity_tasks:
            results = await asyncio.gather(*connectivity_tasks, return_exceptions=True)
            active_keys = sum(1 for result in results if result is True)
            self.logger.info(f"Connectivity test complete: {active_keys}/{len(connectivity_tasks)} API keys active")

        self.initialized = True

    def _get_best_api_key(self, provider: str) -> Optional[APIKeyStatus]:
        """Get the best available API key for a provider"""
        if provider not in self.api_keys_status or not self.api_keys_status[provider]:
            return None

        active_keys = [key for key in self.api_keys_status[provider] if key.is_active]
        
        if not active_keys:
            return None

        # Sort by success count (descending) and error count (ascending)
        active_keys.sort(key=lambda x: (-x.success_count, x.error_count))
        
        best_key = active_keys[0]
        best_key.last_used = time.time()
        
        return best_key

    async def _make_openrouter_request(self, prompt: str, model: str, key_status: APIKeyStatus) -> AIResponse:
        """Make request to OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key_status.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://leakhunterx.com",
            "X-Title": "LeakHunterX"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.1
        }

        try:
            async with self.session.post(url, headers=headers, json=data) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    result = json.loads(response_text)
                    content = result['choices'][0]['message']['content']
                    tokens_used = result.get('usage', {}).get('total_tokens', 0)

                    key_status.success_count += 1
                    key_status.tokens_used += tokens_used
                    
                    return AIResponse(
                        content=content, 
                        model=model, 
                        tokens_used=tokens_used, 
                        success=True,
                        provider='openrouter',
                        api_key_index=self.api_keys_status['openrouter'].index(key_status)
                    )
                else:
                    key_status.error_count += 1
                    error_msg = f"HTTP {response.status}"
                    try:
                        error_data = json.loads(response_text)
                        error_msg = error_data.get('error', {}).get('message', error_msg)
                    except:
                        pass
                    
                    # Deactivate key on certain errors
                    if response.status in [401, 403, 429]:
                        key_status.is_active = False
                        self.logger.warning(f"API key deactivated due to error {response.status}")
                    
                    return AIResponse(
                        content="", 
                        model=model, 
                        tokens_used=0, 
                        success=False, 
                        error=error_msg,
                        provider='openrouter'
                    )

        except aiohttp.ClientError as e:
            key_status.error_count += 1
            return AIResponse(
                content="", 
                model=model, 
                tokens_used=0, 
                success=False, 
                error=f"Network error: {e}",
                provider='openrouter'
            )
        except Exception as e:
            key_status.error_count += 1
            return AIResponse(
                content="", 
                model=model, 
                tokens_used=0, 
                success=False, 
                error=f"Unexpected error: {e}",
                provider='openrouter'
            )

    async def _make_openai_request(self, prompt: str, model: str, key_status: APIKeyStatus) -> AIResponse:
        """Make request to OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key_status.key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.1
        }

        try:
            async with self.session.post(url, headers=headers, json=data) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    result = json.loads(response_text)
                    content = result['choices'][0]['message']['content']
                    tokens_used = result.get('usage', {}).get('total_tokens', 0)

                    key_status.success_count += 1
                    key_status.tokens_used += tokens_used
                    
                    return AIResponse(
                        content=content, 
                        model=model, 
                        tokens_used=tokens_used, 
                        success=True,
                        provider='openai',
                        api_key_index=self.api_keys_status['openai'].index(key_status)
                    )
                else:
                    key_status.error_count += 1
                    error_msg = f"HTTP {response.status}: {response_text}"
                    
                    if response.status in [401, 403, 429]:
                        key_status.is_active = False
                    
                    return AIResponse(
                        content="", 
                        model=model, 
                        tokens_used=0, 
                        success=False, 
                        error=error_msg,
                        provider='openai'
                    )

        except Exception as e:
            key_status.error_count += 1
            return AIResponse(
                content="", 
                model=model, 
                tokens_used=0, 
                success=False, 
                error=f"OpenAI error: {e}",
                provider='openai'
            )

    async def _make_mock_request(self, prompt: str, model: str) -> AIResponse:
        """Mock AI analysis for testing without API keys"""
        await asyncio.sleep(0.3)

        # Generate realistic mock analysis
        if any(keyword in prompt.lower() for keyword in ['leak', 'secret', 'key', 'password', 'token']):
            analysis = """SECURITY ANALYSIS RESULTS:

RISK LEVEL: CRITICAL

CRITICAL FINDINGS:
1. API Key Exposure: Google Maps API key found in client-side JavaScript
2. Hardcoded Credentials: Database connection string visible in source code
3. JWT Secret: Signing key exposed in configuration

HIGH RISK FINDINGS:
- AWS access key in configuration file
- Stripe secret key in environment variables
- OAuth client secret hardcoded

RECOMMENDATIONS:
- Rotate all exposed keys immediately
- Implement proper secret management system
- Move sensitive data to secure environment variables
- Conduct security code review"""
        else:
            analysis = "SECURITY ANALYSIS: No critical security issues detected. Code appears properly secured."

        tokens_used = len(analysis.split()) * 1.3

        return AIResponse(
            content=analysis, 
            model=model, 
            tokens_used=int(tokens_used), 
            success=True,
            provider='mock'
        )

    async def analyze_content(self, content: str, context: str = "security analysis") -> AIResponse:
        """
        Analyze content with AI - Main method with intelligent fallbacks
        """
        if not self.initialized:
            await self.initialize_connectivity()

        self.stats['total_requests'] += 1

        # Create optimized prompt
        prompt = self._create_analysis_prompt(content, context)

        # Try providers in order of preference
        providers_to_try = ['openrouter', 'openai', 'google', 'anthropic', 'mock']

        for provider in providers_to_try:
            if provider not in self.available_models:
                continue

            # Try models for this provider
            for model in self.available_models[provider]:
                self.logger.info(f"🔍 Analyzing with {provider}/{model}...")

                try:
                    if provider == 'openrouter':
                        key_status = self._get_best_api_key(provider)
                        if not key_status:
                            self.logger.warning(f"No active API keys for {provider}")
                            continue
                        response = await self._make_openrouter_request(prompt, model, key_status)
                    
                    elif provider == 'openai':
                        key_status = self._get_best_api_key(provider)
                        if not key_status:
                            continue
                        response = await self._make_openai_request(prompt, model, key_status)
                    
                    elif provider == 'mock':
                        response = await self._make_mock_request(prompt, model)
                    
                    else:
                        # Skip unsupported providers for now
                        continue

                    if response.success:
                        self.stats['successful_requests'] += 1
                        self.stats['tokens_used'] += response.tokens_used
                        
                        self.logger.info(f"✅ AI analysis successful with {provider}/{model} "
                                      f"({response.tokens_used} tokens)")
                        return response
                    else:
                        self.logger.warning(f"❌ AI analysis failed with {provider}/{model}: {response.error}")
                        continue

                except Exception as e:
                    self.logger.error(f"Exception with {provider}/{model}: {e}")
                    continue

        # All attempts failed
        self.stats['failed_requests'] += 1
        return AIResponse(
            content="", 
            model="none", 
            tokens_used=0, 
            success=False, 
            error="All AI providers and models failed"
        )

    def _create_analysis_prompt(self, content: str, context: str) -> str:
        """Create optimized prompt for security analysis"""
        content_preview = content[:6000] + "..." if len(content) > 6000 else content

        prompt = f"""
Perform a comprehensive security analysis on the following {context}:

CONTENT TO ANALYZE:
{content_preview}

ANALYSIS REQUIREMENTS:
1. Start with overall RISK LEVEL (Low/Medium/High/Critical)
2. List specific security findings with clear descriptions
3. Categorize findings by severity (Critical/High/Medium/Low)
4. Provide actionable recommendations for remediation
5. Focus on exposed secrets, keys, tokens, and credentials

RESPONSE FORMAT:
RISK LEVEL: [Your Assessment]

CRITICAL FINDINGS:
- [Finding 1]
- [Finding 2]

HIGH RISK FINDINGS:  
- [Finding 1]

MEDIUM RISK FINDINGS:
- [Finding 1]

RECOMMENDATIONS:
- [Recommendation 1]

Be concise, thorough, and security-focused.
"""

        return prompt.strip()

    async def analyze_js_for_leaks(self, js_content: str, file_url: str = "") -> Dict[str, Any]:
        """Analyze JavaScript content for potential leaks"""
        context = f"JavaScript file analysis{' - ' + file_url if file_url else ''}"
        response = await self.analyze_content(js_content, context)

        if response.success:
            findings = self._parse_ai_findings(response.content)
            self.stats['findings_validated'] += len(findings.get('critical_findings', []))

            return {
                'success': True,
                'file_url': file_url,
                'analysis': response.content,
                'findings': findings,
                'model_used': f"{response.provider}/{response.model}" if response.provider else response.model,
                'tokens_used': response.tokens_used,
                'provider': response.provider if response.provider else 'unknown'
            }
        else:
            return {
                'success': False,
                'file_url': file_url,
                'error': response.error,
                'model_used': response.model,
                'tokens_used': 0,
                'provider': response.provider if response.provider else 'unknown'
            }

    def _parse_ai_findings(self, analysis: str) -> Dict[str, Any]:
        """Parse AI analysis response into structured findings"""
        findings = {
            'risk_level': 'Unknown',
            'critical_findings': [],
            'high_findings': [],
            'medium_findings': [],
            'low_findings': [],
            'recommendations': []
        }

        try:
            current_section = None
            
            for line in analysis.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Detect risk level
                if line.upper().startswith('RISK LEVEL:'):
                    findings['risk_level'] = line.split(':', 1)[1].strip()
                    continue

                # Detect section headers
                if line.upper().startswith('CRITICAL'):
                    current_section = 'critical_findings'
                    continue
                elif line.upper().startswith('HIGH'):
                    current_section = 'high_findings'
                    continue
                elif line.upper().startswith('MEDIUM'):
                    current_section = 'medium_findings'
                    continue
                elif line.upper().startswith('LOW'):
                    current_section = 'low_findings'
                    continue
                elif line.upper().startswith('RECOMMEND'):
                    current_section = 'recommendations'
                    continue

                # Add content to current section
                if current_section and line.startswith('-'):
                    content = line[1:].strip()
                    if content and current_section in findings:
                        findings[current_section].append(content)

        except Exception as e:
            self.logger.error(f"Error parsing AI findings: {e}")

        return findings

    async def batch_analyze_js_files(self, js_files: List[Dict]) -> Dict[str, Any]:
        """Batch analyze multiple JS files with intelligent concurrency control"""
        self.stats['batch_operations'] += 1

        results = {
            'total_files': len(js_files),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_tokens_used': 0,
            'files_analyzed': []
        }

        if not js_files:
            return results

        # Limit concurrent requests to avoid rate limiting
        semaphore = asyncio.Semaphore(2)  # Conservative limit

        async def analyze_single_file(js_file):
            async with semaphore:
                file_url = js_file.get('url', '')
                content = js_file.get('content', '')

                if not content and file_url:
                    content = f"JavaScript file from {file_url}"

                analysis_result = await self.analyze_js_for_leaks(content, file_url)
                return analysis_result

        # Process files with progress tracking
        tasks = [analyze_single_file(js_file) for js_file in js_files]
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results['files_analyzed'].append(result)

                if result['success']:
                    results['successful_analyses'] += 1
                    results['total_tokens_used'] += result.get('tokens_used', 0)
                else:
                    results['failed_analyses'] += 1

                # Progress logging
                if (i + 1) % 5 == 0 or (i + 1) == len(js_files):
                    self.logger.info(f"📊 AI Analysis Progress: {i + 1}/{len(js_files)} files "
                                  f"({results['successful_analyses']} successful)")

            except Exception as e:
                self.logger.error(f"Error in batch analysis task: {e}")
                results['failed_analyses'] += 1
                results['files_analyzed'].append({
                    'success': False,
                    'file_url': js_files[i].get('url', 'unknown'),
                    'error': str(e),
                    'tokens_used': 0
                })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive AI analyzer statistics"""
        success_rate = (self.stats['successful_requests'] / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0

        # API key status summary
        api_key_status = {}
        for provider, keys in self.api_keys_status.items():
            if keys:
                active = sum(1 for k in keys if k.is_active)
                api_key_status[provider] = {
                    'total': len(keys),
                    'active': active,
                    'inactive': len(keys) - active
                }

        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': round(success_rate, 1),
            'tokens_used': self.stats['tokens_used'],
            'batch_operations': self.stats['batch_operations'],
            'findings_validated': self.stats['findings_validated'],
            'api_key_rotations': self.stats['api_key_rotations'],
            'connectivity_tests': self.stats['connectivity_tests'],
            'api_key_status': api_key_status,
            'available_models': self.available_models
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_connectivity()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup"""
        await self.close_session()

# Simple test function
async def test_ai_analyzer():
    """Test the AI analyzer"""
    logging.basicConfig(level=logging.INFO)

    async with AIAnalyzer() as ai:
        print("Testing AI Analyzer...")
        
        # Test with sample content
        sample_js = """
        const apiKey = "AIzaSyABC123def456ghi789jkl";
        const dbConfig = {
            host: "localhost",
            user: "admin", 
            password: "secret123",
            database: "production"
        };
        """

        result = await ai.analyze_js_for_leaks(sample_js, "test.js")
        print(f"Success: {result['success']}")
        print(f"Model: {result.get('model_used')}")
        
        if result['success']:
            print(f"Analysis: {result['analysis'][:200]}...")
        
        stats = ai.get_stats()
        print(f"Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_ai_analyzer())