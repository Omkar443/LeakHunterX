#!/usr/bin/env python3
"""
LeakHunterX - Advanced AI Analyzer
Professional Grade with OpenRouter Support
"""

import logging
import asyncio
import aiohttp
import json
import yaml
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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

class AIAnalyzer:
    """
    Advanced AI Analyzer with OpenRouter Support
    Robust error handling and comprehensive features
    """

    def __init__(self, config_file: str = "config/secrets.json"):
        self.logger = logging.getLogger("LeakHunterX.AIAnalyzer")
        self.config_file = config_file
        self.api_keys = self._load_api_keys()
        self.available_models = self._discover_available_models()
        self.session = None
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'tokens_used': 0,
            'batch_operations': 0,
            'findings_validated': 0
        }

        self.logger.info(f"AI Analyzer initialized with {len(self.available_models)} available models")

    def _load_api_keys(self) -> Dict[str, List[str]]:
        """Load API keys from configuration file"""
        default_keys = {
            'openrouter': [],
            'openai': [],
            'google': [],
            'anthropic': [],
            'groq': []
        }

        try:
            if not os.path.exists(self.config_file):
                self.logger.warning(f"Config file not found: {self.config_file}")
                return default_keys

            with open(self.config_file, 'r') as f:
                # Handle JSON format (secrets.json)
                if self.config_file.endswith('.json'):
                    secrets = json.load(f)
                    openrouter_config = secrets.get("OPENROUTER_CONFIG", {})
                    api_keys_list = openrouter_config.get("api_keys", [])
                    
                    if api_keys_list:
                        default_keys['openrouter'] = api_keys_list
                        self.logger.info(f"Loaded {len(api_keys_list)} OpenRouter API keys")
                    else:
                        self.logger.warning("No API keys found in config")

                # Handle YAML format
                elif self.config_file.endswith(('.yaml', '.yml')):
                    loaded_keys = yaml.safe_load(f) or {}
                    for key, value in loaded_keys.items():
                        if key in default_keys and isinstance(value, list):
                            default_keys[key] = value

        except Exception as e:
            self.logger.error(f"Error loading API keys: {e}")

        return default_keys

    def _discover_available_models(self) -> Dict[str, List[str]]:
        """Discover which AI models are available based on API keys"""
        available_models = {}

        # OpenRouter models (your current setup)
        if self.api_keys.get('openrouter'):
            available_models['openrouter'] = [
                'deepseek/deepseek-r1',
                'meta-llama/llama-3-70b-instruct', 
                'google/gemini-pro',
                'anthropic/claude-3-sonnet'
            ]
            self.logger.info("OpenRouter models available")

        # OpenAI models
        if self.api_keys.get('openai'):
            available_models['openai'] = [
                'gpt-3.5-turbo',
                'gpt-4',
                'gpt-4-turbo'
            ]
            self.logger.info("OpenAI models available")

        # Google AI models
        if self.api_keys.get('google'):
            available_models['google'] = [
                'gemini-pro',
                'models/gemini-pro'
            ]
            self.logger.info("Google AI models available")

        # Fallback to mock mode if no API keys
        if not available_models:
            available_models['mock'] = ['mock-ai']
            self.logger.warning("No API keys found - using mock AI mode")

        return available_models

    async def setup_session(self):
        """Initialize aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=5, verify_ssl=False)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def analyze_with_openrouter(self, prompt: str, model: str = "deepseek/deepseek-r1") -> AIResponse:
        """Analyze with OpenRouter API"""
        api_key = self.api_keys['openrouter'][0] if self.api_keys.get('openrouter') else None
        if not api_key:
            return AIResponse("", model, 0, False, "No OpenRouter API key")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
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
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    tokens_used = result.get('usage', {}).get('total_tokens', 0)

                    self.stats['successful_requests'] += 1
                    self.stats['tokens_used'] += tokens_used

                    return AIResponse(content, model, tokens_used, True)
                else:
                    error_text = await response.text()
                    self.stats['failed_requests'] += 1
                    return AIResponse("", model, 0, False, f"HTTP {response.status}: {error_text}")

        except Exception as e:
            self.stats['failed_requests'] += 1
            return AIResponse("", model, 0, False, f"OpenRouter error: {e}")

    async def analyze_with_openai(self, prompt: str, model: str = "gpt-3.5-turbo") -> AIResponse:
        """Analyze with OpenAI API"""
        api_key = self.api_keys['openai'][0] if self.api_keys.get('openai') else None
        if not api_key:
            return AIResponse("", model, 0, False, "No OpenAI API key")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
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
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    tokens_used = result.get('usage', {}).get('total_tokens', 0)

                    self.stats['successful_requests'] += 1
                    self.stats['tokens_used'] += tokens_used

                    return AIResponse(content, model, tokens_used, True)
                else:
                    error_text = await response.text()
                    self.stats['failed_requests'] += 1
                    return AIResponse("", model, 0, False, f"HTTP {response.status}: {error_text}")

        except Exception as e:
            self.stats['failed_requests'] += 1
            return AIResponse("", model, 0, False, f"OpenAI error: {e}")

    async def analyze_with_mock(self, prompt: str, model: str = "mock-ai") -> AIResponse:
        """Mock AI analysis for testing without API keys"""
        await asyncio.sleep(0.5)

        # Generate realistic mock analysis
        if any(keyword in prompt.lower() for keyword in ['leak', 'secret', 'key', 'password', 'token']):
            analysis = """Security Analysis Results:

CRITICAL FINDINGS:
1. API Key Exposure: Google Maps API key found in client-side code
2. Hardcoded Credentials: Database connection string visible
3. JWT Secret: Signing key exposed in JavaScript

HIGH RISK:
- AWS access key in configuration file
- Stripe secret key in environment variables

RECOMMENDATIONS:
- Rotate all exposed keys immediately
- Implement proper secret management
- Move sensitive data to environment variables"""
        else:
            analysis = "Security Analysis: No critical issues detected. Code appears secure."

        tokens_used = len(analysis.split()) * 1.3

        self.stats['successful_requests'] += 1
        self.stats['tokens_used'] += int(tokens_used)

        return AIResponse(analysis, model, int(tokens_used), True)

    async def analyze_content(self, content: str, context: str = "security analysis") -> AIResponse:
        """
        Analyze content with AI - Main method
        """
        await self.setup_session()
        self.stats['total_requests'] += 1

        # Create optimized prompt
        prompt = self._create_analysis_prompt(content, context)

        # Try available models in order of preference
        models_to_try = []

        if 'openrouter' in self.available_models:
            models_to_try.extend([('openrouter', model) for model in self.available_models['openrouter']])
        if 'openai' in self.available_models:
            models_to_try.extend([('openai', model) for model in self.available_models['openai']])
        if 'google' in self.available_models:
            models_to_try.extend([('google', model) for model in self.available_models['google']])
        if 'mock' in self.available_models:
            models_to_try.extend([('mock', model) for model in self.available_models['mock']])

        # Try each model until one works
        for provider, model in models_to_try:
            self.logger.info(f"Analyzing with {provider}/{model}...")

            try:
                if provider == 'openrouter':
                    response = await self.analyze_with_openrouter(prompt, model)
                elif provider == 'openai':
                    response = await self.analyze_with_openai(prompt, model)
                elif provider == 'mock':
                    response = await self.analyze_with_mock(prompt, model)
                else:
                    response = await self.analyze_with_mock(prompt, model)

                if response.success:
                    self.logger.info(f"AI analysis successful with {model} ({response.tokens_used} tokens)")
                    return response
                else:
                    self.logger.warning(f"AI analysis failed with {model}: {response.error}")
                    continue

            except Exception as e:
                self.logger.error(f"Exception with {provider}/{model}: {e}")
                continue

        return AIResponse("", "none", 0, False, "All AI models failed")

    def _create_analysis_prompt(self, content: str, context: str) -> str:
        """Create optimized prompt for security analysis"""
        content_preview = content[:6000] + "..." if len(content) > 6000 else content

        prompt = f"""
Perform a comprehensive security analysis on the following {context}:

CONTENT TO ANALYZE:
{content_preview}

Please analyze this content and identify:
1. API keys, tokens, or secrets that may be exposed
2. Security misconfigurations or vulnerabilities  
3. Sensitive information disclosure
4. Hardcoded credentials or passwords
5. Any other security concerns

Format your response as:
- Start with overall risk assessment (Low/Medium/High/Critical)
- List specific findings with clear descriptions
- Provide recommendations for remediation
- Be concise but thorough

Focus on actionable security insights.
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
                'model_used': response.model,
                'tokens_used': response.tokens_used
            }
        else:
            return {
                'success': False,
                'file_url': file_url,
                'error': response.error,
                'model_used': response.model,
                'tokens_used': 0
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
            lines = analysis.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Detect risk level
                lower_line = line.lower()
                if 'risk assessment:' in lower_line or 'risk level:' in lower_line:
                    if 'critical' in lower_line:
                        findings['risk_level'] = 'Critical'
                    elif 'high' in lower_line:
                        findings['risk_level'] = 'High'
                    elif 'medium' in lower_line:
                        findings['risk_level'] = 'Medium'
                    elif 'low' in lower_line:
                        findings['risk_level'] = 'Low'

                # Detect findings
                elif any(keyword in lower_line for keyword in ['finding', 'issue', 'vulnerability', 'exposure']):
                    if 'critical' in lower_line:
                        findings['critical_findings'].append(line)
                    elif 'high' in lower_line:
                        findings['high_findings'].append(line)
                    elif 'medium' in lower_line:
                        findings['medium_findings'].append(line)
                    elif 'low' in lower_line:
                        findings['low_findings'].append(line)
                    else:
                        findings['medium_findings'].append(line)

                # Detect recommendations
                elif 'recommend' in lower_line:
                    findings['recommendations'].append(line)

        except Exception as e:
            self.logger.error(f"Error parsing AI findings: {e}")

        return findings

    async def batch_analyze_js_files(self, js_files: List[Dict]) -> Dict[str, Any]:
        """Batch analyze multiple JS files"""
        self.stats['batch_operations'] += 1

        results = {
            'total_files': len(js_files),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_tokens_used': 0,
            'files_analyzed': []
        }

        # Limit concurrent requests
        semaphore = asyncio.Semaphore(3)

        async def analyze_single_file(js_file):
            async with semaphore:
                file_url = js_file.get('url', '')
                content = js_file.get('content', '')

                if not content and file_url:
                    content = f"JavaScript file from {file_url}"

                analysis_result = await self.analyze_js_for_leaks(content, file_url)
                return analysis_result

        # Process files concurrently
        tasks = [analyze_single_file(js_file) for js_file in js_files]
        
        completed = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            results['files_analyzed'].append(result)

            if result['success']:
                results['successful_analyses'] += 1
                results['total_tokens_used'] += result.get('tokens_used', 0)
            else:
                results['failed_analyses'] += 1

            completed += 1
            if completed % 5 == 0:
                self.logger.info(f"AI Analysis Progress: {completed}/{len(js_files)} files")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get AI analyzer statistics"""
        success_rate = (self.stats['successful_requests'] / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0

        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': round(success_rate, 1),
            'tokens_used': self.stats['tokens_used'],
            'batch_operations': self.stats['batch_operations'],
            'findings_validated': self.stats['findings_validated'],
            'available_models': self.available_models
        }

    async def __aenter__(self):
        """Async context manager"""
        await self.setup_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup"""
        await self.close_session()

# Test function
async def test_ai_analyzer():
    """Test the AI analyzer"""
    logging.basicConfig(level=logging.INFO)

    async with AIAnalyzer() as ai:
        # Test with sample content
        sample_js = """
        const apiKey = "AIzaSyABC123def456ghi789jkl";
        const dbConfig = {
            host: "localhost",
            user: "admin", 
            password: "secret123",
            database: "production"
        };
        const jwtSecret = "my-super-secret-jwt-key";
        """

        print("Testing AI Analyzer...")
        result = await ai.analyze_js_for_leaks(sample_js, "test.js")

        print(f"Success: {result['success']}")
        print(f"Model: {result.get('model_used')}")
        print(f"Tokens: {result.get('tokens_used')}")

        if result['success']:
            print(f"Analysis preview: {result['analysis'][:200]}...")

        stats = ai.get_stats()
        print(f"Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_ai_analyzer())
