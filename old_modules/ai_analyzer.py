#!/usr/bin/env python3
"""
LeakHunterX AI Analyzer Module
Industry-Grade AI-Powered Security Analysis with Robust JSON Parsing
"""

import asyncio
import aiohttp
import json
import os
import time
import random
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class AIModel(Enum):
    DEEPSEEK_R1 = "deepseek/deepseek-r1"
    LLAMA_8B = "meta-llama/llama-3.1-8b-instruct"
    GEMINI_FLASH = "google/gemini-2.0-flash-exp"
    QWEN_CODER = "qwen/qwen-2.5-coder-32b-instruct"

@dataclass
class AIRequest:
    prompt: str
    operation: str
    max_tokens: int = 1000
    temperature: float = 0.1

class AIScoring:
    """
    Industry-Grade AI Scoring and Analysis Engine
    Features:
    - Nuclear-proof JSON parsing from ANY AI model
    - Clean professional interface
    - Multi-API key rotation with failover
    - Intelligent model selection and fallback
    - Comprehensive error handling and logging
    """

    def __init__(self, api_keys: List[str] = None):
        # API Configuration
        self.api_keys = api_keys or []
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Model Configuration - Priority order
        self.model_priority = [
            AIModel.DEEPSEEK_R1.value,
            AIModel.LLAMA_8B.value,
            AIModel.GEMINI_FLASH.value,
            AIModel.QWEN_CODER.value
        ]

        # State Management
        self.current_key_index = 0
        self.current_model_index = 0
        self.active_keys = []
        self.failed_keys = set()
        self.key_models = {}  # Track working models per key
        self.ai_working = False
        self.testing_in_progress = False
        self.testing_task = None
        self.testing_started = False

        # Statistics and Monitoring
        self.ai_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_processing_time': 0,
            'keys_used': 0,
            'keys_exhausted': 0,
            'models_used': set(),
            'average_response_time': 0
        }

        # Key-specific performance tracking
        self.key_stats = {}
        self.model_performance = {}

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5

        # Initialize if keys are provided
        if self.api_keys:
            self._initialize_key_stats()

    def _initialize_key_stats(self):
        """Initialize statistics for all API keys"""
        self.active_keys = self.api_keys.copy()
        for key in self.api_keys:
            key_short = key[-8:]
            self.key_stats[key_short] = {
                'requests_made': 0,
                'tokens_used': 0,
                'failures': 0,
                'success_rate': 0,
                'last_used': 0,
                'model': None,
                'average_response_time': 0
            }

    def start_silent_background_testing(self):
        """Start AI connection testing in background with clean output"""
        if self.testing_started or self.testing_in_progress:
            return self.testing_task

        self.testing_started = True
        self.testing_in_progress = True
        
        self.testing_task = asyncio.create_task(self._silent_comprehensive_connection_test())
        return self.testing_task

    async def _silent_comprehensive_connection_test(self) -> bool:
        """Comprehensive AI connection testing with clean output"""
        self._print_status("🤖 AI System: Testing connectivity...", "info")
        
        tasks = []
        for model in self.model_priority:
            for key in self.api_keys:
                if key in self.failed_keys:
                    continue
                task = self._test_single_combination_async(key, model)
                tasks.append(task)

        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=15.0)

            working_combinations = 0
            for result in results:
                if isinstance(result, tuple) and len(result) == 3:
                    key, model, success = result
                    if success:
                        key_short = key[-8:]
                        self.key_models[key_short] = model
                        self.key_stats[key_short]['model'] = model
                        working_combinations += 1

            if working_combinations > 0:
                self.ai_working = True
                failed_models = len(self.model_priority) - working_combinations
                status_msg = f"🔧 AI System: {working_combinations} models ready"
                if failed_models > 0:
                    status_msg += f" | {failed_models} unavailable"
                self._print_status(status_msg, "success")
                return True
            else:
                self._print_status("❌ AI System: No working models", "warning")
                return False

        except asyncio.TimeoutError:
            if self.key_models:
                self.ai_working = True
                self._print_status(f"🔧 AI System: {len(self.key_models)} models ready (partial)", "success")
                return True
            else:
                self._print_status("❌ AI System: Connection testing timed out", "warning")
                return False
        except Exception as e:
            self._print_status(f"❌ AI System testing failed: {e}", "warning")
            return False

    async def _test_single_combination_async(self, key: str, model: str) -> Tuple[str, str, bool]:
        """Test a single key-model combination asynchronously"""
        test_prompt = "Respond with exactly: OK"

        try:
            headers = self._build_headers(key)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 10,
                "temperature": 0.1
            }

            timeout = aiohttp.ClientTimeout(total=10)
            connector = aiohttp.TCPConnector(ssl=False)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.post(self.api_url, json=payload, headers=headers, ssl=False) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content'].strip()
                        return key, model, content == "OK"
                    else:
                        return key, model, False

        except Exception:
            return key, model, False

    async def ensure_ai_working(self) -> bool:
        """Ensure AI system is operational"""
        if not self.api_keys:
            self._print_status("❌ No API keys available", "warning")
            return False

        if self.ai_working:
            return True

        if self.testing_in_progress:
            if self.testing_task:
                try:
                    await asyncio.wait_for(self.testing_task, timeout=5.0)
                    return self.ai_working
                except asyncio.TimeoutError:
                    self._print_status("⚠️ AI testing taking longer than expected", "warning")
                    return len(self.key_models) > 0
        else:
            if not self.testing_started:
                self._print_status("🚀 Starting AI connection testing...", "info")
                success = await self._silent_comprehensive_connection_test()
                return success

        return False

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        """Build standardized headers for OpenRouter API"""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://leakhunterx.com",
            "X-Title": "LeakHunterX Security Scanner"
        }

    def _get_next_working_key_model(self) -> Tuple[Optional[str], Optional[str]]:
        """Get next available key-model combination using round-robin"""
        if not self.active_keys:
            return None, None

        for _ in range(len(self.active_keys)):
            key = self.active_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.active_keys)

            key_short = key[-8:]
            if key_short in self.key_models:
                return key, self.key_models[key_short]

        return None, None

    async def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    async def _make_ai_request(self, request: AIRequest, max_retries: int = 2) -> Any:
        """Make AI API request with exponential backoff retry"""
        for attempt in range(max_retries + 1):
            await self._rate_limit()

            key, model = self._get_next_working_key_model()
            if not key or not model:
                raise Exception("No working key-model combinations available")

            key_short = key[-8:]
            start_time = time.time()

            try:
                headers = self._build_headers(key)
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt(request.operation)
                        },
                        {
                            "role": "user",
                            "content": request.prompt
                        }
                    ],
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }

                connector = aiohttp.TCPConnector(ssl=False)
                timeout = aiohttp.ClientTimeout(total=30)
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        ssl=False
                    ) as response:

                        if response.status == 200:
                            result = await response.json()
                            end_time = time.time()
                            processing_time = end_time - start_time

                            content = result['choices'][0]['message']['content']
                            usage = result.get('usage', {})
                            tokens_used = usage.get('total_tokens', 0)

                            self._update_stats(key_short, model, tokens_used, processing_time, True)
                            self._print_status(f"✅ AI Analysis: Complete | {tokens_used} tokens | {processing_time:.1f}s", "success")

                            return content
                        else:
                            error_text = await response.text()
                            self._handle_api_error(response.status, error_text, key)
                            self._update_stats(key_short, model, 0, time.time() - start_time, False)
                            if attempt < max_retries:
                                continue
                            raise Exception(f"API error {response.status}")

            except asyncio.TimeoutError:
                self._mark_key_failed(key, "Request timeout")
                self._update_stats(key_short, model, 0, time.time() - start_time, False)
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                self._update_stats(key_short, model, 0, time.time() - start_time, False)
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

    def _handle_api_error(self, status_code: int, error_text: str, key: str):
        """Handle API errors with appropriate actions"""
        key_short = key[-8:]

        if status_code == 402 or "credit" in error_text.lower():
            self._mark_key_failed(key, "Credits exhausted")
        elif status_code == 429:
            self._print_status(f"  ⚠️ Rate limited for key ...{key_short}", "warning")
        elif status_code == 404 or "No endpoints found" in error_text:
            self._print_status(f"  ⚠️ Model not available for key ...{key_short}", "warning")
        elif status_code == 401:
            self._mark_key_failed(key, "Invalid API key")
        else:
            self._print_status(f"  ❌ API error {status_code} for key ...{key_short}", "debug")

    def _mark_key_failed(self, key: str, reason: str):
        """Mark a key as failed with reason"""
        if key in self.active_keys:
            self.active_keys.remove(key)
            self.failed_keys.add(key)
            key_short = key[-8:]
            self.key_stats[key_short]['failures'] += 1
            self._print_status(f"Key ...{key_short} marked as failed: {reason}", "warning")

    def _get_system_prompt(self, operation: str) -> str:
        """Get appropriate system prompt for operation type"""
        if operation == "prioritize_urls":
            return """CRITICAL: You MUST return ONLY valid JSON. No text, no explanations, no thinking.

REQUIRED JSON FORMAT:
[
  {
    "url": "full_url_here",
    "priority": "HIGH|MEDIUM|LOW", 
    "score": 1-10,
    "reason": "brief_security_reason"
  }
]

RULES:
1. Return ONLY the JSON array above
2. No other text before or after
3. Ensure all quotes, brackets, and braces are correct
4. Priority must be HIGH, MEDIUM, or LOW
5. Score must be 1-10

FAILURE TO RETURN PURE JSON WILL BREAK THE SYSTEM."""

        elif operation == "analyze_endpoints":
            return """Return ONLY valid JSON array. No explanations.

FORMAT:
[
  {
    "endpoint": "endpoint_path",
    "risk_level": "HIGH|MEDIUM|LOW",
    "testing_approaches": ["approach1", "approach2"],
    "potential_vulnerabilities": ["vuln1", "vuln2"]
  }
]

Return ONLY JSON array, nothing else."""

        elif operation == "validate_leaks":
            return """Return ONLY valid JSON array. No explanations.

FORMAT:
[
  {
    "original_type": "leak_type",
    "original_value": "leak_value",
    "is_valid": true|false,
    "confidence": 0.0-1.0,
    "risk_level": "CRITICAL|HIGH|MEDIUM|LOW"
  }
]

Return ONLY JSON array, nothing else."""

        elif operation == "generate_insights":
            return """Return ONLY valid JSON object. No explanations.

FORMAT:
{
  "risk_assessment": "CRITICAL|HIGH|MEDIUM|LOW",
  "key_findings_summary": "brief_summary",
  "testing_recommendations": ["rec1", "rec2"],
  "potential_impact": "impact_description"
}

Return ONLY JSON object, nothing else."""

        return "Return ONLY valid JSON. No explanations, no thinking, no markdown."

    def _update_stats(self, key_short: str, model: str, tokens: int, processing_time: float, success: bool):
        """Update comprehensive statistics"""
        self.ai_stats['total_requests'] += 1

        if success:
            self.ai_stats['successful_requests'] += 1
            self.ai_stats['total_tokens_used'] += tokens
            self.ai_stats['total_processing_time'] += processing_time
            self.ai_stats['models_used'].add(model)

            self.key_stats[key_short]['requests_made'] += 1
            self.key_stats[key_short]['tokens_used'] += tokens
            self.key_stats[key_short]['last_used'] = time.time()

            if model not in self.model_performance:
                self.model_performance[model] = {'total_requests': 0, 'total_time': 0}
            self.model_performance[model]['total_requests'] += 1
            self.model_performance[model]['total_time'] += processing_time

        else:
            self.ai_stats['failed_requests'] += 1
            self.key_stats[key_short]['failures'] += 1

    def _extract_json_from_response(self, content: str) -> Any:
        """
        ULTRA-ROBUST JSON extraction that can handle ANY AI model output
        This will extract JSON from literally any response format
        """
        if not content or not content.strip():
            self._print_status("AI returned empty response", "warning")
            return []

        original_content = content
        content = content.strip()

        # NUCLEAR CLEANING - Remove everything that's not JSON
        content = self._nuclear_content_cleanup(content)

        # MULTI-LAYER JSON EXTRACTION STRATEGY
        extraction_attempts = [
            self._attempt_direct_json_parse,
            self._attempt_json_array_extraction,
            self._attempt_json_object_extraction, 
            self._attempt_bracket_based_extraction,
            self._attempt_ai_thinking_extraction,
            self._attempt_line_by_line_reconstruction,
            self._attempt_manual_json_construction,
            self._create_emergency_fallback
        ]

        for i, attempt_method in enumerate(extraction_attempts):
            try:
                result = attempt_method(content)
                if result is not None:
                    return result
            except Exception:
                continue

        # If ALL methods fail, create emergency fallback
        self._print_status("All JSON extraction methods failed, using emergency fallback", "error")
        return self._create_emergency_fallback(original_content)

    def _nuclear_content_cleanup(self, content: str) -> str:
        """Remove EVERYTHING that could interfere with JSON parsing"""
        
        # Remove ALL thinking/explanation blocks (aggressive patterns)
        thinking_patterns = [
            r'<think>.*?</think>',
            r'<.*?>',
            r'```.*?```',
            r'`.*?`',
            r'Okay, let\'s.*?(?=\[|\{)',
            r'First, let me.*?(?=\[|\{)', 
            r'I need to.*?(?=\[|\{)',
            r'Thinking.*?(?=\[|\{)',
            r'Analysis.*?(?=\[|\{)',
            r'Here.*?(?=\[|\{)',
            r'Certainly!.*?(?=\[|\{)',
            r'I\'ll analyze.*?(?=\[|\{)',
            r'Based on.*?(?=\[|\{)',
            r'Let me.*?(?=\[|\{)',
            r'The URLs.*?(?=\[|\{)',
            r'Following.*?(?=\[|\{)',
            r'JSON.*?(?=\[|\{)',
            r'Output.*?(?=\[|\{)',
            r'Result.*?(?=\[|\{)',
            r'Response.*?(?=\[|\{)',
        ]
        
        for pattern in thinking_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove common AI response prefixes/suffixes
        content = re.sub(r'^[^{[]*', '', content)
        content = re.sub(r'[^}\]]*$', '', content)
        
        # Remove excessive whitespace but preserve structure
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()

    def _attempt_direct_json_parse(self, content: str) -> Any:
        """Attempt 1: Direct JSON parsing"""
        try:
            return json.loads(content)
        except:
            raise

    def _attempt_json_array_extraction(self, content: str) -> Any:
        """Attempt 2: Extract JSON array with flexible patterns"""
        
        array_patterns = [
            r'\[\s*\{.*?\}\s*\]',
            r'\[\s*\{[\s\S]*?\}\s*\]',
            r'\[[\s\S]*?\]',
        ]
        
        for pattern in array_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    cleaned = self._repair_json_string(match)
                    return json.loads(cleaned)
                except:
                    continue
        raise

    def _attempt_json_object_extraction(self, content: str) -> Any:
        """Attempt 3: Extract JSON objects and wrap in array"""
        
        object_pattern = r'\{\s*"url"\s*:\s*"[^"]*"[^}]*\}'
        matches = re.findall(object_pattern, content, re.DOTALL)
        
        if matches:
            objects = []
            for match in matches:
                try:
                    cleaned = self._repair_json_string(match)
                    obj = json.loads(cleaned)
                    if isinstance(obj, dict) and 'url' in obj:
                        objects.append(obj)
                except:
                    continue
            
            if objects:
                return objects
        raise

    def _attempt_bracket_based_extraction(self, content: str) -> Any:
        """Attempt 4: Bracket-based extraction (very robust)"""
        
        start = content.find('[')
        end = content.rfind(']')
        
        if start != -1 and end != -1 and end > start:
            json_content = content[start:end+1]
            try:
                cleaned = self._repair_json_string(json_content)
                return json.loads(cleaned)
            except:
                pass
        raise

    def _attempt_ai_thinking_extraction(self, content: str) -> Any:
        """Attempt 5: Extract from AI thinking patterns"""
        
        url_pattern = r'(https?://[^\s"\',]+)'
        urls = re.findall(url_pattern, content)
        
        if urls:
            prioritized = []
            for url in urls[:15]:
                url_context = content[max(0, content.find(url)-100):content.find(url)+100]
                
                priority = 'LOW'
                score = 3
                
                if any(kw in url_context.lower() for kw in ['high', 'critical', 'important', 'sensitive', 'auth', 'login', 'admin']):
                    priority = 'HIGH'
                    score = 9
                elif any(kw in url_context.lower() for kw in ['medium', 'moderate', 'user', 'api']):
                    priority = 'MEDIUM' 
                    score = 6
                
                prioritized.append({
                    'url': url,
                    'priority': priority,
                    'score': score,
                    'reason': 'AI context analysis',
                    'confidence': 0.7
                })
            
            return prioritized
        raise

    def _attempt_line_by_line_reconstruction(self, content: str) -> Any:
        """Attempt 6: Line-by-line JSON reconstruction"""
        
        lines = content.split('\n')
        json_lines = []
        in_json = False
        brace_stack = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('[') or line.startswith('{'):
                in_json = True
                brace_stack.append('[' if line.startswith('[') else '{')
            
            if in_json:
                json_lines.append(line)
                
                for char in line:
                    if char == '{': brace_stack.append('{')
                    elif char == '}': brace_stack.pop() if brace_stack and brace_stack[-1] == '{' else None
                    elif char == '[': brace_stack.append('[')  
                    elif char == ']': brace_stack.pop() if brace_stack and brace_stack[-1] == '[' else None
                
                if not brace_stack:
                    break
        
        if json_lines:
            reconstructed = ' '.join(json_lines)
            try:
                return json.loads(reconstructed)
            except:
                fixed = self._repair_json_string(reconstructed)
                try:
                    return json.loads(fixed)
                except:
                    pass
        raise

    def _attempt_manual_json_construction(self, content: str) -> Any:
        """Attempt 7: Manual JSON construction from patterns"""
        
        pattern = r'(https?://[^\s]+)[^"]*?(high|medium|low|critical)[^"]*?(\d+)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        if matches:
            results = []
            for url, priority, score in matches:
                priority = priority.upper()
                try:
                    score_val = min(10, max(1, int(score)))
                except:
                    score_val = 5
                    
                results.append({
                    'url': url.strip(),
                    'priority': priority,
                    'score': score_val,
                    'reason': 'Pattern extraction',
                    'confidence': 0.6
                })
            
            return results
        raise

    def _repair_json_string(self, json_str: str) -> str:
        """Comprehensive JSON repair for common issues"""
        
        # Fix unescaped quotes in strings
        json_str = re.sub(r'(?<!\\)"([^"]*?)(?<!\\)"', 
                         lambda m: '"' + m.group(1).replace('"', '\\"') + '"', json_str)
        
        # Fix missing commas between objects
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix unescaped control characters
        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        
        # Ensure proper array/object closure
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        # Remove any non-printable characters
        json_str = ''.join(char for char in json_str if char.isprintable() or char in ' \n\r\t')
        
        return json_str

    def _create_emergency_fallback(self, original_content: str) -> list:
        """Final emergency fallback when everything else fails"""
        self._print_status("Using emergency fallback response", "warning")
        
        url_pattern = r'https?://[^\s"\',]+'
        urls = list(set(re.findall(url_pattern, original_content)))[:10]
        
        if not urls:
            return []
        
        results = []
        for url in urls:
            priority = 'LOW'
            score = 3
            
            if any(kw in url.lower() for kw in ['api', 'auth', 'login', 'admin', 'token']):
                priority = 'HIGH'
                score = 8
            elif any(kw in url.lower() for kw in ['user', 'account', 'dashboard', 'config']):
                priority = 'MEDIUM'
                score = 5
                
            results.append({
                'url': url,
                'priority': priority,
                'score': score,
                'reason': 'Emergency fallback',
                'confidence': 0.3
            })
        
        return results

    def _normalize_ai_response(self, response: Any, operation: str) -> Any:
        """Normalize AI response to expected format with validation"""
        if not response:
            return [] if operation != "generate_insights" else {}

        try:
            if operation == "prioritize_urls":
                if isinstance(response, list):
                    return self._validate_prioritized_urls(response)
                elif isinstance(response, dict):
                    return self._validate_prioritized_urls(response.get('urls', response.get('prioritized_urls', [])))

            elif operation == "analyze_endpoints":
                if isinstance(response, list):
                    return response
                elif isinstance(response, dict):
                    return response.get('endpoints', [])

            elif operation == "validate_leaks":
                if isinstance(response, list):
                    return response
                elif isinstance(response, dict):
                    return response.get('leaks', response.get('validated_leaks', []))

            elif operation == "generate_insights":
                if isinstance(response, dict):
                    return self._validate_insights(response)
                else:
                    return {"error": "Invalid insights format"}

            return response if response else ([] if operation != "generate_insights" else {})

        except Exception as e:
            logger.error(f"Error normalizing AI response: {e}")
            return [] if operation != "generate_insights" else {}

    def _validate_prioritized_urls(self, urls: List[Dict]) -> List[Dict]:
        """Validate and normalize prioritized URLs"""
        validated = []
        for item in urls:
            if isinstance(item, dict) and item.get('url'):
                validated.append({
                    'url': item['url'],
                    'priority': item.get('priority', 'MEDIUM'),
                    'score': min(10, max(1, item.get('score', 5))),
                    'reason': item.get('reason', 'AI prioritized'),
                    'confidence': min(1.0, max(0.0, item.get('confidence', 0.7)))
                })
        return validated

    def _validate_insights(self, insights: Dict) -> Dict:
        """Validate and normalize insights"""
        return {
            'risk_assessment': insights.get('risk_assessment', 'UNKNOWN'),
            'key_findings_summary': insights.get('key_findings_summary', 'No significant findings'),
            'testing_recommendations': insights.get('testing_recommendations', ['Continue manual testing']),
            'potential_impact': insights.get('potential_impact', 'Unable to assess'),
            'confidence': min(1.0, max(0.0, insights.get('confidence', 0.0)))
        }

    # PUBLIC API METHODS

    async def prioritize_urls(self, urls: List[str], domain: str) -> List[Dict]:
        """AI-powered URL prioritization for security testing"""
        if not urls or not await self.ensure_ai_working():
            return self._fallback_prioritization(urls)

        urls_to_analyze = urls[:15]

        prompt = f"""Analyze these URLs from {domain} and prioritize them for security testing.
        Focus on authentication, API endpoints, admin interfaces, and sensitive data handlers.

        URLs to analyze:
        {chr(10).join(f"- {url}" for url in urls_to_analyze)}
        """

        try:
            request = AIRequest(prompt=prompt, operation="prioritize_urls")
            response = await self._make_ai_request(request)
            result = self._extract_json_from_response(response)
            normalized = self._normalize_ai_response(result, "prioritize_urls")

            if normalized:
                self._print_status(f"✅ AI Analysis: {len(normalized)} targets prioritized", "success")
                return normalized
            else:
                self._print_status("⚠️ AI returned empty prioritization, using fallback", "warning")
                return self._fallback_prioritization(urls)

        except Exception as e:
            self._print_status(f"AI URL prioritization failed: {e}", "error")
            return self._fallback_prioritization(urls)

    def _fallback_prioritization(self, urls: List[str]) -> List[Dict]:
        """Intelligent fallback prioritization when AI fails"""
        priority_keywords = {
            'HIGH': ['login', 'auth', 'admin', 'api', 'rest', 'graphql', 'oauth', 'token', 'password', 'secret', 'key'],
            'MEDIUM': ['user', 'account', 'profile', 'dashboard', 'upload', 'file', 'config', 'setting', 'database'],
            'LOW': ['static', 'css', 'js', 'img', 'image', 'icon', 'favicon', 'font']
        }

        prioritized = []
        for url in urls[:20]:
            url_lower = url.lower()
            priority = 'LOW'
            score = 3

            for high_kw in priority_keywords['HIGH']:
                if high_kw in url_lower:
                    priority = 'HIGH'
                    score = 9
                    break

            if priority == 'LOW':
                for medium_kw in priority_keywords['MEDIUM']:
                    if medium_kw in url_lower:
                        priority = 'MEDIUM'
                        score = 6
                        break

            prioritized.append({
                'url': url,
                'priority': priority,
                'score': score,
                'reason': f"Contains {priority.lower()} priority keywords",
                'confidence': 0.7
            })

        self._print_status(f"🔧 Fallback prioritized {len(prioritized)} URLs", "info")
        return prioritized

    async def analyze_endpoints(self, endpoints: List[str], context: str = "") -> List[Dict]:
        """AI analysis of discovered endpoints"""
        if not endpoints or not await self.ensure_ai_working():
            return []

        prompt = f"""Analyze these API endpoints for security testing priorities.
        {context}

        Endpoints:
        {chr(10).join(f"- {endpoint}" for endpoint in endpoints[:12])}
        """

        try:
            request = AIRequest(prompt=prompt, operation="analyze_endpoints", max_tokens=800)
            response = await self._make_ai_request(request)
            result = self._extract_json_from_response(response)
            return self._normalize_ai_response(result, "analyze_endpoints")

        except Exception as e:
            self._print_status(f"AI endpoint analysis failed: {e}", "error")
            return []

    async def validate_leaks(self, leaks: List[Dict]) -> List[Dict]:
        """AI validation of detected security leaks"""
        if not leaks or not await self.ensure_ai_working():
            return self.rank_findings(leaks)

        leaks_summary = []
        for leak in leaks[:8]:
            leaks_summary.append({
                'type': leak.get('type', 'unknown'),
                'value': leak.get('value', '')[:80],
                'context': leak.get('context', ''),
                'source': leak.get('source_url', 'unknown')
            })

        prompt = f"""Validate these potential security leaks:
        {json.dumps(leaks_summary, indent=2)}
        """

        try:
            request = AIRequest(prompt=prompt, operation="validate_leaks")
            response = await self._make_ai_request(request)
            result = self._extract_json_from_response(response)
            return self._normalize_ai_response(result, "validate_leaks")

        except Exception as e:
            self._print_status(f"AI leak validation failed: {e}", "error")
            return self.rank_findings(leaks)

    async def generate_report_insights(self, scan_results: Dict) -> Dict:
        """AI-powered security insights generation"""
        if not await self.ensure_ai_working():
            return self._get_fallback_insights()

        summary = {
            'domain': scan_results.get('scan_config', {}).get('domain', 'unknown'),
            'subdomains_found': len(scan_results.get('subdomains', [])),
            'urls_crawled': len(scan_results.get('urls', [])),
            'js_files_analyzed': len(scan_results.get('js_files', [])),
            'leaks_detected': len(scan_results.get('leaks', [])),
            'high_confidence_leaks': len(scan_results.get('high_confidence_leaks', [])),
            'critical_findings': len([l for l in scan_results.get('leaks', []) if l.get('severity') in ['CRITICAL', 'HIGH']])
        }

        prompt = f"""Analyze these security scan results:
        {json.dumps(summary, indent=2)}
        """

        try:
            request = AIRequest(prompt=prompt, operation="generate_insights")
            response = await self._make_ai_request(request)
            result = self._extract_json_from_response(response)
            return self._normalize_ai_response(result, "generate_insights")

        except Exception as e:
            self._print_status(f"AI insights generation failed: {e}", "error")
            return self._get_fallback_insights()

    def _get_fallback_insights(self) -> Dict:
        """Fallback insights when AI is unavailable"""
        return {
            'risk_assessment': 'UNKNOWN',
            'key_findings_summary': 'AI analysis unavailable - review findings manually',
            'testing_recommendations': ['Continue manual security testing', 'Verify all detected leaks'],
            'potential_impact': 'Unable to assess without AI analysis',
            'confidence': 0.0
        }

    # SCORING AND UTILITY METHODS

    def score(self, finding: Dict) -> Dict:
        """Comprehensive finding scoring algorithm"""
        base_scores = {
            'api_key': 90, 'password': 85, 'secret': 80, 'token': 75,
            'credential': 70, 'private_key': 95, 'aws_key': 90,
            'endpoint': 50, 'email': 40, 'url': 30
        }

        finding_type = finding.get('type', 'unknown').lower()
        base_score = base_scores.get(finding_type, 50)

        # Context-based scoring
        context = finding.get('context', '').lower()
        value = finding.get('value', '').lower()

        context_boosters = {
            'auth': 15, 'login': 12, 'admin': 15, 'api': 10,
            'config': 10, 'database': 12, 'key': 10, 'secret': 15
        }

        for keyword, boost in context_boosters.items():
            if keyword in context or keyword in value:
                base_score += boost

        # Normalize and cap score
        score = min(100, max(0, base_score))

        # Determine severity
        if score >= 80:
            severity = 'CRITICAL'
        elif score >= 60:
            severity = 'HIGH'
        elif score >= 40:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        # Calculate confidence
        confidence = min(1.0, score / 100)

        scored_finding = finding.copy()
        scored_finding.update({
            'score': score,
            'severity': severity,
            'confidence': round(confidence, 2)
        })

        return scored_finding

    def rank_findings(self, findings: List[Dict]) -> List[Dict]:
        """Rank findings by score and confidence"""
        return sorted(
            findings,
            key=lambda x: (x.get('score', 0), x.get('confidence', 0)),
            reverse=True
        )

    def filter_high_confidence(self, findings: List[Dict]) -> List[Dict]:
        """Filter for high-confidence, high-impact findings"""
        return [
            f for f in findings
            if f.get('confidence', 0) >= 0.7 and f.get('score', 0) >= 60
        ]

    # ANALYTICS AND REPORTING

    def get_ai_stats(self) -> Dict:
        """Get comprehensive AI usage statistics"""
        total_time = self.ai_stats['total_processing_time']
        total_requests = self.ai_stats['total_requests']

        stats = {
            "total_requests": self.ai_stats['total_requests'],
            "successful_requests": self.ai_stats['successful_requests'],
            "failed_requests": self.ai_stats['failed_requests'],
            "success_rate": round((self.ai_stats['successful_requests'] / total_requests * 100) if total_requests > 0 else 0, 1),
            "total_tokens_used": self.ai_stats['total_tokens_used'],
            "total_processing_time": f"{total_time:.2f}s",
            "average_response_time": f"{(total_time / total_requests) if total_requests > 0 else 0:.2f}s",
            "active_keys": len(self.active_keys),
            "exhausted_keys": self.ai_stats['keys_exhausted'],
            "models_used": list(self.ai_stats['models_used']),
            "key_performance": self.key_stats
        }
        return stats

    def print_ai_summary(self):
        """Print clean AI performance summary"""
        stats = self.get_ai_stats()

        self._print_status("🤖 AI Performance Summary:", "info")
        self._print_status(f"  Total Requests: {stats['total_requests']}", "info")
        self._print_status(f"  Successful: {stats['successful_requests']} ({stats['success_rate']}%)",
                    "success" if stats['success_rate'] > 80 else "warning")
        self._print_status(f"  Failed: {stats['failed_requests']}",
                    "warning" if stats['failed_requests'] > 0 else "info")
        self._print_status(f"  Tokens Used: {stats['total_tokens_used']:,}", "info")
        self._print_status(f"  Avg Response Time: {stats['average_response_time']}", "info")
        self._print_status(f"  Active Keys: {stats['active_keys']}", "info")
        self._print_status(f"  Models Used: {', '.join(stats['models_used'])}", "info")

    def _print_status(self, message: str, level: str = "info"):
        """Print clean status messages without technical details"""
        from rich.console import Console
        console = Console()

        # Filter out technical AI messages for clean interface
        technical_patterns = [
            "HTTP 404 for",
            "AI Request:",
            "AI Response:",
            "Key ...",
            "tokens,",
            "Raw AI response",
            "JSON parse error",
            "Raw content preview"
        ]
        
        if any(pattern in message for pattern in technical_patterns):
            return  # Skip technical messages for clean interface
    
        styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow", 
            "error": "red",
            "debug": "dim"
        }

        style = styles.get(level, "white")
        console.print(message, style=style)

# Export the class
__all__ = ['AIScoring']
