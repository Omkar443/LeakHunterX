import re
import math
import logging
from typing import List, Dict, Any

class LeakDetector:
    """
    Aggressive leak detector for keys, tokens, credentials and bucket URLs.
    Uses regex + entropy to detect high-value secrets.
    """

    # CORRECTED Regex rules - Fixed double backslashes
    PATTERNS = {
        # Cloud Keys
        "aws_access_key": r"AKIA[0-9A-Z]{16}",
        "aws_secret_key": r"(?i)aws[^'\"\n]{0,20}?['\"\s:=]([A-Za-z0-9/+=]{40})",

        # Google Cloud
        "google_api_key": r"AIza[0-9A-Za-z\-_]{35}",
        "google_oauth_token": r"ya29\.[0-9A-Za-z\-_]+",

        # API Keys
        "generic_api_key": r"(?i)(api[_-]?key|apikey|secret[_-]?key)['\"\s:=]+([A-Za-z0-9\-_.=]{20,})",
        "bearer_token": r"(?i)bearer[\s]+([A-Za-z0-9\-_.=]{50,})",

        # Authentication
        "jwt_token": r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*",
        "basic_auth": r"(?i)basic[\s]+([A-Za-z0-9+/=]{20,})",

        # Service-specific keys
        "stripe_key": r"(sk|pk)_(test|live)_[0-9a-zA-Z]{24}",
        "twilio_key": r"SK[0-9a-fA-F]{32}",
        "slack_token": r"xox[baprs]-[0-9a-zA-Z]{10,48}",
        "github_token": r"gh[pousr]_[A-Za-z0-9_]{36}",
        "mailgun_key": r"key-[0-9a-zA-Z]{32}",

        # Database connections
        "mongodb_uri": r"mongodb(\+srv)?://[^\s\"']+",
        "mysql_connection": r"mysql://[^\s\"']+",
        "postgres_connection": r"postgres(ql)?://[^\s\"']+",
        "redis_connection": r"redis://[^\s\"']+",

        # Cloud storage
        "s3_bucket": r"s3://[a-z0-9\-\.]+",
        "s3_url": r"https?://[a-z0-9\-\.]+\.s3\.(?:[a-z0-9\-\.]+)?amazonaws\.com",
        "google_storage": r"gs://[a-z0-9\-\.]+",
        "azure_storage": r"https?://[a-z0-9]+\.blob\.core\.windows\.net",

        # Email/password combos
        "email_password": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[:|,|\s]+[^\s\"']+",

        # Internal endpoints
        "admin_endpoint": r"(?i)(admin|internal|private|staging|dev|test)[^\"'\s]{0,30}",
    }

    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive
        if aggressive:
            self._enable_aggressive_patterns()

    def _enable_aggressive_patterns(self):
        """Enable more aggressive detection patterns"""
        aggressive_patterns = {
            "private_ip": r"(?i)(192\.168|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))\.[0-9]{1,3}\.[0-9]{1,3}",
            "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
            "ssh_private_key": r"-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
        }
        self.PATTERNS.update(aggressive_patterns)

    def entropy(self, text: str) -> float:
        """Calculate Shannon entropy for a string"""
        if not text:
            return 0.0

        text = str(text)
        entropy = 0.0
        text_length = len(text)

        for char in set(text):
            p_x = float(text.count(char)) / text_length
            if p_x > 0:
                entropy += - p_x * math.log2(p_x)

        return entropy

    def compute_severity(self, leak_type: str, entropy: float) -> str:
        """Compute severity based on leak type and entropy"""
        high_severity_types = {
            'aws_access_key', 'aws_secret_key', 'stripe_key', 'twilio_key',
            'github_token', 'ssh_private_key', 'credit_card'
        }

        medium_severity_types = {
            'google_api_key', 'slack_token', 'mailgun_key', 'jwt_token',
            'bearer_token', 'mongodb_uri', 'mysql_connection'
        }

        if leak_type in high_severity_types:
            return "HIGH"
        elif leak_type in medium_severity_types:
            return "MEDIUM" if entropy > 3.0 else "LOW"
        elif entropy > 4.0:
            return "HIGH"
        elif entropy > 3.0:
            return "MEDIUM"
        else:
            return "LOW"

    def check_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Check content for leaks - FIXED METHOD NAME to match main script
        Main script expects: potential_leaks = leak_detector.check_content(endpoints)
        """
        findings = []

        if not content or not isinstance(content, str):
            return findings

        for name, regex in self.PATTERNS.items():
            try:
                matches = re.findall(regex, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Handle tuple matches (regex groups)
                    if isinstance(match, tuple):
                        # Take the last non-empty group
                        leak_value = next((m for m in match[::-1] if m), "")
                    else:
                        leak_value = match

                    if leak_value and len(leak_value) > 5:  # Basic length filter
                        # Calculate entropy
                        ent = self.entropy(leak_value)
                        severity = self.compute_severity(name, ent)
                        suspicious = ent > 3.5

                        findings.append({
                            "type": name,
                            "value": leak_value[:100],  # Truncate for safety
                            "severity": severity,
                            "entropy": round(ent, 2),
                            "suspicious": suspicious,
                            "confidence": min(0.3 + (ent * 0.2), 0.95)  # Dynamic confidence
                        })

            except Exception as e:
                logging.debug(f"Regex error for {name}: {e}")
                continue

        # Remove duplicates based on value and type
        unique_findings = []
        seen = set()

        for finding in findings:
            key = (finding["type"], finding["value"])
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)

        return unique_findings

    def group_by_severity(self, findings: List[Dict]) -> Dict[str, List[Dict]]:
        """Group findings by severity level"""
        grouped = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}

        for finding in findings:
            severity = finding.get("severity", "LOW")
            if severity in grouped:
                grouped[severity].append(finding)
            else:
                grouped["LOW"].append(finding)

        return grouped

    def get_stats(self, findings: List[Dict]) -> Dict[str, int]:
        """Get statistics about findings"""
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for finding in findings:
            severity = finding.get("severity", "LOW")
            if severity in severity_counts:
                severity_counts[severity] += 1

        return {
            "total_findings": len(findings),
            "by_severity": severity_counts,
            "suspicious_count": sum(1 for f in findings if f.get("suspicious", False))
        }
