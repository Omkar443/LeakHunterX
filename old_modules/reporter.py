#!/usr/bin/env python3
"""
LeakHunterX Reporter Module
Enhanced with Detailed Contextual Reporting and Actionable Insights
"""

import os
import json
import html
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class Reporter:
    """
    Generates detailed JSON + HTML reports with full context.
    Shows exact locations, URLs, and actionable insights for each finding.
    """

    def __init__(self, domain: str, output_dir: str = "results"):
        self.domain = domain
        self.output_dir = os.path.join(output_dir, domain)
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, results: Dict[str, Any]):
        """
        Main save method called by the main script
        Enhanced to provide detailed contextual information
        """
        try:
            print(f"💾 Saving detailed reports for {self.domain}...")

            # Save the complete results as JSON with enhanced context
            json_path = self.save_json(results)

            # Generate detailed HTML report with full context
            leaks = results.get("leaks", [])
            if leaks:
                html_path = self.save_detailed_html(leaks, results)
                print(f"📄 Detailed HTML report: {html_path}")

            # Save additional data files
            self.save_additional_files(results)

            print(f"✅ Detailed reports saved to: {self.output_dir}")
            return True

        except Exception as e:
            print(f"❌ Error saving reports: {e}")
            return False

    def save_json(self, results: Dict[str, Any]) -> str:
        """Save complete results as JSON with enhanced context"""
        file_path = os.path.join(self.output_dir, "scan_results.json")

        # Enhanced data structure with full context
        report_data = {
            "domain": self.domain,
            "scan_date": datetime.now().isoformat(),
            "summary": {
                "subdomains_found": len(results.get("subdomains", [])),
                "urls_crawled": len(results.get("urls", [])),
                "js_files_analyzed": len(results.get("js_files", [])),
                "leaks_detected": len(results.get("leaks", [])),
                "critical_leaks": sum(1 for l in results.get("leaks", []) if l.get('severity') == 'CRITICAL'),
                "high_confidence_leaks": len(results.get("high_confidence_leaks", []))
            },
            "subdomains": results.get("subdomains", []),
            "urls": results.get("urls", []),
            "js_files": results.get("js_files", []),
            "leaks": self._enhance_leaks_context(results.get("leaks", [])),
            "high_confidence_leaks": results.get("high_confidence_leaks", []),
            "ai_prioritized_urls": results.get("ai_prioritized_urls", []),
            "scan_config": results.get("scan_config", {})
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return file_path

    def _enhance_leaks_context(self, leaks: List[Dict]) -> List[Dict]:
        """Add contextual information to leaks for better reporting"""
        enhanced_leaks = []
        for leak in leaks:
            enhanced_leak = leak.copy()
            
            # Add contextual information if missing
            if 'source_url' not in enhanced_leak:
                enhanced_leak['source_url'] = 'Unknown'
                
            if 'context' not in enhanced_leak:
                enhanced_leak['context'] = 'Discovered during automated scanning'
                
            # Add actionable insights based on leak type
            enhanced_leak['actionable_insights'] = self._get_actionable_insights(leak)
            
            enhanced_leaks.append(enhanced_leak)
            
        return enhanced_leaks

    def _get_actionable_insights(self, leak: Dict) -> List[str]:
        """Generate actionable insights based on leak type and context"""
        leak_type = leak.get('type', '').lower()
        insights = []
        
        if 'api_key' in leak_type or 'secret' in leak_type or 'token' in leak_type:
            insights.extend([
                "🔑 This appears to be an API key or secret token",
                "🚨 Immediate rotation of this credential is recommended",
                "📋 Check if this key has access to sensitive data or services",
                "🔍 Review access logs for any unauthorized usage"
            ])
            
        elif 'admin' in leak_type or 'endpoint' in leak_type:
            insights.extend([
                "🔐 This appears to be an administrative endpoint",
                "🛡️ Verify if proper authentication is required",
                "📊 Check access logs for unauthorized access attempts",
                "⚙️ Consider implementing additional security controls"
            ])
            
        elif 'password' in leak_type:
            insights.extend([
                "🔑 This appears to be a password or credential",
                "🚨 Immediate password reset is required",
                "📋 Check if this password is used elsewhere",
                "🔍 Review system logs for compromise indicators"
            ])
            
        else:
            insights.extend([
                "🔍 Review this finding manually for context",
                "📋 Determine the potential impact if exploited",
                "🛡️ Consider implementing additional monitoring",
                "📊 Document this finding for future reference"
            ])
            
        return insights

    def save_detailed_html(self, findings: List[Dict], full_results: Dict[str, Any]) -> str:
        """Save detailed HTML report with full context"""
        file_path = os.path.join(self.output_dir, "detailed_leaks_report.html")

        # Sort findings by severity (critical first)
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_findings = sorted(findings, key=lambda x: severity_order.get(x.get("severity", "LOW"), 4))

        html_content = self._generate_detailed_html_content(sorted_findings, full_results)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return file_path

    def _generate_detailed_html_content(self, findings: List[Dict], full_results: Dict[str, Any]) -> str:
        """Generate detailed HTML content with full context"""
        
        # Calculate statistics
        critical_count = sum(1 for f in findings if f.get('severity') == 'CRITICAL')
        high_count = sum(1 for f in findings if f.get('severity') == 'HIGH')
        validated_count = sum(1 for f in findings if f.get('validated') == 'format_valid')
        
        # Group findings by source for better organization
        findings_by_source = {}
        for finding in findings:
            source = finding.get('source_url', 'Unknown Source')
            if source not in findings_by_source:
                findings_by_source[source] = []
            findings_by_source[source].append(finding)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeakHunterX Detailed Report — {self.domain}</title>
    <style>
        body {{
            background-color: #1a1a1a;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #2d2d2d;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #444;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #ff4444;
            margin: 0;
            font-size: 2.5em;
        }}
        h2 {{
            color: #ff6b6b;
            border-bottom: 1px solid #444;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        h3 {{
            color: #ffa726;
            margin-top: 25px;
        }}
        .subtitle {{
            color: #888;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #3a3a3a;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            background: #444;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #ff4444;
        }}
        .stat-label {{
            color: #aaa;
            font-size: 0.9em;
        }}
        .finding-card {{
            background: #363636;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            transition: all 0.3s ease;
        }}
        .finding-card:hover {{
            background: #404040;
            border-color: #ff4444;
        }}
        .finding-header {{
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #444;
        }}
        .finding-type {{
            font-weight: bold;
            font-size: 1.1em;
            color: #ffa726;
        }}
        .finding-severity {{
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .critical {{ background: #ff4444; color: white; }}
        .high {{ background: #ff6b35; color: white; }}
        .medium {{ background: #ffa726; color: black; }}
        .low {{ background: #66bb6a; color: black; }}
        .finding-details {{
            margin: 10px 0;
        }}
        .detail-row {{
            display: flex;
            margin: 8px 0;
        }}
        .detail-label {{
            font-weight: bold;
            color: #aaa;
            min-width: 120px;
        }}
        .detail-value {{
            flex: 1;
            word-break: break-all;
        }}
        .source-url {{
            color: #4fc3f7;
            text-decoration: none;
        }}
        .source-url:hover {{
            text-decoration: underline;
        }}
        .insights {{
            background: #2a2a2a;
            border-left: 4px solid #ffa726;
            padding: 15px;
            margin-top: 15px;
            border-radius: 0 8px 8px 0;
        }}
        .insight-item {{
            margin: 8px 0;
            padding-left: 10px;
        }}
        .no-leaks {{
            text-align: center;
            color: #66bb6a;
            font-size: 1.2em;
            padding: 40px;
            background: #2a2a2a;
            border-radius: 8px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #444;
            padding-top: 20px;
        }}
        .source-section {{
            margin: 25px 0;
            padding: 15px;
            background: #363636;
            border-radius: 8px;
        }}
        .source-header {{
            color: #4fc3f7;
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        .copy-btn {{
            background: #444;
            border: none;
            color: #e0e0e0;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        .copy-btn:hover {{
            background: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 LeakHunterX Detailed Report</h1>
            <div class="subtitle">Comprehensive Security Assessment for {self.domain}</div>
            <div class="subtitle">Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(findings)}</div>
                <div class="stat-label">Total Findings</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{critical_count}</div>
                <div class="stat-label">Critical Severity</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{high_count}</div>
                <div class="stat-label">High Severity</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(full_results.get('high_confidence_leaks', []))}</div>
                <div class="stat-label">High Confidence</div>
            </div>
        </div>

        {self._generate_executive_summary(findings)}
        {self._generate_detailed_findings_by_source(findings_by_source)}
        {self._generate_scan_context(full_results)}

        <div class="footer">
            🔒 Generated by LeakHunterX - Advanced Security Reconnaissance Tool<br>
            📧 For security concerns, contact your security team immediately
        </div>
    </div>

    <script>
        function copyToClipboard(text) {{
            navigator.clipboard.writeText(text).then(function() {{
                alert('Copied to clipboard: ' + text);
            }}, function(err) {{
                console.error('Could not copy text: ', err);
            }});
        }}
    </script>
</body>
</html>"""

    def _generate_executive_summary(self, findings: List[Dict]) -> str:
        """Generate executive summary section"""
        if not findings:
            return '<div class="no-leaks">🎉 No security findings detected during this scan!</div>'

        critical_count = sum(1 for f in findings if f.get('severity') == 'CRITICAL')
        high_count = sum(1 for f in findings if f.get('severity') == 'HIGH')
        
        summary_html = """
        <div class="source-section">
            <h2>📊 Executive Summary</h2>
            <div class="finding-details">
        """

        if critical_count > 0:
            summary_html += f"""
                <div class="detail-row">
                    <div class="detail-label">🚨 Critical Issues:</div>
                    <div class="detail-value"><span class="critical">{critical_count} findings require immediate attention</span></div>
                </div>
            """
        
        if high_count > 0:
            summary_html += f"""
                <div class="detail-row">
                    <div class="detail-label">⚠️ High Severity:</div>
                    <div class="detail-value"><span class="high">{high_count} findings need prompt review</span></div>
                </div>
            """

        summary_html += f"""
                <div class="detail-row">
                    <div class="detail-label">📋 Total Findings:</div>
                    <div class="detail-value">{len(findings)} security items discovered across all scanned assets</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🎯 Next Steps:</div>
                    <div class="detail-value">Review each finding below for detailed context and remediation steps</div>
                </div>
            </div>
        </div>
        """

        return summary_html

    def _generate_detailed_findings_by_source(self, findings_by_source: Dict) -> str:
        """Generate detailed findings organized by source"""
        if not findings_by_source:
            return '<div class="no-leaks">🎉 No security findings to report!</div>'

        findings_html = '<h2>🔍 Detailed Security Findings</h2>'

        for source, source_findings in findings_by_source.items():
            findings_html += f"""
            <div class="source-section">
                <div class="source-header">📍 Source: {self._format_source_url(source)}</div>
            """

            for finding in source_findings:
                findings_html += self._generate_finding_card(finding)

            findings_html += "</div>"

        return findings_html

    def _format_source_url(self, url: str) -> str:
        """Format source URL for display"""
        if url.startswith(('http://', 'https://')):
            return f'<a href="{url}" class="source-url" target="_blank">{url}</a>'
        elif url == 'Unknown Source':
            return '<em>Unknown Source (discovered during scanning)</em>'
        else:
            return html.escape(url)

    def _generate_finding_card(self, finding: Dict) -> str:
        """Generate a detailed finding card"""
        severity = finding.get('severity', 'LOW')
        severity_class = severity.lower()
        
        # Get actionable insights
        insights = finding.get('actionable_insights', self._get_actionable_insights(finding))
        
        finding_html = f"""
        <div class="finding-card">
            <div class="finding-header">
                <div class="finding-type">🔎 {finding.get('type', 'Unknown Finding')}</div>
                <div class="finding-severity {severity_class}">{severity}</div>
            </div>
            
            <div class="finding-details">
                <div class="detail-row">
                    <div class="detail-label">📝 Description:</div>
                    <div class="detail-value">{finding.get('reason', finding.get('context', 'Security finding discovered during scan'))}</div>
                </div>
                
                <div class="detail-row">
                    <div class="detail-label">🎯 Value Found:</div>
                    <div class="detail-value">
                        <code>{html.escape(str(finding.get('value', 'No value')))}</code>
                        <button class="copy-btn" onclick="copyToClipboard('{html.escape(str(finding.get('value', '')))}')">Copy</button>
                    </div>
                </div>
                
                <div class="detail-row">
                    <div class="detail-label">📊 Confidence:</div>
                    <div class="detail-value">{finding.get('confidence', 0)} / 1.0</div>
                </div>
                
                <div class="detail-row">
                    <div class="detail-label">🔗 Source URL:</div>
                    <div class="detail-value">{self._format_source_url(finding.get('source_url', 'Unknown'))}</div>
                </div>
                
                {self._generate_additional_context(finding)}
            </div>
            
            <div class="insights">
                <strong>💡 Actionable Insights:</strong>
                {''.join(f'<div class="insight-item">{insight}</div>' for insight in insights)}
            </div>
        </div>
        """
        
        return finding_html

    def _generate_additional_context(self, finding: Dict) -> str:
        """Generate additional context for the finding"""
        context_html = ""
        
        if finding.get('context'):
            context_html += f"""
                <div class="detail-row">
                    <div class="detail-label">📋 Context:</div>
                    <div class="detail-value">{html.escape(str(finding.get('context')))}</div>
                </div>
            """
            
        if finding.get('exploitability', 0) > 0:
            context_html += f"""
                <div class="detail-row">
                    <div class="detail-label">⚡ Exploitability:</div>
                    <div class="detail-value">{finding.get('exploitability', 0)} / 10</div>
                </div>
            """
            
        return context_html

    def _generate_scan_context(self, full_results: Dict[str, Any]) -> str:
        """Generate scan context and statistics"""
        return f"""
        <div class="source-section">
            <h2>📈 Scan Context & Statistics</h2>
            <div class="finding-details">
                <div class="detail-row">
                    <div class="detail-label">🌐 Subdomains Found:</div>
                    <div class="detail-value">{len(full_results.get('subdomains', []))}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🔗 URLs Crawled:</div>
                    <div class="detail-value">{len(full_results.get('urls', []))}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">📜 JS Files Analyzed:</div>
                    <div class="detail-value">{len(full_results.get('js_files', []))}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🤖 AI-Prioritized URLs:</div>
                    <div class="detail-value">{len(full_results.get('ai_prioritized_urls', []))}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">⚙️ Scan Mode:</div>
                    <div class="detail-value">{'Aggressive' if full_results.get('scan_config', {}).get('aggressive') else 'Standard'}</div>
                </div>
            </div>
        </div>
        """

    def save_additional_files(self, results: Dict[str, Any]):
        """Save additional data files for analysis"""
        # Save subdomains list
        if results.get("subdomains"):
            subdomains_path = os.path.join(self.output_dir, "subdomains.txt")
            with open(subdomains_path, "w") as f:
                for subdomain in results["subdomains"]:
                    f.write(f"{subdomain}\n")

        # Save URLs list
        if results.get("urls"):
            urls_path = os.path.join(self.output_dir, "urls.txt")
            with open(urls_path, "w") as f:
                for url in results["urls"]:
                    f.write(f"{url}\n")

        # Save JS files list
        if results.get("js_files"):
            js_files_path = os.path.join(self.output_dir, "js_files.txt")
            with open(js_files_path, "w") as f:
                for js_file in results["js_files"]:
                    f.write(f"{js_file}\n")

        # Save high-confidence leaks separately
        if results.get("high_confidence_leaks"):
            high_conf_path = os.path.join(self.output_dir, "high_confidence_leaks.json")
            with open(high_conf_path, "w", encoding="utf-8") as f:
                json.dump(results["high_confidence_leaks"], f, indent=2, ensure_ascii=False)

    def get_report_paths(self) -> Dict[str, str]:
        """Get paths to all generated reports"""
        return {
            "json_report": os.path.join(self.output_dir, "scan_results.json"),
            "html_report": os.path.join(self.output_dir, "detailed_leaks_report.html"),
            "subdomains": os.path.join(self.output_dir, "subdomains.txt"),
            "urls": os.path.join(self.output_dir, "urls.txt"),
            "js_files": os.path.join(self.output_dir, "js_files.txt"),
            "high_confidence_leaks": os.path.join(self.output_dir, "high_confidence_leaks.json")
        }
