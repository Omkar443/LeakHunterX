#!/usr/bin/env python3
"""
LeakHunterX Reporter Module - Enhanced with AI Insights & Batch Processing
Professional reporting with business context and actionable remediation
"""

import os
import json
import html
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class Reporter:
    """
    Enhanced Reporter with AI-Powered Insights
    - Professional business-focused reporting
    - AI-generated risk assessment and recommendations
    - Streaming-friendly output formats
    - Executive summaries for management
    """

    def __init__(self, domain: str, output_dir: str = "results"):
        self.domain = domain
        self.output_dir = os.path.join(output_dir, domain)
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, results: Dict[str, Any]):
        """
        Enhanced save method with AI insights and professional reporting
        """
        try:
            print(f"💾 Generating professional reports for {self.domain}...")

            # Save enhanced JSON with AI insights
            json_path = self.save_enhanced_json(results)

            # Generate comprehensive HTML report
            leaks = results.get("leaks", [])
            ai_insights = results.get("ai_insights", {})
            
            if leaks or ai_insights:
                html_path = self.save_professional_html_report(results)
                print(f"📄 Professional HTML report: {html_path}")

            # Save executive summary for management
            exec_summary_path = self.save_executive_summary(results)
            print(f"📋 Executive summary: {exec_summary_path}")

            # Save additional data files
            self.save_additional_files(results)

            print(f"✅ Professional reports saved to: {self.output_dir}")
            return True

        except Exception as e:
            print(f"❌ Error saving reports: {e}")
            return False

    def save_enhanced_json(self, results: Dict[str, Any]) -> str:
        """Save enhanced JSON with AI insights and business context"""
        file_path = os.path.join(self.output_dir, "scan_results.json")

        # Enhanced data structure with AI insights
        report_data = {
            "metadata": {
                "domain": self.domain,
                "scan_date": datetime.now().isoformat(),
                "report_version": "2.0",
                "tool": "LeakHunterX Enhanced"
            },
            "executive_summary": self._generate_executive_summary_data(results),
            "risk_assessment": self._generate_risk_assessment(results),
            "technical_findings": {
                "summary": {
                    "total_findings": len(results.get("leaks", [])),
                    "critical_severity": sum(1 for l in results.get("leaks", []) if l.get('severity') == 'CRITICAL'),
                    "high_severity": sum(1 for l in results.get("leaks", []) if l.get('severity') == 'HIGH'),
                    "ai_validated": len(results.get("high_confidence_leaks", [])),
                    "subdomains_discovered": len(results.get("subdomains", [])),
                    "urls_crawled": len(results.get("urls", [])),
                    "js_files_analyzed": len(results.get("js_files", []))
                },
                "findings": self._enhance_findings_with_business_context(results.get("leaks", [])),
                "high_confidence_findings": results.get("high_confidence_leaks", [])
            },
            "attack_surface_analysis": {
                "subdomains": results.get("subdomains", []),
                "critical_endpoints": self._extract_critical_endpoints(results),
                "admin_interfaces": results.get("admin_panels", []),
                "api_endpoints": self._extract_api_endpoints(results)
            },
            "ai_insights": results.get("ai_insights", {}),
            "remediation_roadmap": self._generate_remediation_roadmap(results),
            "scan_configuration": results.get("scan_config", {})
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return file_path

    def _generate_executive_summary_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary data for management"""
        leaks = results.get("leaks", [])
        ai_insights = results.get("ai_insights", {})
        
        critical_count = sum(1 for l in leaks if l.get('severity') == 'CRITICAL')
        high_count = sum(1 for l in leaks if l.get('severity') == 'HIGH')
        validated_count = len(results.get("high_confidence_leaks", []))

        return {
            "overall_risk_level": ai_insights.get('risk_assessment', 'UNKNOWN'),
            "key_findings_count": len(leaks),
            "critical_issues": critical_count,
            "high_priority_issues": high_count,
            "validated_findings": validated_count,
            "business_impact": ai_insights.get('potential_impact', 'Assessment pending'),
            "recommended_actions": ai_insights.get('testing_recommendations', ['Review findings manually']),
            "next_steps": [
                "Immediate review of critical findings",
                "Coordinate with security team for remediation",
                "Schedule follow-up assessment"
            ]
        }

    def _generate_risk_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        leaks = results.get("leaks", [])
        
        # Calculate risk scores
        critical_score = sum(10 for l in leaks if l.get('severity') == 'CRITICAL')
        high_score = sum(5 for l in leaks if l.get('severity') == 'HIGH')
        medium_score = sum(2 for l in leaks if l.get('severity') == 'MEDIUM')
        
        total_risk_score = critical_score + high_score + medium_score
        
        # Determine risk level
        if total_risk_score >= 20:
            risk_level = "CRITICAL"
        elif total_risk_score >= 10:
            risk_level = "HIGH"
        elif total_risk_score >= 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_score": total_risk_score,
            "breakdown": {
                "critical_contributions": critical_score,
                "high_contributions": high_score,
                "medium_contributions": medium_score
            },
            "factors_considered": [
                "Number and severity of security findings",
                "Sensitivity of exposed data",
                "Potential business impact",
                "Attack surface complexity"
            ]
        }

    def _enhance_findings_with_business_context(self, findings: List[Dict]) -> List[Dict]:
        """Add business context to findings"""
        enhanced_findings = []
        
        for finding in findings:
            enhanced = finding.copy()
            
            # Add business impact assessment
            enhanced['business_impact'] = self._assess_business_impact(finding)
            
            # Add remediation priority
            enhanced['remediation_priority'] = self._determine_remediation_priority(finding)
            
            # Add exploitability assessment
            enhanced['exploitability'] = self._assess_exploitability(finding)
            
            # Add compliance implications
            enhanced['compliance_implications'] = self._identify_compliance_issues(finding)
            
            enhanced_findings.append(enhanced)
            
        return enhanced_findings

    def _assess_business_impact(self, finding: Dict) -> str:
        """Assess business impact of a finding"""
        severity = finding.get('severity', 'LOW')
        finding_type = finding.get('type', '').lower()
        
        if severity == 'CRITICAL':
            if any(keyword in finding_type for keyword in ['aws', 'google', 'api_key', 'secret', 'private_key']):
                return "HIGH - Potential financial loss and data breach"
            return "HIGH - Critical system compromise possible"
        
        elif severity == 'HIGH':
            if 'password' in finding_type or 'credential' in finding_type:
                return "HIGH - Account takeover and data access"
            return "MEDIUM - Limited data exposure possible"
        
        else:
            return "LOW - Minimal direct business impact"

    def _determine_remediation_priority(self, finding: Dict) -> str:
        """Determine remediation priority"""
        severity = finding.get('severity', 'LOW')
        
        if severity == 'CRITICAL':
            return "IMMEDIATE - Address within 24 hours"
        elif severity == 'HIGH':
            return "URGENT - Address within 72 hours"
        elif severity == 'MEDIUM':
            return "HIGH - Address within 1 week"
        else:
            return "MEDIUM - Address in next maintenance window"

    def _assess_exploitability(self, finding: Dict) -> Dict[str, Any]:
        """Assess exploitability of a finding"""
        severity = finding.get('severity', 'LOW')
        finding_type = finding.get('type', '').lower()
        
        base_score = {
            'CRITICAL': 9,
            'HIGH': 7,
            'MEDIUM': 5,
            'LOW': 3
        }.get(severity, 3)
        
        # Adjust based on finding type
        if any(keyword in finding_type for keyword in ['api_key', 'secret', 'token']):
            base_score += 2
        elif 'password' in finding_type:
            base_score += 1
            
        return {
            "score": min(10, base_score),
            "level": "HIGH" if base_score >= 8 else "MEDIUM" if base_score >= 5 else "LOW",
            "factors": ["Automated tools can exploit", "Publicly accessible"] if base_score >= 7 else ["Manual exploitation required"]
        }

    def _identify_compliance_issues(self, finding: Dict) -> List[str]:
        """Identify compliance implications"""
        finding_type = finding.get('type', '').lower()
        implications = []
        
        if any(keyword in finding_type for keyword in ['api_key', 'secret', 'password', 'credential']):
            implications.extend(["SOC2 - Security controls", "GDPR - Data protection"])
        
        if 'personal' in finding_type or 'user' in finding_type:
            implications.append("GDPR - Personal data exposure")
            
        if 'financial' in finding_type or 'payment' in finding_type:
            implications.extend(["PCI-DSS - Payment data security", "SOX - Financial controls"])
            
        return implications if implications else ["General security best practices"]

    def _extract_critical_endpoints(self, results: Dict[str, Any]) -> List[str]:
        """Extract critical endpoints from findings"""
        critical_endpoints = set()
        
        for finding in results.get("leaks", []):
            if finding.get('severity') in ['CRITICAL', 'HIGH']:
                source_url = finding.get('source_url')
                if source_url and source_url != 'Unknown':
                    critical_endpoints.add(source_url)
                    
        return list(critical_endpoints)[:20]  # Limit to top 20

    def _extract_api_endpoints(self, results: Dict[str, Any]) -> List[str]:
        """Extract API endpoints from findings"""
        api_endpoints = set()
        
        for finding in results.get("leaks", []):
            finding_type = finding.get('type', '').lower()
            if 'api' in finding_type or 'endpoint' in finding_type:
                source_url = finding.get('source_url')
                if source_url and source_url != 'Unknown':
                    api_endpoints.add(source_url)
                    
        return list(api_endpoints)

    def _generate_remediation_roadmap(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate remediation roadmap"""
        leaks = results.get("leaks", [])
        
        critical_findings = [l for l in leaks if l.get('severity') == 'CRITICAL']
        high_findings = [l for l in leaks if l.get('severity') == 'HIGH']
        
        return {
            "immediate_actions": {
                "timeframe": "24-48 hours",
                "tasks": [
                    "Rotate all exposed API keys and credentials",
                    "Review and secure critical endpoints",
                    "Implement additional monitoring",
                    "Notify relevant stakeholders"
                ]
            },
            "short_term_actions": {
                "timeframe": "1 week",
                "tasks": [
                    "Complete security review of all findings",
                    "Implement recommended security controls",
                    "Update incident response procedures",
                    "Conduct team security awareness briefing"
                ]
            },
            "long_term_improvements": {
                "timeframe": "1 month",
                "tasks": [
                    "Implement automated security scanning",
                    "Enhance security training programs",
                    "Review and update security policies",
                    "Schedule follow-up penetration testing"
                ]
            }
        }

    def save_professional_html_report(self, results: Dict[str, Any]) -> str:
        """Save professional HTML report with AI insights"""
        file_path = os.path.join(self.output_dir, "professional_security_report.html")
        
        html_content = self._generate_professional_html_content(results)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return file_path

    def _generate_professional_html_content(self, results: Dict[str, Any]) -> str:
        """Generate professional HTML content"""
        leaks = results.get("leaks", [])
        ai_insights = results.get("ai_insights", {})
        exec_summary = self._generate_executive_summary_data(results)
        risk_assessment = self._generate_risk_assessment(results)
        
        # Sort findings by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_findings = sorted(leaks, key=lambda x: severity_order.get(x.get("severity", "LOW"), 4))
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Security Report — {self.domain}</title>
    <style>
        /* Enhanced professional styling */
        body {{
            background: linear-gradient(135deg, #1a2a3a, #0d1b2a);
            color: #e0e0e0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 40px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 30px;
            margin-bottom: 40px;
        }}
        h1 {{
            color: #ff6b6b;
            margin: 0;
            font-size: 2.8em;
            font-weight: 300;
            letter-spacing: -0.5px;
        }}
        .subtitle {{
            color: #888;
            font-size: 1.2em;
            margin-top: 10px;
            font-weight: 300;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: bold;
            margin-top: 15px;
            font-size: 1.1em;
        }}
        .risk-critical {{ background: #ff4444; color: white; }}
        .risk-high {{ background: #ff6b35; color: white; }}
        .risk-medium {{ background: #ffa726; color: black; }}
        .risk-low {{ background: #66bb6a; color: black; }}
        
        .executive-summary {{
            background: rgba(255, 255, 255, 0.08);
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
            border-left: 4px solid #4fc3f7;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
        }}
        .metric-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #4fc3f7;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #aaa;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .finding-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
            transition: all 0.3s ease;
        }}
        .finding-card:hover {{
            border-color: #ff6b6b;
            background: rgba(255, 255, 255, 0.08);
        }}
        
        .ai-insights {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
            color: white;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Professional Security Assessment</h1>
            <div class="subtitle">Comprehensive Security Report for {self.domain}</div>
            <div class="subtitle">Generated on {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</div>
            <div class="risk-badge risk-{risk_assessment['risk_level'].lower()}">
                Overall Risk: {risk_assessment['risk_level']}
            </div>
        </div>

        {self._generate_executive_summary_html(exec_summary, risk_assessment)}
        {self._generate_ai_insights_section(ai_insights)}
        {self._generate_metrics_section(results)}
        {self._generate_findings_section(sorted_findings)}
        {self._generate_remediation_section(results)}

        <div class="footer">
            <strong>LeakHunterX Professional Edition</strong><br>
            Advanced Security Reconnaissance & AI-Powered Analysis<br>
            For security emergencies, contact your security team immediately
        </div>
    </div>
</body>
</html>"""

    def _generate_executive_summary_html(self, exec_summary: Dict, risk_assessment: Dict) -> str:
        """Generate executive summary HTML"""
        return f"""
        <div class="executive-summary">
            <h2>📊 Executive Summary</h2>
            <p><strong>Overall Risk Level:</strong> <span class="risk-badge risk-{risk_assessment['risk_level'].lower()}">{risk_assessment['risk_level']}</span></p>
            <p><strong>Business Impact:</strong> {exec_summary['business_impact']}</p>
            <p><strong>Key Findings:</strong> {exec_summary['key_findings_count']} security issues identified</p>
            <p><strong>Critical Issues:</strong> {exec_summary['critical_issues']} require immediate attention</p>
            
            <h3>🎯 Recommended Immediate Actions:</h3>
            <ul>
                {"".join(f"<li>{action}</li>" for action in exec_summary['next_steps'])}
            </ul>
        </div>
        """

    def _generate_ai_insights_section(self, ai_insights: Dict) -> str:
        """Generate AI insights section"""
        if not ai_insights:
            return ""
            
        return f"""
        <div class="ai-insights">
            <h2>🤖 AI-Powered Security Insights</h2>
            <p><strong>Risk Assessment:</strong> {ai_insights.get('risk_assessment', 'N/A')}</p>
            <p><strong>Key Findings:</strong> {ai_insights.get('key_findings_summary', 'N/A')}</p>
            
            <h3>🔧 Recommended Testing Approaches:</h3>
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in ai_insights.get('testing_recommendations', []))}
            </ul>
        </div>
        """

    def _generate_metrics_section(self, results: Dict) -> str:
        """Generate metrics section"""
        leaks = results.get("leaks", [])
        critical_count = sum(1 for l in leaks if l.get('severity') == 'CRITICAL')
        high_count = sum(1 for l in leaks if l.get('severity') == 'HIGH')
        
        return f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-number">{len(leaks)}</div>
                <div class="metric-label">Total Findings</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{critical_count}</div>
                <div class="metric-label">Critical Severity</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{high_count}</div>
                <div class="metric-label">High Severity</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{len(results.get('subdomains', []))}</div>
                <div class="metric-label">Subdomains</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{len(results.get('js_files', []))}</div>
                <div class="metric-label">JS Files Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{len(results.get('high_confidence_leaks', []))}</div>
                <div class="metric-label">AI-Validated</div>
            </div>
        </div>
        """

    def _generate_findings_section(self, findings: List[Dict]) -> str:
        """Generate findings section"""
        if not findings:
            return '<div class="executive-summary"><h2>🎉 No Security Findings</h2><p>No security issues were detected during this assessment.</p></div>'
            
        findings_html = '<h2>🔍 Detailed Security Findings</h2>'
        
        for finding in findings:
            findings_html += self._generate_finding_card_html(finding)
            
        return findings_html

    def _generate_finding_card_html(self, finding: Dict) -> str:
        """Generate individual finding card HTML"""
        severity = finding.get('severity', 'LOW')
        business_impact = finding.get('business_impact', 'N/A')
        remediation_priority = finding.get('remediation_priority', 'N/A')
        
        return f"""
        <div class="finding-card">
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #ffa726;">{finding.get('type', 'Unknown Finding')}</h3>
                <span class="risk-badge risk-{severity.lower()}">{severity}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                <div>
                    <strong>🔍 Value Found:</strong><br>
                    <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px; display: block; margin-top: 5px;">
                        {html.escape(str(finding.get('value', 'No value')))}
                    </code>
                </div>
                <div>
                    <strong>📊 Confidence:</strong> {finding.get('confidence', 0)}<br>
                    <strong>🎯 Business Impact:</strong> {business_impact}<br>
                    <strong>⏰ Remediation Priority:</strong> {remediation_priority}
                </div>
            </div>
            
            <div style="margin-top: 15px;">
                <strong>💡 Actionable Insights:</strong><br>
                {self._generate_actionable_insights(finding)}
            </div>
        </div>
        """

    def _generate_actionable_insights(self, finding: Dict) -> str:
        """Generate actionable insights for a finding"""
        insights = finding.get('actionable_insights', [])
        if not insights:
            insights = self._get_actionable_insights(finding)
            
        return "<ul>" + "".join(f"<li>{insight}</li>" for insight in insights) + "</ul>"

    def _generate_remediation_section(self, results: Dict) -> str:
        """Generate remediation roadmap section"""
        roadmap = self._generate_remediation_roadmap(results)
        
        return f"""
        <div class="executive-summary">
            <h2>🛠️ Remediation Roadmap</h2>
            
            <h3>🚨 Immediate Actions (24-48 hours)</h3>
            <ul>
                {"".join(f"<li>{task}</li>" for task in roadmap['immediate_actions']['tasks'])}
            </ul>
            
            <h3>📅 Short-term Actions (1 week)</h3>
            <ul>
                {"".join(f"<li>{task}</li>" for task in roadmap['short_term_actions']['tasks'])}
            </ul>
            
            <h3>🎯 Long-term Improvements (1 month)</h3>
            <ul>
                {"".join(f"<li>{task}</li>" for task in roadmap['long_term_improvements']['tasks'])}
            </ul>
        </div>
        """

    def save_executive_summary(self, results: Dict[str, Any]) -> str:
        """Save standalone executive summary"""
        file_path = os.path.join(self.output_dir, "executive_summary.md")
        
        exec_summary = self._generate_executive_summary_data(results)
        risk_assessment = self._generate_risk_assessment(results)
        
        summary_content = f"""# Executive Security Summary - {self.domain}

## 📊 Quick Overview
- **Overall Risk Level**: {risk_assessment['risk_level']}
- **Total Findings**: {exec_summary['key_findings_count']}
- **Critical Issues**: {exec_summary['critical_issues']}
- **High Priority Issues**: {exec_summary['high_priority_issues']}

## 🎯 Immediate Concerns
{chr(10).join(f"- {issue}" for issue in exec_summary['next_steps'])}

## 💡 Recommended Actions
{chr(10).join(f"- {action}" for action in exec_summary['recommended_actions'])}

## 📈 Business Impact
{exec_summary['business_impact']}

---
*Generated by LeakHunterX on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
            
        return file_path

    def save_additional_files(self, results: Dict[str, Any]):
        """Save additional data files"""
        # Save subdomains
        if results.get("subdomains"):
            with open(os.path.join(self.output_dir, "subdomains.txt"), "w") as f:
                for subdomain in results["subdomains"]:
                    f.write(f"{subdomain}\n")

        # Save URLs
        if results.get("urls"):
            with open(os.path.join(self.output_dir, "urls.txt"), "w") as f:
                for url in results["urls"]:
                    f.write(f"{url}\n")

        # Save JS files
        if results.get("js_files"):
            with open(os.path.join(self.output_dir, "js_files.txt"), "w") as f:
                for js_file in results["js_files"]:
                    f.write(f"{js_file}\n")

        # Save high-confidence leaks
        if results.get("high_confidence_leaks"):
            with open(os.path.join(self.output_dir, "high_confidence_leaks.json"), "w") as f:
                json.dump(results["high_confidence_leaks"], f, indent=2)

    def _get_actionable_insights(self, leak: Dict) -> List[str]:
        """Generate actionable insights based on leak type"""
        leak_type = leak.get('type', '').lower()
        insights = []

        if any(keyword in leak_type for keyword in ['api_key', 'secret', 'token']):
            insights.extend([
                "Immediate credential rotation required",
                "Review access logs for unauthorized usage",
                "Check associated services for compromise",
                "Implement key management system"
            ])

        elif 'password' in leak_type or 'credential' in leak_type:
            insights.extend([
                "Immediate password reset required",
                "Check for credential reuse across systems",
                "Review authentication logs",
                "Implement multi-factor authentication"
            ])

        elif 'endpoint' in leak_type or 'admin' in leak_type:
            insights.extend([
                "Verify authentication requirements",
                "Review access control configurations",
                "Monitor for unauthorized access attempts",
                "Consider implementing WAF rules"
            ])

        else:
            insights.extend([
                "Manual review required for context assessment",
                "Determine potential business impact",
                "Implement appropriate monitoring",
                "Document for future reference"
            ])

        return insights

    def get_report_paths(self) -> Dict[str, str]:
        """Get paths to all generated reports"""
        return {
            "json_report": os.path.join(self.output_dir, "scan_results.json"),
            "html_report": os.path.join(self.output_dir, "professional_security_report.html"),
            "executive_summary": os.path.join(self.output_dir, "executive_summary.md"),
            "subdomains": os.path.join(self.output_dir, "subdomains.txt"),
            "urls": os.path.join(self.output_dir, "urls.txt"),
            "js_files": os.path.join(self.output_dir, "js_files.txt"),
            "high_confidence_leaks": os.path.join(self.output_dir, "high_confidence_leaks.json")
        }
