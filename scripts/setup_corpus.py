"""Generate synthetic business documents with cross-references for multi-hop QA."""

import json
import os

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")

# Shared entities that cross-reference across documents
COMPANIES = [
    {"name": "Apex Corp", "id": "apex_corp", "industry": "Technology", "hq": "San Francisco, CA"},
    {"name": "Meridian Ltd", "id": "meridian_ltd", "industry": "Manufacturing", "hq": "Chicago, IL"},
    {"name": "Vanguard Systems", "id": "vanguard_systems", "industry": "Defense", "hq": "Arlington, VA"},
    {"name": "Pinnacle Health", "id": "pinnacle_health", "industry": "Healthcare", "hq": "Boston, MA"},
    {"name": "Summit Financial Group", "id": "summit_financial", "industry": "Finance", "hq": "New York, NY"},
    {"name": "Crestline Energy", "id": "crestline_energy", "industry": "Energy", "hq": "Houston, TX"},
    {"name": "Horizon Logistics", "id": "horizon_logistics", "industry": "Logistics", "hq": "Memphis, TN"},
    {"name": "Atlas Consulting", "id": "atlas_consulting", "industry": "Consulting", "hq": "Washington, DC"},
]


def generate_financial_reports() -> list[dict]:
    """Generate 10 financial reports with realistic cross-referencing data."""
    reports = []
    financials = [
        {"company": COMPANIES[0], "year": 2024, "revenue": 142.5, "expenses": 114.3,
         "net_income": 28.2, "yoy_growth": 12.3,
         "notes": "Revenue growth driven by cloud services division. Signed $45M contract with Meridian Ltd for supply chain automation. Invested $8.5M in Crestline Energy renewable initiative."},
        {"company": COMPANIES[0], "year": 2023, "revenue": 126.9, "expenses": 105.1,
         "net_income": 21.8, "yoy_growth": 8.7,
         "notes": "Expanded operations to Chicago office. Partnership with Vanguard Systems contributed $18M in defense contracts. Atlas Consulting engaged for digital transformation at $3.2M."},
        {"company": COMPANIES[1], "year": 2024, "revenue": 89.7, "expenses": 75.3,
         "net_income": 14.4, "yoy_growth": 6.2,
         "notes": "Manufacturing output increased 15%. Apex Corp automation contract worth $45M over 3 years. Horizon Logistics handles all North American shipping at $12M/year."},
        {"company": COMPANIES[1], "year": 2023, "revenue": 84.5, "expenses": 72.1,
         "net_income": 12.4, "yoy_growth": 4.1,
         "notes": "Opened new plant in Detroit. Pinnacle Health medical device contract generated $22M. Quality compliance audit by Atlas Consulting."},
        {"company": COMPANIES[2], "year": 2024, "revenue": 215.8, "expenses": 178.4,
         "net_income": 37.4, "yoy_growth": 9.8,
         "notes": "Defense contracts up 14%. Joint venture with Apex Corp on cybersecurity platform worth $32M. Summit Financial Group manages pension fund at $890M AUM."},
        {"company": COMPANIES[3], "year": 2024, "revenue": 67.3, "expenses": 58.9,
         "net_income": 8.4, "yoy_growth": 11.5,
         "notes": "Telehealth revenue grew 45%. Supply contract with Meridian Ltd for medical devices at $22M/year. Crestline Energy provides facility power at preferential rates."},
        {"company": COMPANIES[4], "year": 2024, "revenue": 312.6, "expenses": 245.8,
         "net_income": 66.8, "yoy_growth": 7.4,
         "notes": "AUM grew to $48.2B. Manages Vanguard Systems pension ($890M) and Apex Corp 401k ($125M). Advisory fees from Crestline Energy IPO preparation at $15M."},
        {"company": COMPANIES[5], "year": 2024, "revenue": 178.4, "expenses": 152.1,
         "net_income": 26.3, "yoy_growth": 15.7,
         "notes": "Renewable energy division grew 34%. Apex Corp invested $8.5M in solar initiative. Supplies power to Pinnacle Health at $4.2M/year. IPO planned for Q2 2025 with Summit Financial as advisor."},
        {"company": COMPANIES[6], "year": 2024, "revenue": 56.8, "expenses": 49.2,
         "net_income": 7.6, "yoy_growth": 5.3,
         "notes": "Fleet expanded by 120 vehicles. Primary shipper for Meridian Ltd ($12M/year) and Pinnacle Health ($6.8M/year). Atlas Consulting optimized route efficiency, saving $2.1M."},
        {"company": COMPANIES[7], "year": 2024, "revenue": 34.2, "expenses": 27.8,
         "net_income": 6.4, "yoy_growth": 18.9,
         "notes": "Hired 45 new consultants. Engaged by Apex Corp ($3.2M), Meridian Ltd ($1.8M), and Horizon Logistics ($1.5M). Compliance audit work for Vanguard Systems at $2.7M."},
    ]

    for f in financials:
        c = f["company"]
        doc_id = f"{c['id']}_{f['year']}_financial"
        text = f"""{c['name']} — Annual Financial Report {f['year']}

Headquarters: {c['hq']}
Industry: {c['industry']}

Financial Summary:
- Revenue: ${f['revenue']}M
- Operating Expenses: ${f['expenses']}M
- Net Income: ${f['net_income']}M
- Year-over-Year Revenue Growth: {f['yoy_growth']}%

Notes:
{f['notes']}

Report filed with SEC on March 15, {f['year'] + 1}."""

        reports.append({
            "doc_id": doc_id,
            "title": f"{c['name']} Financial Report {f['year']}",
            "text": text,
            "metadata": {
                "type": "financial_report",
                "company": c["name"],
                "company_id": c["id"],
                "year": f["year"],
                "revenue_millions": f["revenue"],
                "net_income_millions": f["net_income"],
            },
        })
    return reports


def generate_contracts() -> list[dict]:
    """Generate 10 contracts between companies in the corpus."""
    contracts_data = [
        {"parties": [COMPANIES[0], COMPANIES[1]], "type": "Technology Services",
         "value": 45.0, "term": "3 years", "effective": "2024-01-15", "expiry": "2027-01-14",
         "governing_law": "State of California",
         "description": "Supply chain automation platform development and deployment",
         "liability_cap": 90.0, "termination": "90 days written notice; immediate for material breach",
         "special": "Includes IP assignment clause for custom modules. Apex Corp retains platform IP."},
        {"parties": [COMPANIES[1], COMPANIES[3]], "type": "Supply Agreement",
         "value": 22.0, "term": "2 years", "effective": "2023-06-01", "expiry": "2025-05-31",
         "governing_law": "State of Illinois",
         "description": "Medical device manufacturing and supply",
         "liability_cap": 44.0, "termination": "60 days written notice",
         "special": "FDA compliance required. Meridian Ltd liable for manufacturing defects."},
        {"parties": [COMPANIES[0], COMPANIES[2]], "type": "Joint Venture",
         "value": 32.0, "term": "5 years", "effective": "2024-03-01", "expiry": "2029-02-28",
         "governing_law": "State of Virginia",
         "description": "Cybersecurity platform for defense applications",
         "liability_cap": 64.0, "termination": "180 days written notice; requires DoD approval",
         "special": "ITAR restricted. Security clearance required for all personnel. Revenue split 60/40 Vanguard/Apex."},
        {"parties": [COMPANIES[2], COMPANIES[4]], "type": "Financial Services",
         "value": 4.5, "term": "5 years", "effective": "2022-01-01", "expiry": "2026-12-31",
         "governing_law": "State of New York",
         "description": "Pension fund management ($890M AUM)",
         "liability_cap": 9.0, "termination": "120 days written notice",
         "special": "Performance benchmark: S&P 500 + 2%. Fee reduction if underperforming 2 consecutive quarters."},
        {"parties": [COMPANIES[5], COMPANIES[3]], "type": "Energy Supply",
         "value": 4.2, "term": "4 years", "effective": "2024-02-01", "expiry": "2028-01-31",
         "governing_law": "State of Texas",
         "description": "Renewable energy supply for hospital facilities",
         "liability_cap": 8.4, "termination": "60 days written notice; force majeure clause",
         "special": "100% renewable sources. Preferential rate of $0.068/kWh vs market $0.089/kWh. Rate locked for first 2 years."},
        {"parties": [COMPANIES[1], COMPANIES[6]], "type": "Logistics Services",
         "value": 12.0, "term": "3 years", "effective": "2023-09-01", "expiry": "2026-08-31",
         "governing_law": "State of Tennessee",
         "description": "North American shipping and distribution",
         "liability_cap": 24.0, "termination": "90 days written notice",
         "special": "SLA: 98.5% on-time delivery. Penalty of $50K per 0.1% below SLA per quarter."},
        {"parties": [COMPANIES[0], COMPANIES[7]], "type": "Consulting Services",
         "value": 3.2, "term": "1 year", "effective": "2024-04-01", "expiry": "2025-03-31",
         "governing_law": "State of California",
         "description": "Digital transformation strategy and implementation",
         "liability_cap": 6.4, "termination": "30 days written notice",
         "special": "Deliverables: roadmap by month 3, pilot by month 6, full rollout by month 12."},
        {"parties": [COMPANIES[5], COMPANIES[4]], "type": "Advisory Services",
         "value": 15.0, "term": "18 months", "effective": "2024-06-01", "expiry": "2025-11-30",
         "governing_law": "State of New York",
         "description": "IPO preparation and advisory for Crestline Energy",
         "liability_cap": 30.0, "termination": "Completion of IPO or mutual agreement",
         "special": "Success fee: 2.5% of IPO proceeds. Summit Financial exclusive advisor. Quiet period provisions apply."},
        {"parties": [COMPANIES[6], COMPANIES[3]], "type": "Logistics Services",
         "value": 6.8, "term": "2 years", "effective": "2024-01-01", "expiry": "2025-12-31",
         "governing_law": "State of Massachusetts",
         "description": "Medical supply chain and cold-chain logistics",
         "liability_cap": 13.6, "termination": "60 days written notice",
         "special": "Cold-chain compliance required. Temperature monitoring with real-time alerts. Horizon liable for spoilage due to temperature deviation."},
        {"parties": [COMPANIES[7], COMPANIES[2]], "type": "Consulting Services",
         "value": 2.7, "term": "1 year", "effective": "2024-07-01", "expiry": "2025-06-30",
         "governing_law": "State of Virginia",
         "description": "Compliance audit and regulatory advisory",
         "liability_cap": 5.4, "termination": "30 days written notice",
         "special": "ITAR compliance focus. All Atlas personnel must hold active security clearance. Deliverable: compliance gap analysis by month 4."},
    ]

    contracts = []
    for i, c in enumerate(contracts_data, 1):
        doc_id = f"contract_{c['parties'][0]['id']}_{c['parties'][1]['id']}"
        p1, p2 = c["parties"][0]["name"], c["parties"][1]["name"]
        text = f"""CONTRACT AGREEMENT — {c['type']}

Parties:
- Party A: {p1} ({c['parties'][0]['hq']})
- Party B: {p2} ({c['parties'][1]['hq']})

Effective Date: {c['effective']}
Expiration Date: {c['expiry']}
Term: {c['term']}
Contract Value: ${c['value']}M

Description:
{c['description']}

Liability:
- Cap: ${c['liability_cap']}M (2x contract value)
- Each party indemnifies the other for negligence and willful misconduct

Termination:
{c['termination']}

Governing Law: {c['governing_law']}

Special Provisions:
{c['special']}

Executed on {c['effective']} by authorized representatives of both parties."""

        contracts.append({
            "doc_id": doc_id,
            "title": f"{p1} — {p2} {c['type']} Contract",
            "text": text,
            "metadata": {
                "type": "contract",
                "parties": [c["parties"][0]["id"], c["parties"][1]["id"]],
                "contract_type": c["type"],
                "value_millions": c["value"],
                "effective_date": c["effective"],
                "expiry_date": c["expiry"],
            },
        })
    return contracts


def generate_compliance_reports() -> list[dict]:
    """Generate 8 compliance reports referencing companies and contracts."""
    reports_data = [
        {"company": COMPANIES[0], "department": "Cloud Services", "audit_date": "2024-09-15",
         "auditor": "Atlas Consulting", "status": "Compliant with findings",
         "findings": [
             "SOC 2 Type II certification renewed successfully",
             "3 minor access control gaps in development environment",
             "Meridian Ltd data handling meets contractual requirements",
         ],
         "remediation": "Access control gaps to be remediated by 2024-12-01. Development environment segmentation planned."},
        {"company": COMPANIES[1], "department": "Medical Devices", "audit_date": "2024-08-20",
         "auditor": "Atlas Consulting", "status": "Non-compliant — remediation required",
         "findings": [
             "FDA 21 CFR Part 820 audit identified 2 major findings",
             "Batch tracking system gaps for Pinnacle Health supply line",
             "Documentation incomplete for 12% of quality control tests",
         ],
         "remediation": "Corrective action plan submitted to FDA. Batch tracking system upgrade contracted to Apex Corp. Deadline: 2025-02-28."},
        {"company": COMPANIES[2], "department": "Cybersecurity Division", "audit_date": "2024-10-01",
         "auditor": "Internal + Atlas Consulting", "status": "Compliant",
         "findings": [
             "NIST 800-171 compliance verified",
             "ITAR controls for Apex Corp joint venture properly implemented",
             "All personnel security clearances current",
         ],
         "remediation": "No remediation required. Next audit scheduled for 2025-04-01."},
        {"company": COMPANIES[3], "department": "Telehealth Operations", "audit_date": "2024-07-10",
         "auditor": "External — HealthGuard Auditors", "status": "Compliant with observations",
         "findings": [
             "HIPAA compliance verified across all telehealth platforms",
             "Crestline Energy power supply meets 99.97% uptime SLA",
             "Horizon Logistics cold-chain temperature deviations: 2 incidents in Q2 (within contract tolerance)",
         ],
         "remediation": "Enhanced monitoring dashboard for cold-chain logistics requested from Horizon. Implementation by 2024-11-01."},
        {"company": COMPANIES[4], "department": "Fund Management", "audit_date": "2024-11-05",
         "auditor": "SEC Examination", "status": "Compliant",
         "findings": [
             "Fiduciary duty obligations met for all managed accounts",
             "Vanguard Systems pension fund performance: S&P 500 + 3.1% (exceeds benchmark)",
             "Apex Corp 401k administration properly documented",
             "Crestline Energy IPO advisory — Chinese wall procedures verified",
         ],
         "remediation": "No remediation required. Commended for Chinese wall implementation during Crestline IPO advisory."},
        {"company": COMPANIES[5], "department": "Environmental & Safety", "audit_date": "2024-06-15",
         "auditor": "EPA Regional Office", "status": "Compliant with minor findings",
         "findings": [
             "Emissions below permitted levels by 22%",
             "Solar installation at Pinnacle Health facilities meets safety standards",
             "Minor documentation gap in waste disposal records for Q1 2024",
         ],
         "remediation": "Waste disposal documentation process updated. Training completed for all field personnel by 2024-08-01."},
        {"company": COMPANIES[6], "department": "Fleet Operations", "audit_date": "2024-05-20",
         "auditor": "DOT Federal Motor Carrier Safety", "status": "Compliant with findings",
         "findings": [
             "Fleet safety rating: Satisfactory",
             "Cold-chain compliance for Pinnacle Health shipments verified",
             "2 vehicles exceeded hours-of-service limits in Q1 (corrected)",
             "Route optimization by Atlas Consulting reduced fuel consumption 8%",
         ],
         "remediation": "Electronic logging device upgrades for 15 older vehicles. Completion by 2024-09-01."},
        {"company": COMPANIES[7], "department": "Professional Standards", "audit_date": "2024-04-10",
         "auditor": "Internal Quality Review", "status": "Compliant",
         "findings": [
             "All consultant certifications current",
             "Conflict of interest review: no issues with concurrent Apex Corp and Meridian Ltd engagements",
             "Vanguard Systems ITAR clearance requirements met by all assigned staff",
         ],
         "remediation": "No remediation required. Annual clearance renewal process documented."},
    ]

    reports = []
    for r in reports_data:
        c = r["company"]
        doc_id = f"{c['id']}_compliance_{r['audit_date'].replace('-', '')}"
        findings_text = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(r["findings"]))
        text = f"""COMPLIANCE AUDIT REPORT

Company: {c['name']}
Department: {r['department']}
Audit Date: {r['audit_date']}
Auditor: {r['auditor']}
Overall Status: {r['status']}

Findings:
{findings_text}

Remediation Actions:
{r['remediation']}

This report is confidential and intended for authorized personnel only."""

        reports.append({
            "doc_id": doc_id,
            "title": f"{c['name']} — {r['department']} Compliance Report",
            "text": text,
            "metadata": {
                "type": "compliance_report",
                "company": c["name"],
                "company_id": c["id"],
                "department": r["department"],
                "audit_date": r["audit_date"],
                "status": r["status"],
            },
        })
    return reports


def main() -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)

    all_docs = generate_financial_reports() + generate_contracts() + generate_compliance_reports()

    for doc in all_docs:
        path = os.path.join(CORPUS_DIR, f"{doc['doc_id']}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)

    print(f"Generated {len(all_docs)} documents in {CORPUS_DIR}")
    for doc in all_docs:
        print(f"  {doc['doc_id']}: {doc['title']}")


if __name__ == "__main__":
    main()
