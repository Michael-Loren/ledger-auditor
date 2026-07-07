"""Generate the synthetic demo dataset: lease PDF, insurance policy PDF,
12 months of bank statements (CSV + 2 PDF renditions), ground-truth findings,
and the 50-question eval set with answers computed from the generated data.

Everything is deterministic (fixed seed) so eval numbers are reproducible.

Usage:  python scripts/generate_data.py [--out data]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

SEED = 20250101
YEAR = 2025

# ---------------------------------------------------------------- persona / contract facts
PERSONA = {
    "name": "Alex Rivera",
    "address": "412 Maple Avenue, Apt 3B, Springfield, IL 62704",
    "landlord": "Maple Properties LLC",
    "insurer": "Granite Shield Insurance Co.",
    "policy_number": "RS-2024-118203",
}

LEASE = {
    "start": "2024-02-01",
    "end": "2026-01-31",
    "base_rent": 1850.00,
    "rent_due_day": 1,
    "late_grace_day": 5,
    "late_fee": 50.00,
    "increase_cap_pct": 3.0,
    "increase_notice_days": 60,
    "security_deposit": 2775.00,
    "pet_fee": 35.00,
    "parking_fee": 75.00,
}

INSURANCE = {
    "monthly_premium": 18.50,
    "deductible": 500.00,
    "personal_property_limit": 30000.00,
    "liability_limit": 100000.00,
    "effective": "2025-01-01",
}

# Planted anomalies -------------------------------------------------------
RAISED_RENT = 1942.50          # 5% increase effective July 1 — cap allows 3% (max 1905.50)
ALLOWED_RENT = round(LEASE["base_rent"] * 1.03, 2)          # 1905.50
LATE_FEE_CHARGED = 75.00       # April: lease says $50
PET_FEE_CHARGED_LATE = 45.00   # Sep-Dec: lease says $35
INSURANCE_CHARGED_LATE = 24.75 # Sep-Dec: policy says $18.50
STREAMAX_PRICES = {1: 12.99, 5: 15.99, 10: 17.99}  # price creep at month boundaries


def month_price(prices: dict[int, float], month: int) -> float:
    p = None
    for m in sorted(prices):
        if month >= m:
            p = prices[m]
    return p


# ---------------------------------------------------------------- transactions
def build_transactions() -> list[dict]:
    rng = random.Random(SEED)
    txns: list[dict] = []

    def add(d: date, desc: str, amount: float, category: str):
        txns.append({
            "date": d.isoformat(),
            "description": desc,
            "amount": round(amount, 2),
            "category": category,
        })

    for month in range(1, 13):
        first = date(YEAR, month, 1)

        # income
        add(first, "DIRECT DEPOSIT - NORTHWIND ANALYTICS PAYROLL", 2410.00, "income")
        add(date(YEAR, month, 15), "DIRECT DEPOSIT - NORTHWIND ANALYTICS PAYROLL", 2410.00, "income")

        # rent (April paid late on the 9th; July onward landlord charges raised rent)
        rent = RAISED_RENT if month >= 7 else LEASE["base_rent"]
        rent_day = 9 if month == 4 else 1
        add(date(YEAR, month, rent_day), "MAPLE PROPERTIES LLC - RENT", -rent, "housing")
        if month == 4:
            add(date(YEAR, month, 9), "MAPLE PROPERTIES LLC - LATE FEE", -LATE_FEE_CHARGED, "housing")

        # pet fee ($45 from September, lease says $35) and parking
        pet = PET_FEE_CHARGED_LATE if month >= 9 else LEASE["pet_fee"]
        add(first, "MAPLE PROPERTIES LLC - PET FEE", -pet, "housing")
        add(first, "MAPLE PROPERTIES LLC - PARKING", -LEASE["parking_fee"], "housing")

        # renters insurance autopay ($24.75 from September, policy says $18.50)
        ins = INSURANCE_CHARGED_LATE if month >= 9 else INSURANCE["monthly_premium"]
        add(date(YEAR, month, 3), "GRANITE SHIELD INS PREMIUM AUTOPAY", -ins, "insurance")

        # utilities & subscriptions
        add(date(YEAR, month, 12), "CITY POWER & LIGHT", -round(rng.uniform(48, 112), 2), "utilities")
        add(date(YEAR, month, 8), "COMLINK INTERNET", -59.99, "utilities")
        add(date(YEAR, month, 20), "TELCO MOBILE", -40.00, "utilities")
        add(date(YEAR, month, 6), "FITSTREAM PRO SUBSCRIPTION", -39.99, "subscriptions")
        add(date(YEAR, month, 17), f"STREAMAX MONTHLY", -month_price(STREAMAX_PRICES, month), "subscriptions")
        add(date(YEAR, month, 2), "IRONWORKS GYM", -45.00, "subscriptions")

        # groceries: every ~7 days
        d = first + timedelta(days=rng.randint(1, 4))
        while d.month == month:
            add(d, "WHOLEFIELDS MARKET", -round(rng.uniform(58, 142), 2), "groceries")
            d += timedelta(days=rng.randint(6, 8))

        # dining / gas / misc
        for _ in range(rng.randint(4, 7)):
            day = rng.randint(1, 28)
            add(date(YEAR, month, day), rng.choice(
                ["CORNER BISTRO", "PHO HOUSE 88", "LUNA CAFE", "TACOS DEL REY"]),
                -round(rng.uniform(14, 68), 2), "dining")
        for _ in range(rng.randint(2, 4)):
            add(date(YEAR, month, rng.randint(1, 28)), "GAS-N-GO #114",
                -round(rng.uniform(28, 52), 2), "transport")

        # duplicate charge anomaly: CLEANCO billed twice same day in March
        if month == 3:
            add(date(YEAR, 3, 14), "CLEANCO HOME SERVICES", -120.00, "services")
            add(date(YEAR, 3, 14), "CLEANCO HOME SERVICES", -120.00, "services")

    txns.sort(key=lambda t: t["date"])
    return txns


# ---------------------------------------------------------------- PDFs
def _styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=8)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=16)
    return title, h, body


def write_lease_pdf(path: Path):
    title, h, body = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    L = LEASE
    sections = [
        ("RESIDENTIAL LEASE AGREEMENT", None),
        ("1. Parties and Premises",
         f"This Residential Lease Agreement is entered into between {PERSONA['landlord']} "
         f"(\"Landlord\") and {PERSONA['name']} (\"Tenant\") for the premises located at "
         f"{PERSONA['address']} (the \"Premises\")."),
        ("2. Term",
         f"The lease term begins on February 1, 2024 and ends on January 31, 2026, "
         f"unless renewed in writing by both parties at least 30 days before expiration."),
        ("3. Rent",
         f"Tenant shall pay base monthly rent of ${L['base_rent']:,.2f}, due on the 1st day "
         f"of each calendar month, payable to {PERSONA['landlord']}."),
        ("4. Late Payments",
         f"Rent received after the {L['late_grace_day']}th day of the month is late. "
         f"Landlord may charge a late fee of ${L['late_fee']:.2f} for any late payment. "
         f"No other late charge, penalty, or interest may be assessed."),
        ("5. Rent Increases",
         f"Landlord may increase the base rent no more than once per twelve-month period. "
         f"Any increase shall not exceed {L['increase_cap_pct']:.0f}% of the then-current "
         f"base rent and requires written notice to Tenant at least "
         f"{L['increase_notice_days']} days before the effective date."),
        ("6. Security Deposit",
         f"Tenant has paid a security deposit of ${L['security_deposit']:,.2f}, refundable "
         f"within 30 days of move-out less documented damages beyond normal wear and tear."),
        ("7. Pets",
         f"Tenant is permitted one cat. Tenant shall pay a monthly pet fee of "
         f"${L['pet_fee']:.2f}. The pet fee may not be modified during the lease term."),
        ("8. Parking",
         f"One assigned parking space (Space 14) is provided under the parking addendum at "
         f"${L['parking_fee']:.2f} per month."),
        ("9. Utilities",
         "Tenant is responsible for electricity, internet, and telephone service. "
         "Landlord pays for water, sewer, and trash collection."),
        ("10. Maintenance and Entry",
         "Landlord shall maintain the Premises in habitable condition. Landlord may enter "
         "with 24 hours' notice except in emergencies."),
        ("11. Governing Law",
         "This Agreement is governed by the laws of the State of Illinois."),
    ]
    flow = []
    for head, text in sections:
        if text is None:
            flow += [Paragraph(head, title), Spacer(1, 12)]
        else:
            flow += [Paragraph(head, h), Paragraph(text, body)]
    flow += [Spacer(1, 24),
             Paragraph("Signed: January 12, 2024 — Maple Properties LLC / Alex Rivera", body)]
    doc.build(flow)


def write_insurance_pdf(path: Path):
    title, h, body = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    I = INSURANCE
    sections = [
        ("RENTERS INSURANCE POLICY — DECLARATIONS", None),
        ("Policy Information",
         f"Insurer: {PERSONA['insurer']}. Policy Number: {PERSONA['policy_number']}. "
         f"Named Insured: {PERSONA['name']}, {PERSONA['address']}. "
         f"Policy Period: January 1, 2025 to January 1, 2026."),
        ("Premium",
         f"The total monthly premium is ${I['monthly_premium']:.2f}, payable by automatic "
         f"bank draft on or about the 3rd of each month. Premium changes require 30 days' "
         f"advance written notice to the insured and apply only at renewal."),
        ("Coverage A — Personal Property",
         f"Limit: ${I['personal_property_limit']:,.2f}. Deductible: ${I['deductible']:,.2f} "
         f"per occurrence."),
        ("Coverage B — Personal Liability",
         f"Limit: ${I['liability_limit']:,.2f} per occurrence."),
        ("Coverage C — Loss of Use",
         "Limit: $9,000.00 (30% of Coverage A)."),
        ("Exclusions",
         "Flood, earthquake, and intentional acts are excluded. Water damage from burst "
         "pipes is covered; seepage over 14 days is not."),
    ]
    flow = []
    for head, text in sections:
        if text is None:
            flow += [Paragraph(head, title), Spacer(1, 12)]
        else:
            flow += [Paragraph(head, h), Paragraph(text, body)]
    doc.build(flow)


def write_statement_pdf(path: Path, month: int, txns: list[dict]):
    """Render one month's statement as a PDF table (exercises PDF table extraction)."""
    title, h, body = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    rows = [["Date", "Description", "Amount"]]
    for t in txns:
        rows.append([t["date"], t["description"], f"{t['amount']:.2f}"])
    table = Table(rows, colWidths=[1.1 * inch, 4.2 * inch, 1.1 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
    ]))
    flow = [Paragraph(f"FIRST PRAIRIE BANK — Account Statement {YEAR}-{month:02d}", title),
            Paragraph(f"Account holder: {PERSONA['name']} — Checking ****4417", body),
            Spacer(1, 10), table]
    doc.build(flow)


# ---------------------------------------------------------------- ground truth + eval set
def build_ground_truth(txns: list[dict]) -> dict:
    overcharge = round(RAISED_RENT - ALLOWED_RENT, 2)
    return {
        "persona": PERSONA, "lease": LEASE, "insurance": INSURANCE,
        "findings": [
            {"id": "F1", "type": "rent_increase_violation",
             "detail": f"Rent rose from $1850.00 to ${RAISED_RENT:.2f} in July (+5.0%); "
                       f"lease caps increases at 3% (${ALLOWED_RENT:.2f}). "
                       f"Overcharge ${overcharge:.2f}/month for 6 months = ${overcharge*6:.2f}.",
             "sources": ["lease", "transactions"]},
            {"id": "F2", "type": "late_fee_overcharge",
             "detail": f"April late fee charged ${LATE_FEE_CHARGED:.2f}; lease allows $50.00.",
             "sources": ["lease", "transactions"]},
            {"id": "F3", "type": "pet_fee_overcharge",
             "detail": f"Pet fee charged ${PET_FEE_CHARGED_LATE:.2f}/month Sep-Dec; lease "
                       f"fixes it at $35.00 and forbids modification.",
             "sources": ["lease", "transactions"]},
            {"id": "F4", "type": "duplicate_charge",
             "detail": "CLEANCO HOME SERVICES charged $120.00 twice on 2025-03-14.",
             "sources": ["transactions"]},
            {"id": "F5", "type": "insurance_premium_mismatch",
             "detail": f"Insurance autopay rose to ${INSURANCE_CHARGED_LATE:.2f} from Sep; "
                       f"policy premium is $18.50 and changes apply only at renewal.",
             "sources": ["insurance", "transactions"]},
            {"id": "F6", "type": "subscription_price_creep",
             "detail": "STREAMAX rose 12.99 -> 15.99 (May) -> 17.99 (Oct).",
             "sources": ["transactions"]},
        ],
    }


def build_questions(txns: list[dict]) -> list[dict]:
    """50 questions with ground-truth answers computed from the generated data."""
    def total(pred):
        return round(sum(-t["amount"] for t in txns if pred(t)), 2)

    def count(pred):
        return sum(1 for t in txns if pred(t))

    q: list[dict] = []

    def add(question, answer, *, numeric=None, tol=0.01, sources=None,
            category="lookup", retrieval_eval=False):
        q.append({"id": f"Q{len(q)+1:02d}", "question": question, "answer": str(answer),
                  "numeric": numeric, "tolerance": tol,
                  "expected_sources": sources or [], "category": category,
                  "retrieval_eval": retrieval_eval})

    R = lambda doc, frag: {"doc": doc, "must_contain": frag}

    # ---- lease lookups (10)
    add("What is the base monthly rent under the lease?", "$1,850.00", numeric=1850.0,
        sources=[R("lease", "base monthly rent")], retrieval_eval=True)
    add("What late fee does the lease allow?", "$50.00", numeric=50.0,
        sources=[R("lease", "late fee")], retrieval_eval=True)
    add("After what day of the month is rent considered late?", "the 5th", numeric=5,
        sources=[R("lease", "late")], retrieval_eval=True)
    add("What is the maximum annual rent increase allowed by the lease?", "3%", numeric=3.0,
        sources=[R("lease", "increase")], retrieval_eval=True)
    add("How many days of written notice are required before a rent increase?", "60 days",
        numeric=60, sources=[R("lease", "notice")], retrieval_eval=True)
    add("How much is the security deposit?", "$2,775.00", numeric=2775.0,
        sources=[R("lease", "security deposit")], retrieval_eval=True)
    add("What is the monthly pet fee in the lease?", "$35.00", numeric=35.0,
        sources=[R("lease", "pet fee")], retrieval_eval=True)
    add("How much is the parking space per month?", "$75.00", numeric=75.0,
        sources=[R("lease", "parking")], retrieval_eval=True)
    add("Who pays for water under the lease?", "the landlord",
        sources=[R("lease", "water")], retrieval_eval=True)
    add("When does the lease term end?", "January 31, 2026",
        sources=[R("lease", "term")], retrieval_eval=True)

    # ---- insurance lookups (6)
    add("What is the monthly renters insurance premium?", "$18.50", numeric=18.50,
        sources=[R("insurance", "premium")], retrieval_eval=True)
    add("What is the insurance deductible per occurrence?", "$500.00", numeric=500.0,
        sources=[R("insurance", "deductible")], retrieval_eval=True)
    add("What is the personal liability coverage limit?", "$100,000.00", numeric=100000.0,
        sources=[R("insurance", "liability")], retrieval_eval=True)
    add("What is the personal property coverage limit?", "$30,000.00", numeric=30000.0,
        sources=[R("insurance", "personal property")], retrieval_eval=True)
    add("Who is the renters insurance carrier?", "Granite Shield Insurance Co.",
        sources=[R("insurance", "insurer")], retrieval_eval=True)
    add("How much notice is required before an insurance premium change?", "30 days",
        numeric=30, sources=[R("insurance", "notice")], retrieval_eval=True)

    # ---- transaction aggregates (16)
    add("How much was spent at WHOLEFIELDS MARKET in March 2025?",
        total(lambda t: t["description"] == "WHOLEFIELDS MARKET" and t["date"][5:7] == "03"),
        numeric=total(lambda t: t["description"] == "WHOLEFIELDS MARKET" and t["date"][5:7] == "03"),
        category="aggregate")
    add("What was total grocery spending in 2025?",
        total(lambda t: t["category"] == "groceries"),
        numeric=total(lambda t: t["category"] == "groceries"), category="aggregate")
    add("How many FITSTREAM PRO charges were there in 2025?",
        count(lambda t: "FITSTREAM" in t["description"]),
        numeric=count(lambda t: "FITSTREAM" in t["description"]), category="aggregate")
    add("What is the monthly FITSTREAM PRO subscription price?", "$39.99", numeric=39.99,
        category="aggregate")
    add("How much rent (base rent line only) was paid in total during 2025?",
        total(lambda t: t["description"] == "MAPLE PROPERTIES LLC - RENT"),
        numeric=total(lambda t: t["description"] == "MAPLE PROPERTIES LLC - RENT"),
        category="aggregate")
    add("What rent amount was charged in January 2025?", "$1,850.00", numeric=1850.0,
        category="aggregate")
    add("What rent amount was charged in August 2025?", f"${RAISED_RENT:,.2f}",
        numeric=RAISED_RENT, category="aggregate")
    add("What was total spending on dining in 2025?",
        total(lambda t: t["category"] == "dining"),
        numeric=total(lambda t: t["category"] == "dining"), category="aggregate")
    add("How much did CITY POWER & LIGHT bills total in 2025?",
        total(lambda t: t["description"] == "CITY POWER & LIGHT"),
        numeric=total(lambda t: t["description"] == "CITY POWER & LIGHT"), category="aggregate")
    add("What is the monthly internet bill?", "$59.99", numeric=59.99, category="aggregate")
    add("How many paychecks were deposited in 2025?",
        count(lambda t: t["category"] == "income"),
        numeric=count(lambda t: t["category"] == "income"), category="aggregate")
    add("What is the amount of each payroll direct deposit?", "$2,410.00", numeric=2410.0,
        category="aggregate")
    add("What was total parking paid to the landlord in 2025?",
        total(lambda t: "PARKING" in t["description"]),
        numeric=total(lambda t: "PARKING" in t["description"]), category="aggregate")
    add("How much was paid to IRONWORKS GYM over the year?",
        total(lambda t: "IRONWORKS" in t["description"]),
        numeric=total(lambda t: "IRONWORKS" in t["description"]), category="aggregate")
    add("What was the insurance autopay amount in February 2025?", "$18.50", numeric=18.50,
        category="aggregate")
    add("What was the insurance autopay amount in October 2025?",
        f"${INSURANCE_CHARGED_LATE:.2f}", numeric=INSURANCE_CHARGED_LATE, category="aggregate")

    # ---- cross-document audit (14)
    overcharge = round(RAISED_RENT - ALLOWED_RENT, 2)
    add("Did the July 2025 rent increase comply with the lease?",
        f"No. Rent rose 5% to ${RAISED_RENT:,.2f}; the lease caps increases at 3% "
        f"(max ${ALLOWED_RENT:,.2f}).",
        sources=[R("lease", "increase")], category="audit", retrieval_eval=True)
    add("What is the maximum rent the landlord could legally charge after one 3% increase?",
        f"${ALLOWED_RENT:,.2f}", numeric=ALLOWED_RENT,
        sources=[R("lease", "increase")], category="audit", retrieval_eval=True)
    add("By how much per month was Alex overcharged on rent starting July 2025?",
        f"${overcharge:.2f}", numeric=overcharge,
        sources=[R("lease", "increase")], category="audit", retrieval_eval=True)
    add("What is the total rent overcharge for July through December 2025?",
        f"${overcharge*6:.2f}", numeric=round(overcharge * 6, 2),
        sources=[R("lease", "increase")], category="audit", retrieval_eval=True)
    add("Was the April 2025 late fee charged correctly?",
        f"No. ${LATE_FEE_CHARGED:.2f} was charged; the lease allows only $50.00.",
        sources=[R("lease", "late fee")], category="audit", retrieval_eval=True)
    add("Was charging a late fee in April justified at all?",
        "Yes — rent was paid on April 9, after the 5th-of-month grace period.",
        sources=[R("lease", "late")], category="audit", retrieval_eval=True)
    add("Is the pet fee being charged in November 2025 consistent with the lease?",
        f"No. ${PET_FEE_CHARGED_LATE:.2f} was charged; the lease fixes the pet fee at "
        f"$35.00 and says it may not be modified.",
        sources=[R("lease", "pet fee")], category="audit", retrieval_eval=True)
    add("What is the total pet fee overcharge for September through December 2025?",
        "$40.00", numeric=40.0, sources=[R("lease", "pet fee")],
        category="audit", retrieval_eval=True)
    add("Does the insurance autopay from September onward match the policy premium?",
        f"No. ${INSURANCE_CHARGED_LATE:.2f} was drafted; the declarations page lists "
        f"$18.50/month, and changes apply only at renewal.",
        sources=[R("insurance", "premium")], category="audit", retrieval_eval=True)
    add("Were there any duplicate charges in 2025?",
        "Yes — CLEANCO HOME SERVICES charged $120.00 twice on 2025-03-14.",
        category="audit")
    add("Which subscription changed price during 2025, and how?",
        "STREAMAX: $12.99 to $15.99 in May, then $17.99 in October.",
        category="audit")
    add("How much extra did the STREAMAX price increases cost versus staying at $12.99 all year?",
        "$30.00", numeric=round(sum(month_price(STREAMAX_PRICES, m) - 12.99
                                    for m in range(1, 13)), 2), category="audit")
    add("If the landlord refunds all rent, late-fee, and pet-fee overcharges, how much is owed?",
        f"${round(overcharge*6 + (LATE_FEE_CHARGED-50.0) + 40.0, 2):.2f}",
        numeric=round(overcharge * 6 + (LATE_FEE_CHARGED - 50.0) + 40.0, 2),
        sources=[R("lease", "increase"), R("lease", "late fee"), R("lease", "pet fee")],
        category="audit", retrieval_eval=True)
    add("Is water service something Alex should be paying for?",
        "No — the lease assigns water, sewer, and trash to the landlord.",
        sources=[R("lease", "water")], category="audit", retrieval_eval=True)

    # ---- distractor / robustness lookups (4)
    add("Can the landlord enter the apartment without notice?",
        "Only in emergencies; otherwise 24 hours' notice is required.",
        sources=[R("lease", "24 hours")], retrieval_eval=True)
    add("Is flood damage covered by the renters policy?", "No, flood is excluded.",
        sources=[R("insurance", "flood")], retrieval_eval=True)
    add("What is the loss-of-use coverage limit?", "$9,000.00", numeric=9000.0,
        sources=[R("insurance", "loss of use")], retrieval_eval=True)
    add("Which state's law governs the lease?", "Illinois",
        sources=[R("lease", "illinois")], retrieval_eval=True)

    assert len(q) == 50, f"expected 50 questions, got {len(q)}"
    return q


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "statements").mkdir(parents=True, exist_ok=True)

    txns = build_transactions()

    # CSVs for all 12 months
    for month in range(1, 13):
        mt = [t for t in txns if int(t["date"][5:7]) == month]
        with open(out / "statements" / f"{YEAR}-{month:02d}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "description", "amount", "category"])
            w.writeheader()
            w.writerows(mt)

    # two months also rendered as PDF statements (exercises PDF table extraction)
    for month in (2, 8):
        mt = [t for t in txns if int(t["date"][5:7]) == month]
        write_statement_pdf(out / "statements" / f"{YEAR}-{month:02d}.pdf", month, mt)

    write_lease_pdf(out / "docs" / "lease.pdf")
    write_insurance_pdf(out / "docs" / "insurance_policy.pdf")

    with open(out / "ground_truth.json", "w") as f:
        json.dump(build_ground_truth(txns), f, indent=2)
    with open(out / "questions.jsonl", "w") as f:
        for item in build_questions(txns):
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(txns)} transactions, 2 doc PDFs, 12 CSV + 2 PDF statements, "
          f"ground truth, and 50 eval questions to {out}/")


if __name__ == "__main__":
    main()
