"""
Loan lookup tool for CrewAI agents.

Provides check_loan_application_status to retrieve loan details and escalation scores.
"""

import sys
sys.path.insert(0, '.')

from src.ingestion.dataset import LOAN_APPLICATIONS
from typing import Dict, Optional


def check_loan_application_status(record_id: str) -> Dict:
    """
    Check loan application status and compute escalation score.

    Args:
        record_id: Loan application record ID (e.g., "LOAN_0001")

    Returns:
        Dictionary with status, amount, days_since_created, fraud flag, escalation_score, and escalation_recommended

    Escalation Score Formula:
        escalation_score = 0.60 * fraud_flag + 0.40 * recency_normalized
        where:
        - fraud_flag = 1.0 if flagged_for_fraud_review else 0.0
        - recency_normalized = days_since_created / 30.0
        - Result rounded to 4 decimal places

    Escalation Threshold: 0.60
        - Any fraud-flagged application scores >= 0.60 (0.60 on day 0, up to 1.00 at day 30)
        - Non-fraud applications only escalate if aged severely (approaching 30 days)
        - This ensures fraud cases receive priority while avoiding false positives
    """

    # Find the record by record_id
    record = None
    for loan in LOAN_APPLICATIONS:
        if loan.record_id == record_id:
            record = loan
            break

    if not record:
        return {
            "error": f"Record {record_id} not found",
            "record_id": record_id,
        }

    # Compute escalation score
    fraud_flag = 1.0 if record.flagged_for_fraud_review else 0.0
    recency_normalized = record.days_since_created / 30.0

    escalation_score = round(
        (0.60 * fraud_flag) + (0.40 * recency_normalized),
        4
    )

    # Determine if escalation is recommended
    escalation_recommended = escalation_score >= 0.60

    return {
        "record_id": record_id,
        "status": record.status,
        "loan_amount_inr": record.loan_amount_inr,
        "days_since_created": record.days_since_created,
        "flagged_for_fraud_review": record.flagged_for_fraud_review,
        "escalation_score": escalation_score,
        "escalation_recommended": escalation_recommended,
    }


if __name__ == "__main__":
    # Demonstration of the lookup tool
    print("\n" + "=" * 80)
    print("LOAN APPLICATION STATUS LOOKUP DEMONSTRATION")
    print("=" * 80)

    print("\nEscalation Score Formula:")
    print("  escalation_score = 0.60 * fraud_flag + 0.40 * recency_normalized")
    print("  where:")
    print("    - fraud_flag = 1.0 if flagged_for_fraud_review else 0.0")
    print("    - recency_normalized = days_since_created / 30.0")
    print("\nEscalation Threshold: 0.60")
    print("  - Fraud-flagged applications: always >= 0.60")
    print("  - Non-fraud applications: escalate if severely aged")

    # Test with several records
    test_records = ["LOAN_0001", "LOAN_0010", "LOAN_0025", "LOAN_0040", "LOAN_0050"]

    print("\n" + "=" * 80)
    print("SAMPLE LOOKUPS")
    print("=" * 80)

    for record_id in test_records:
        result = check_loan_application_status(record_id)

        print(f"\n{record_id}:")
        print(f"  Status: {result['status']}")
        print(f"  Amount: INR {result['loan_amount_inr']:,}")
        print(f"  Days Old: {result['days_since_created']}")
        print(f"  Fraud Flagged: {result['flagged_for_fraud_review']}")
        print(f"  Escalation Score: {result['escalation_score']:.4f}")
        print(f"  Escalate? {result['escalation_recommended']}")

    print("\n" + "=" * 80)
    print("ESCALATION DISTRIBUTION")
    print("=" * 80)

    # Analyze escalation across all records
    escalated_count = 0
    escalated_records = []
    non_escalated_records = []

    for loan in LOAN_APPLICATIONS:
        result = check_loan_application_status(loan.record_id)
        if result['escalation_recommended']:
            escalated_count += 1
            escalated_records.append(loan.record_id)
        else:
            non_escalated_records.append(loan.record_id)

    escalation_rate = (escalated_count / len(LOAN_APPLICATIONS)) * 100

    print(f"\nTotal Applications: {len(LOAN_APPLICATIONS)}")
    print(f"Escalated: {escalated_count} ({escalation_rate:.1f}%)")
    print(f"Not Escalated: {len(non_escalated_records)} ({100 - escalation_rate:.1f}%)")

    print(f"\nEscalated Records: {escalated_records[:10]}...")

    print("\n" + "=" * 80 + "\n")
