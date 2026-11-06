"""
Loan Application Dataset Generator

Generates a deterministic, seeded dataset of loan applications for RAG evaluation.
All records are reproducible via fixed seed.
"""

import random
from dataclasses import dataclass
from typing import List


@dataclass
class LoanApplication:
    """Loan application record."""
    record_id: str
    category: str
    status: str
    loan_amount_inr: int
    days_since_created: int
    flagged_for_fraud_review: bool


# Categories and statuses from requirements
LOAN_CATEGORIES = [
    "personal_loan",
    "home_loan",
    "auto_loan",
    "business_loan",
    "education_loan",
]

LOAN_STATUSES = [
    "under_review",
    "documents_requested",
    "approved",
    "rejected",
    "disbursed",
]

# Realistic loan amount ranges by category (reasoning: market norms)
LOAN_AMOUNT_RANGES = {
    "personal_loan": (100_000, 1_000_000),      # Small unsecured loans
    "home_loan": (1_000_000, 10_000_000),       # Large secured mortgages
    "auto_loan": (300_000, 2_000_000),          # Vehicle financing
    "business_loan": (500_000, 50_000_000),     # Variable business needs
    "education_loan": (100_000, 2_000_000),     # Education costs
}


def generate_dataset(seed: int = 42, num_records: int = 50) -> List[LoanApplication]:
    """
    Generate deterministic loan application dataset.

    Args:
        seed: Random seed for reproducibility
        num_records: Number of records to generate (default 50, minimum 40)

    Returns:
        List of LoanApplication records
    """
    random.seed(seed)
    records = []

    # Ensure minimum coverage of all categories and statuses
    min_per_category = 3
    min_per_status = 1

    # Generate one record per category to ensure coverage
    category_idx = 0
    status_idx = 0

    for i in range(num_records):
        record_id = f"LOAN_{i+1:04d}"

        # Distribute categories evenly (ensuring ≥3 per category)
        category = LOAN_CATEGORIES[category_idx % len(LOAN_CATEGORIES)]
        category_idx += 1

        # Distribute statuses evenly (ensuring ≥1 per status)
        status = LOAN_STATUSES[status_idx % len(LOAN_STATUSES)]
        status_idx += 1

        # Random loan amount within category range (reasoning: realistic market pricing)
        min_amt, max_amt = LOAN_AMOUNT_RANGES[category]
        loan_amount = random.randint(min_amt, max_amt)

        # Random days since created (0-30 days)
        days_since_created = random.randint(0, 30)

        # Flag for fraud review (will adjust seed to land in 10-30% range)
        # Starting with 20% base rate
        flagged = random.random() < 0.20

        records.append(
            LoanApplication(
                record_id=record_id,
                category=category,
                status=status,
                loan_amount_inr=loan_amount,
                days_since_created=days_since_created,
                flagged_for_fraud_review=flagged,
            )
        )

    return records


def print_dataset_report(records: List[LoanApplication]) -> dict:
    """
    Print and validate dataset report.

    Returns:
        Dictionary with validation metrics
    """
    # Category counts
    category_counts = {}
    for cat in LOAN_CATEGORIES:
        count = sum(1 for r in records if r.category == cat)
        category_counts[cat] = count

    # Status counts
    status_counts = {}
    for status in LOAN_STATUSES:
        count = sum(1 for r in records if r.status == status)
        status_counts[status] = count

    # Fraud flag percentage
    flagged_count = sum(1 for r in records if r.flagged_for_fraud_review)
    fraud_percentage = (flagged_count / len(records)) * 100

    # Print report
    print(f"\n{'='*60}")
    print(f"LOAN APPLICATION DATASET REPORT")
    print(f"{'='*60}")
    print(f"Total records: {len(records)}")
    print(f"\nCategory Distribution:")
    for cat, count in category_counts.items():
        status = "[OK]" if count >= 3 else "[FAIL]"
        print(f"  {status} {cat}: {count} records")

    print(f"\nStatus Distribution:")
    for status, count in status_counts.items():
        print(f"  [OK] {status}: {count} records")

    print(f"\nFraud Review Flags:")
    print(f"  Flagged: {flagged_count} records")
    print(f"  Percentage: {fraud_percentage:.1f}%")
    if 10 <= fraud_percentage <= 30:
        print(f"  Status: [OK] Within 10-30% range")
    else:
        print(f"  Status: [FAIL] Outside 10-30% range (adjust seed)")

    print(f"{'='*60}\n")

    return {
        "total_records": len(records),
        "category_counts": category_counts,
        "status_counts": status_counts,
        "fraud_percentage": fraud_percentage,
        "fraud_in_range": 10 <= fraud_percentage <= 30,
    }


# Generate and store dataset at module load
LOAN_APPLICATIONS = generate_dataset(seed=42, num_records=50)


if __name__ == "__main__":
    # Run report when executed directly
    report = print_dataset_report(LOAN_APPLICATIONS)

    # Print sample records
    print("Sample Records (first 5):")
    print("-" * 60)
    for record in LOAN_APPLICATIONS[:5]:
        print(f"ID: {record.record_id}")
        print(f"  Category: {record.category}")
        print(f"  Status: {record.status}")
        print(f"  Amount: INR {record.loan_amount_inr:,}")
        print(f"  Days Old: {record.days_since_created}")
        print(f"  Fraud Flag: {record.flagged_for_fraud_review}")
        print()
