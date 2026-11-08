import pytest
import asyncio
from src.governance.models import Verdict
from src.governance.review import run_review

@pytest.mark.asyncio
async def test_review_approval():
    draft = "The prepayment penalty is 2%."
    context = "Prepayment penalty is 2%."

    verdict = run_review(draft, context)

    assert verdict.approved is True
    assert verdict.final_answer == draft

@pytest.mark.asyncio
async def test_review_revision():
    # Inject hallucination with a keyword our mock client detects
    draft = "The interest rate is 0% (hallucination_test)."
    context = "Interest rates are governed by standard slabs."

    verdict = run_review(draft, context)

    assert verdict.approved is False
    assert "hallucination" in verdict.reason.lower()
    assert verdict.final_answer != draft
    assert "not 0%" in verdict.final_answer
