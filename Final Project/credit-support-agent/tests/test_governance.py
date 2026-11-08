import pytest
from fastapi.testclient import TestClient
from src.api.server import app, MAX_REQUEST_CHARS
from src.orchestration.crew import retrieval_agent, lookup_agent, composer_agent
from src.tools.crewai_tools import check_application_status, retrieve_banking_policy

client = TestClient(app)

def test_tool_allocation():
    # Verify lookup_agent has the tool
    assert check_application_status in lookup_agent.tools

    # Verify retrieval_agent does NOT have it
    assert check_application_status not in retrieval_agent.tools

    # Verify composer_agent does NOT have it
    assert check_application_status not in composer_agent.tools

def test_request_within_budget():
    query = "What is the status of LOAN_0001?"
    response = client.post("/ask", json={"query": query})
    assert response.status_code != 400

def test_request_exceeds_budget():
    # Construct a query that exceeds the budget
    long_query = "a" * (MAX_REQUEST_CHARS + 1)
    response = client.post("/ask", json={"query": long_query})
    assert response.status_code == 400
    assert response.json()["detail"] == "Token budget exceeded"
