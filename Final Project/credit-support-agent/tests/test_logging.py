import sys
sys.path.insert(0, '.')

import os
import json
import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from src.observability.logging import StructuredLoggingMiddleware
from src.guardrails.pipeline_guardrails import mask_pii_for_logging

LOG_FILE = "logs/app.jsonl"

@pytest.fixture(autouse=True)
def clear_logs():
    """Clear logs before each test."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

def test_every_line_is_valid_json():
    """Every line in logs/app.jsonl parses as JSON."""
    # Ensure middleware is added
    middleware_classes = [m.cls for m in app.user_middleware]
    assert StructuredLoggingMiddleware in middleware_classes

    client = TestClient(app)
    client.post("/ask", json={"query": "What are prepayment penalties?"})

    assert os.path.exists(LOG_FILE), "Log file should exist after request"

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    assert len(lines) > 0, "At least one log entry expected"

    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in log: {e}")

def test_pii_masked_in_log():
    """Raw PAN value must be absent from log."""
    query = "My PAN is ABCDE1234F, check status of LOAN_0011"
    sanitized, pii = mask_pii_for_logging(query)

    assert "[PAN_REDACTED]" in sanitized
    assert "ABCDE1234F" not in sanitized

    client = TestClient(app)
    client.post("/ask", json={"query": query})

    with open(LOG_FILE, "r") as f:
        content = f.read()

    assert "[PAN_REDACTED]" in content
    assert "ABCDE1234F" not in content

def test_trace_and_latency():
    """trace_id and latency_ms must be present in every log record."""
    client = TestClient(app)
    client.get("/health")

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    assert len(lines) > 0, "Log records must exist"

    for line in lines:
        record = json.loads(line)
        assert "trace_id" in record
        assert record["trace_id"] != ""
        assert "latency_ms" in record
        assert isinstance(record["latency_ms"], (int, float))
        assert 0 <= record["latency_ms"] <= 5000
