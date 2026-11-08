import time
import uuid
import json
import os
import logging
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from src.guardrails.pipeline_guardrails import mask_pii_for_logging

# Ensure log directory exists
LOG_FILE = "logs/app.jsonl"
os.makedirs("logs", exist_ok=True)

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Read request body
        body_bytes = await request.body()

        # Re-set request body for downstream processing
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive

        # Process request
        response = await call_next(request)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Sanitize payload
        try:
            payload_str = body_bytes.decode("utf-8")
            sanitized_payload, pii_detected = mask_pii_for_logging(payload_str)
        except Exception:
            sanitized_payload = "[UNPARSEABLE_PAYLOAD]"
            pii_detected = False

        # Structure log entry
        log_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "endpoint": f"{request.method} {request.url.path}",
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "sanitized_payload": sanitized_payload,
            "session_id": request.headers.get("X-Session-ID", "unknown"),
            "guardrail_flags": {
                "pii_detected": pii_detected
            }
        }

        # Append to JSONL
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return response
