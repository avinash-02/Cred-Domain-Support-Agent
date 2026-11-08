import pytest
import time
from src.cache.response_cache import QueryCache, normalize_query
from src.orchestration.structured_output import AgentResponse

# Mock AgentResponse for testing
mock_response = AgentResponse(
    query="What is the loan policy?",
    intent="policy_inquiry",
    answer="Mocked Answer",
    confidence_score=1.0,
    escalation_required=False,
    escalation_reason=None,
    sources=["mock_source"]
)

def test_query_normalization():
    # Identical normalized keys
    key1 = normalize_query("What is the loan policy?")
    key2 = normalize_query("what is the loan policy?   ")
    assert key1 == key2
    assert key1 == "what is the loan policy"

def test_cache_hit_avoids_llm_call():
    cache = QueryCache()

    # Simulate an external "generation" function that is slow
    def slow_generation():
        time.sleep(0.1) # Simulate latency
        return mock_response

    query = "What is the loan policy?"

    # 1. First call: Cache miss
    start_time = time.time()
    response1 = cache.get(query)
    if response1 is None:
        response1 = slow_generation()
        cache.set(query, response1)
    duration1 = time.time() - start_time

    # 2. Second call: Cache hit
    start_time = time.time()
    response2 = cache.get(query)
    duration2 = time.time() - start_time

    print(f"\n--- Cache Evidence ---")
    print(f"Call 1 (Miss) duration: {duration1:.4f}s")
    print(f"Call 2 (Hit) duration: {duration2:.4f}s")
    print(f"Stats: Hits={cache.hits}, Misses={cache.misses}")

    assert response1 == response2
    assert cache.hits == 1
    assert cache.misses == 1
    assert duration2 < duration1 # Hit should be much faster

    cache.clear()
