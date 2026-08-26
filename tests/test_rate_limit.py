"""Tests for the slowapi rate limit (10/minute per IP) on POST /api/v1/detect.

Approach — integration-level via TestClient (chosen deliberately):
    The limiter (src/aigc_detector/api/middleware.py) is a module-level
    ``Limiter(key_func=get_remote_address)`` with in-memory fixed-window
    storage, and the route is decorated ``@limiter.limit("10/minute")``.
    TestClient pins every request's client address to ``"testclient"``,
    which is exactly what ``get_remote_address`` keys on — so 12 rapid
    valid POSTs from the single test client deterministically exhaust the
    10/minute window and request #11 must return 429. No header tricks or
    X-Forwarded-For spoofing are needed (the key func ignores them).

Quota isolation:
    The limiter is a process-wide singleton shared with tests/test_api.py
    (which builds its own apps but registers the same limiter). The
    conftest ``client`` fixture calls ``limiter.reset()`` before AND after
    every test using it, so quota consumed here can never leak into other
    test modules — and vice versa — regardless of execution order.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

MINUTE_LIMIT = 10
VALID_BODY = {"text": "a" * 100}  # passes the 50-char minimum validation


class TestRateLimit:
    def test_429_after_limit_exhausted(self, client: TestClient):
        statuses = [
            client.post("/api/v1/detect", json=VALID_BODY).status_code
            for _ in range(MINUTE_LIMIT + 2)
        ]
        # First 10 requests inside the fixed window succeed (stub pipeline)
        assert statuses[:MINUTE_LIMIT] == [200] * MINUTE_LIMIT
        # 11th and 12th are rejected by the limiter
        assert statuses[MINUTE_LIMIT] == 429
        assert statuses[MINUTE_LIMIT + 1] == 429

    def test_429_response_payload(self, client: TestClient):
        for _ in range(MINUTE_LIMIT):
            assert client.post("/api/v1/detect", json=VALID_BODY).status_code == 200
        blocked = client.post("/api/v1/detect", json=VALID_BODY)
        assert blocked.status_code == 429
        # Custom handler from middleware.py: {"detail": "Rate limit exceeded..."}
        assert "Rate limit exceeded" in blocked.json()["detail"]

    def test_file_endpoint_independent_bucket(self, client: TestClient):
        """/detect and /detect/file have separate per-endpoint buckets."""
        # Exhaust the /detect bucket entirely
        for _ in range(MINUTE_LIMIT + 1):
            client.post("/api/v1/detect", json=VALID_BODY)
        # /detect/file still allowed (own decorator scope, same IP)
        resp = client.post(
            "/api/v1/detect/file?include_segments=false",
            files={"file": ("story.txt", ("word " * 40).encode(), "text/plain")},
        )
        assert resp.status_code == 200
