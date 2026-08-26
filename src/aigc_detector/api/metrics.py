"""Thread-safe Prometheus text-format metrics (dependency-free).

Hand-formatted exposition — no prometheus-client dependency. All mutations
take a module-wide ``threading.Lock``; detect endpoints call in from the
event loop and (for latency observations) potentially from worker threads,
so lock granularity is per-call.

Exposed series:
    - aigc_requests_total{endpoint,status}   counter
    - aigc_detect_seconds_bucket{endpoint,le} histogram (cumulative buckets)
    - aigc_detect_seconds_sum{endpoint}      histogram sum
    - aigc_detect_seconds_count{endpoint}    histogram count
    - aigc_requests_in_flight                gauge

``endpoint`` labels are short route names: ``detect``, ``detect_file``.
"""

from __future__ import annotations

import threading

LATENCY_BUCKETS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
_INF = float("inf")


def _fmt_le(bound: float) -> str:
    return "+Inf" if bound == _INF else f"{bound:g}"


class MetricsRegistry:
    """In-process metrics store; safe for concurrent use across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_total: dict[tuple[str, int], int] = {}
        self._hist_bucket: dict[tuple[str, float], int] = {}
        self._hist_sum: dict[str, float] = {}
        self._hist_count: dict[str, int] = {}
        self._in_flight: int = 0

    # ------------------------------------------------------------------
    # Mutation API (called from endpoints / exception handlers)
    # ------------------------------------------------------------------

    def inc_request(self, endpoint: str, status: int) -> None:
        """Count one completed request with its final HTTP status."""
        with self._lock:
            self._requests_total[(endpoint, status)] = self._requests_total.get((endpoint, status), 0) + 1

    def observe_seconds(self, endpoint: str, seconds: float) -> None:
        """Record one latency observation into the cumulative histogram."""
        with self._lock:
            for bound in LATENCY_BUCKETS:
                if seconds <= bound:
                    key = (endpoint, bound)
                    self._hist_bucket[key] = self._hist_bucket.get(key, 0) + 1
            inf_key = (endpoint, _INF)
            self._hist_bucket[inf_key] = self._hist_bucket.get(inf_key, 0) + 1
            self._hist_sum[endpoint] = self._hist_sum.get(endpoint, 0.0) + max(seconds, 0.0)
            self._hist_count[endpoint] = self._hist_count.get(endpoint, 0) + 1

    def inc_in_flight(self) -> None:
        with self._lock:
            self._in_flight += 1

    def dec_in_flight(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    # ------------------------------------------------------------------
    # Read API (tests + exposition)
    # ------------------------------------------------------------------

    def request_count(self, endpoint: str, status: int) -> int:
        with self._lock:
            return self._requests_total.get((endpoint, status), 0)

    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def render(self) -> str:
        """Render the registry in Prometheus text exposition format 0.0.4."""
        with self._lock:
            lines: list[str] = []
            lines.append("# HELP aigc_requests_total Total requests processed by endpoint.")
            lines.append("# TYPE aigc_requests_total counter")
            for (endpoint, status), count in sorted(self._requests_total.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                lines.append(f'aigc_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}')

            lines.append("# HELP aigc_detect_seconds Detection endpoint latency in seconds.")
            lines.append("# TYPE aigc_detect_seconds histogram")
            endpoints = sorted(set(self._hist_count) | {ep for ep, _ in self._hist_bucket})
            for endpoint in endpoints:
                for bound in LATENCY_BUCKETS:
                    count = self._hist_bucket.get((endpoint, bound), 0)
                    lines.append(f'aigc_detect_seconds_bucket{{endpoint="{endpoint}",le="{_fmt_le(bound)}"}} {count}')
                inf_count = self._hist_bucket.get((endpoint, _INF), 0)
                lines.append(f'aigc_detect_seconds_bucket{{endpoint="{endpoint}",le="+Inf"}} {inf_count}')
                hist_sum = self._hist_sum.get(endpoint, 0.0)
                lines.append(f'aigc_detect_seconds_sum{{endpoint="{endpoint}"}} {hist_sum:.6f}')
                hist_count = self._hist_count.get(endpoint, 0)
                lines.append(f'aigc_detect_seconds_count{{endpoint="{endpoint}"}} {hist_count}')

            lines.append("# HELP aigc_requests_in_flight Requests currently being processed.")
            lines.append("# TYPE aigc_requests_in_flight gauge")
            lines.append(f"aigc_requests_in_flight {self._in_flight}")
            return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()
