"""Shared helpers for stress tests."""

import asyncio
import os

import aiohttp
import pytest

from _harness_import import _harness_mod as H  # noqa: F401


def stress_duration() -> float:
    return float(os.environ.get("STRESS_DURATION", "60"))


def run_stress(server_url: str, endpoint_key: str, duration: float | None = None,
               concurrency: int | None = None) -> dict:
    """endpoint_key is matched against either Endpoint.path or a suffix of .name."""
    ep = next(
        (e for e in H.ENDPOINTS if e.path == endpoint_key or e.name.endswith(endpoint_key)),
        None,
    )
    if ep is None:
        pytest.fail(f"unknown endpoint: {endpoint_key}")

    async def go():
        async with aiohttp.ClientSession() as s:
            return await H.measure_stress(
                s, server_url, ep,
                duration_s=duration or stress_duration(),
                concurrency=concurrency,
            )
    return asyncio.run(go())


def assert_healthy(result: dict, max_error_rate: float = 0.0):
    assert result["total"] > 0, f"zero requests: {result}"
    assert result["error_rate"] <= max_error_rate, (
        f"{result['endpoint']}: error_rate={result['error_rate']*100:.2f}% "
        f"(ceiling {max_error_rate*100:.2f}%)"
    )
    assert result["errors"] == 0 or max_error_rate > 0, (
        f"{result['endpoint']}: {result['errors']} errors in {result['total']} reqs"
    )
