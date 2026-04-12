#!/usr/bin/env python3
"""
Compare reward server endpoints under concurrent load.

Test intent:
1. Compare the old reverse-proxy URL and the direct IP URL with the same payload.
2. Separate HTTP transport failures from reward business failures.
3. Estimate whether the proxy layer is the primary bottleneck under concurrent load.

Typical run:
python3 my/compare_reward_endpoints.py \
  --server-url 'https://your-proxy-host/proxy/48001/compute_reward_v2' \
  --server-url 'http://10.244.232.168:48001/compute_reward_v2' \
  --requests 32 \
  --concurrency 8 \
  --request-timeout-sec 1200 \
  --connect-timeout-sec 30 \
  --output-json /tmp/reward_compare.json

Smoke test:
python3 my/compare_reward_endpoints.py \
  --server-url 'https://your-proxy-host/proxy/48001/compute_reward_v2' \
  --server-url 'http://10.244.232.168:48001/compute_reward_v2' \
  --requests 8 \
  --concurrency 2

How to read the output:
1. If the proxy endpoint shows many `request_error` results while the direct IP endpoint mostly returns
   `success` / `judge_fail` / `render_engine_error`, the proxy layer is likely dropping connections.
2. If both endpoints are stable, the main issue is more likely inside the training-side request pattern.
3. If both endpoints show many transport failures, continue checking client connection reuse, concurrency,
   and network limits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


DEFAULT_SAMPLE_FILE = (
    Path(__file__).resolve().parent.parent / "rollouts" / "baseline_Qwen3-4B" / "6.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare proxy vs direct-IP reward endpoints with the same request payload."
    )
    parser.add_argument(
        "--server-url",
        action="append",
        required=True,
        help="Reward endpoint URL. Pass this flag multiple times to compare multiple endpoints.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help=f"JSONL rollout sample used to construct the request. Default: {DEFAULT_SAMPLE_FILE}",
    )
    parser.add_argument(
        "--sample-line",
        type=int,
        default=1,
        help="1-based line number in --sample-file.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=32,
        help="Total requests per endpoint.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum in-flight requests per endpoint.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=1200.0,
        help="Total timeout for one HTTP request.",
    )
    parser.add_argument(
        "--connect-timeout-sec",
        type=float,
        default=30.0,
        help="Connect timeout for one HTTP request.",
    )
    parser.add_argument(
        "--stagger-ms",
        type=float,
        default=0.0,
        help="Delay between request launches in milliseconds.",
    )
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Allow inheriting proxy-related environment variables. Default is disabled.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the final comparison report as JSON.",
    )
    return parser.parse_args()


def load_sample(*, sample_file: Path, sample_line: int) -> dict[str, Any]:
    if sample_line < 1:
        raise ValueError("--sample-line must be >= 1")
    if not sample_file.exists():
        raise FileNotFoundError(f"sample file not found: {sample_file}")

    with sample_file.open(encoding="utf-8") as file:
        for line_idx, line in enumerate(file, start=1):
            if line_idx == sample_line:
                return json.loads(line)

    raise ValueError(f"sample line {sample_line} out of range for {sample_file}")


def extract_query(*, prompt_text: str) -> str:
    match = re.search(r"User request:\n(.+?)\n/no_think", prompt_text, re.S)
    if match:
        return match.group(1).strip()
    raise ValueError("failed to extract query from rollout sample")


def build_payload(*, sample: dict[str, Any], request_index: int) -> dict[str, Any]:
    query = extract_query(prompt_text=str(sample["input"]))
    return {
        "data_source": "tmp",
        "solution_str": sample["output"],
        "ground_truth": sample.get("gts", ""),
        "extra_info": {
            "index": f"compare_{request_index}_{uuid.uuid4().hex}",
            "query": query,
        },
    }


async def send_one_request(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    server_url: str,
    payload: dict[str, Any],
    request_index: int,
) -> dict[str, Any]:
    async with semaphore:
        started_at = time.perf_counter()
        try:
            response = await client.post(server_url, json=payload)
            latency_sec = time.perf_counter() - started_at
            result: dict[str, Any] = {
                "request_index": request_index,
                "ok": True,
                "latency_sec": latency_sec,
                "http_status": response.status_code,
                "overall_status": "<missing>",
                "error_message": "",
            }
            try:
                body = response.json()
            except Exception as error:  # noqa: BLE001
                result["ok"] = False
                result["overall_status"] = "invalid_json"
                result["error_message"] = f"invalid JSON response: {error}"
                return result

            result["overall_status"] = str(body.get("overall_status", "<missing>"))
            result["error_message"] = str(body.get("error_message", ""))
            return result
        except Exception as error:  # noqa: BLE001
            latency_sec = time.perf_counter() - started_at
            return {
                "request_index": request_index,
                "ok": False,
                "latency_sec": latency_sec,
                "http_status": None,
                "overall_status": "request_error",
                "error_message": str(error),
            }


def summarize_results(*, server_url: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    http_status_counts = Counter()
    overall_status_counts = Counter()
    error_message_counts = Counter()
    latencies = []

    for item in results:
        http_status_counts[str(item["http_status"])] += 1
        overall_status_counts[str(item["overall_status"])] += 1
        if item["error_message"]:
            error_message_counts[str(item["error_message"])] += 1
        latencies.append(float(item["latency_sec"]))

    sorted_latencies = sorted(latencies)

    def percentile(*, q: float) -> float:
        if not sorted_latencies:
            return 0.0
        idx = min(len(sorted_latencies) - 1, max(0, int(q * (len(sorted_latencies) - 1))))
        return sorted_latencies[idx]

    return {
        "server_url": server_url,
        "requests": len(results),
        "http_status_counts": dict(http_status_counts),
        "overall_status_counts": dict(overall_status_counts),
        "top_error_messages": error_message_counts.most_common(10),
        "latency_sec": {
            "min": min(sorted_latencies) if sorted_latencies else 0.0,
            "p50": percentile(q=0.50),
            "p90": percentile(q=0.90),
            "p99": percentile(q=0.99),
            "max": max(sorted_latencies) if sorted_latencies else 0.0,
            "mean": statistics.fmean(sorted_latencies) if sorted_latencies else 0.0,
        },
    }


async def run_one_endpoint(
    *,
    server_url: str,
    sample: dict[str, Any],
    requests: int,
    concurrency: int,
    request_timeout_sec: float,
    connect_timeout_sec: float,
    stagger_ms: float,
    trust_env: bool,
) -> dict[str, Any]:
    timeout = httpx.Timeout(timeout=request_timeout_sec, connect=connect_timeout_sec)
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=trust_env,
    ) as client:
        tasks = []
        for request_index in range(requests):
            payload = build_payload(sample=sample, request_index=request_index)
            task = asyncio.create_task(
                send_one_request(
                    client=client,
                    semaphore=semaphore,
                    server_url=server_url,
                    payload=payload,
                    request_index=request_index,
                )
            )
            tasks.append(task)
            if stagger_ms > 0:
                await asyncio.sleep(stagger_ms / 1000.0)
        results = await asyncio.gather(*tasks)
    return summarize_results(server_url=server_url, results=results)


async def amain() -> dict[str, Any]:
    args = parse_args()
    sample = load_sample(sample_file=args.sample_file, sample_line=args.sample_line)
    report = {
        "sample_file": str(args.sample_file),
        "sample_line": args.sample_line,
        "requests_per_endpoint": args.requests,
        "concurrency_per_endpoint": args.concurrency,
        "request_timeout_sec": args.request_timeout_sec,
        "connect_timeout_sec": args.connect_timeout_sec,
        "stagger_ms": args.stagger_ms,
        "trust_env": args.trust_env,
        "endpoints": [],
    }

    for server_url in args.server_url:
        endpoint_report = await run_one_endpoint(
            server_url=server_url,
            sample=sample,
            requests=args.requests,
            concurrency=args.concurrency,
            request_timeout_sec=args.request_timeout_sec,
            connect_timeout_sec=args.connect_timeout_sec,
            stagger_ms=args.stagger_ms,
            trust_env=args.trust_env,
        )
        report["endpoints"].append(endpoint_report)

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return report


def main() -> None:
    report = asyncio.run(amain())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
