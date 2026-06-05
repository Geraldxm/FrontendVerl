#!/usr/bin/env python3
"""
Naive reward client V6.

版本定位（快速区分）:
- 对接 `NaiveSingleRewardServerV6`（`/compute_reward_v6`）。
- 信号集合沿用 V5：5 个 render 一层信号 + 3 个 VLM 静态视觉信号（共 8 维）。
- 原样透传 `render_info.debug_info`，便于 VERL 主框架记录 render latency 与 timeout stage。
- 不引入长度惩罚；长度与 Tailwind class 统计仅作为服务端 debug 信息回传。

接口要求:
    async def compute_score(*, data_source, solution_str, ground_truth, extra_info=None, **kwargs)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np

DEFAULT_SERVER_URL = "http://127.0.0.1:48005/compute_reward_v6"
SERVER_URL = os.getenv("REWARD_SERVER_URL", DEFAULT_SERVER_URL)
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 1000
DEFAULT_REQUEST_TIMEOUT_SEC = 3000.0
DEFAULT_CONNECT_TIMEOUT_SEC = 3000.0
DEFAULT_VISUAL_JUDGE_MODE = "joint"
VALID_VISUAL_JUDGE_MODES = {"joint", "separate"}

_ASYNC_CLIENT_CLS = httpx.AsyncClient


def _convert_to_serializable(*, obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {
            key: _convert_to_serializable(obj=value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_convert_to_serializable(obj=value) for value in obj]
    return obj


def _build_default_reward_signals() -> dict[str, Any]:
    return {
        "format_score": 0.0,
        "console_errors": 0.0,
        "network_violations": 0.0,
        "a11y_score": 0.0,
        "element_hit_rate": 0.0,
        "instructional_alignment": 0.0,
        "visual_elements": 0.0,
        "layout_and_cohesion": 0.0,
    }


def _build_request_error_result(*, error_message: str) -> dict[str, Any]:
    return {
        "task_id": "",
        "score": 0.0,
        "overall_status": "request_error",
        "error_message": error_message,
        "reward_signals": _build_default_reward_signals(),
        "render_info": {},
        "judge_info": {},
    }


def _extract_query_from_origin_info(*, origin_info: Any) -> str:
    parsed_origin_info = origin_info
    if isinstance(parsed_origin_info, str):
        try:
            parsed_origin_info = json.loads(parsed_origin_info)
        except json.JSONDecodeError:
            return ""

    if not isinstance(parsed_origin_info, dict):
        return ""

    for key in ("query", "question", "questions"):
        value = parsed_origin_info.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def _normalize_extra_info(*, extra_info: Any) -> tuple[dict[str, Any], str]:
    if extra_info is None:
        return {}, ""

    normalized_extra_info = extra_info
    if isinstance(normalized_extra_info, str):
        try:
            normalized_extra_info = json.loads(normalized_extra_info)
        except json.JSONDecodeError as error:
            return {}, f"extra_info is not valid JSON: {error}"

    if not isinstance(normalized_extra_info, dict):
        return {}, (
            "extra_info must be a dict or JSON object string, "
            f"got {type(normalized_extra_info).__name__}"
        )

    normalized_extra_info = _convert_to_serializable(obj=normalized_extra_info)
    if not isinstance(normalized_extra_info, dict):
        return {}, "extra_info must serialize to a JSON object"

    if not normalized_extra_info.get("index"):
        normalized_extra_info["index"] = f"client_{uuid.uuid4().hex}"

    if not normalized_extra_info.get("query") and not normalized_extra_info.get("questions"):
        fallback_query = _extract_query_from_origin_info(
            origin_info=normalized_extra_info.get("origin_info")
        )
        if fallback_query:
            normalized_extra_info["query"] = fallback_query

    if not normalized_extra_info.get("query") and not normalized_extra_info.get("questions"):
        return {}, "query or questions must be provided in extra_info"

    return normalized_extra_info, ""


def _normalize_visual_judge_mode(*, value: Any) -> tuple[str, str]:
    if value is None:
        return DEFAULT_VISUAL_JUDGE_MODE, ""

    mode = str(value).strip().lower()
    if mode not in VALID_VISUAL_JUDGE_MODES:
        return "", (
            "visual_judge_mode must be one of: "
            + ", ".join(sorted(VALID_VISUAL_JUDGE_MODES))
        )
    return mode, ""


def _build_payload(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "data_source": _convert_to_serializable(obj=data_source),
        "solution_str": _convert_to_serializable(obj=solution_str),
        "ground_truth": _convert_to_serializable(obj=ground_truth),
        "extra_info": extra_info,
    }


def _build_http_client(
    *,
    max_connections: int,
    max_keepalive_connections: int,
    request_timeout_sec: float,
    connect_timeout_sec: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )
    timeout = httpx.Timeout(
        timeout=request_timeout_sec,
        connect=connect_timeout_sec,
    )
    client_kwargs: dict[str, Any] = {
        "limits": limits,
        "timeout": timeout,
        "trust_env": False,
    }
    if transport is not None:
        client_kwargs["transport"] = transport
    return _ASYNC_CLIENT_CLS(**client_kwargs)


def _normalize_success_result(*, result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("response JSON must be an object")

    raw_signals = result.get("reward_signals", {})
    normalized_signals = _build_default_reward_signals()
    if isinstance(raw_signals, dict):
        for key in normalized_signals:
            if key in raw_signals:
                normalized_signals[key] = raw_signals[key]

    return {
        "task_id": result.get("task_id", ""),
        "score": float(result.get("score", 0.0)),
        "overall_status": result.get("overall_status", "unknown"),
        "error_message": result.get("error_message", "unknown"),
        "reward_signals": normalized_signals,
        "render_info": result.get("render_info", {}),
        "judge_info": result.get("judge_info", {}),
    }


async def compute_score(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    normalized_extra_info, extra_info_error = _normalize_extra_info(extra_info=extra_info)
    if extra_info_error:
        return _build_request_error_result(error_message=extra_info_error)

    mode_source = kwargs.get(
        "visual_judge_mode",
        normalized_extra_info.get("visual_judge_mode", DEFAULT_VISUAL_JUDGE_MODE),
    )
    visual_judge_mode, mode_error = _normalize_visual_judge_mode(value=mode_source)
    if mode_error:
        return _build_request_error_result(error_message=mode_error)
    normalized_extra_info["visual_judge_mode"] = visual_judge_mode

    step_value = kwargs.get("step", kwargs.get("global_steps"))
    if step_value is not None and "step" not in normalized_extra_info:
        normalized_extra_info["step"] = _convert_to_serializable(obj=step_value)

    payload = _build_payload(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=normalized_extra_info,
    )

    server_url = str(kwargs.get("server_url", SERVER_URL))
    max_connections = int(kwargs.get("max_connections", DEFAULT_MAX_CONNECTIONS))
    max_keepalive_connections = int(
        kwargs.get(
            "max_keepalive_connections",
            DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        )
    )
    request_timeout_sec = float(
        kwargs.get("request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC)
    )
    connect_timeout_sec = float(
        kwargs.get("connect_timeout_sec", DEFAULT_CONNECT_TIMEOUT_SEC)
    )
    transport = kwargs.get("transport")

    try:
        async with _build_http_client(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            request_timeout_sec=request_timeout_sec,
            connect_timeout_sec=connect_timeout_sec,
            transport=transport,
        ) as client:
            response = await client.post(server_url, json=payload)
            response.raise_for_status()
            result = response.json()
        return _normalize_success_result(result=result)
    except Exception as error:
        return _build_request_error_result(error_message=str(error))


def test_compute_score() -> None:
    sample_path = Path(__file__).resolve().parent.parent / "test" / "llm_response_2.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))

    async def run_test() -> None:
        result = await compute_score(
            data_source="tmp",
            solution_str=sample["llm_response"],
            ground_truth="",
            extra_info={
                "index": f"manual_{uuid.uuid4().hex}",
                "query": sample["query"],
            },
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(run_test())


if __name__ == "__main__":
    test_compute_score()
