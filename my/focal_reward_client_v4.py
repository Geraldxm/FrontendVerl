#!/usr/bin/env python3
"""
Naive reward client V4.

版本定位（快速区分）:
- 对接 `NaiveSingleRewardServerV4`（`/compute_reward_v4`）。
- render-only：只返回 5 个一层信号，不再依赖 judge/VLM 打分维度。
- `extra_info` 本地不再强制要求 `query/questions`，仅保留 `index` 自动补齐与 `step/global_steps` 注入。

接口要求:
    async def compute_score(*, data_source, solution_str, ground_truth, extra_info=None, **kwargs)

当前职责:
1. 规范化调用方传入的参数
2. 请求 NaiveSingleRewardServerV4 的 /compute_reward_v4 接口
3. 原样透传服务端字段; 请求异常时返回统一降级结构
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

DEFAULT_SERVER_URL = "http://127.0.0.1:48003/compute_reward_v4"
SERVER_URL = os.getenv("REWARD_SERVER_URL", DEFAULT_SERVER_URL)
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 1000
DEFAULT_REQUEST_TIMEOUT_SEC = 3000.0
DEFAULT_CONNECT_TIMEOUT_SEC = 3000.0
DEFAULT_EXPERIMENT_NAME = "none"

_ASYNC_CLIENT_CLS = httpx.AsyncClient


def _read_experiment_name_from_env() -> str:
    for env_name in ("EXPERIMENT_NAME", "EXPERIENT_NAME", "TRAINER_EXPERIMENT_NAME"):
        env_value = os.getenv(env_name)
        if env_value is not None:
            normalized_env_value = env_value.strip()
            if normalized_env_value:
                return normalized_env_value
    return DEFAULT_EXPERIMENT_NAME


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
    }


def _build_request_error_result(*, error_message: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "overall_status": "request_error",
        "error_message": error_message,
        "reward_signals": _build_default_reward_signals(),
        "render_info": {},
        "judge_info": {},
    }


def _normalize_extra_info(*, extra_info: Any) -> tuple[dict[str, Any], str]:
    normalized_extra_info = extra_info
    if normalized_extra_info is None:
        normalized_extra_info = {}
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

    if not normalized_extra_info.get("experiment_name"):
        normalized_extra_info["experiment_name"] = _read_experiment_name_from_env()

    return normalized_extra_info, ""


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
