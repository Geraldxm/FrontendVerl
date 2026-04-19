#!/usr/bin/env python3
"""
Naive reward client V2.

接口要求:
    async def compute_score(*, data_source, solution_str, ground_truth, extra_info=None, **kwargs)

当前职责:
1. 规范化调用方传入的参数
2. 请求 NaiveSingleRewardServerV2 的 /compute_reward_v2 接口
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

DEFAULT_SERVER_URL = "http://127.0.0.1:48001/compute_reward_v2"
SERVER_URL = os.getenv("REWARD_SERVER_URL", DEFAULT_SERVER_URL)
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 1000
DEFAULT_REQUEST_TIMEOUT_SEC = 3000.0
DEFAULT_CONNECT_TIMEOUT_SEC = 3000.0

_ASYNC_CLIENT_CLS = httpx.AsyncClient


def _convert_to_serializable(*, obj: Any) -> Any:
    """
    输入: 任意 Python / numpy 对象。
    输出: 可直接放入 JSON payload 的基础类型。
    边界: 未命中的类型保持原样返回，由上层继续决定是否可接受。
    """
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
    """
    输入: 无。
    输出: 默认的 reward_signals 字典。
    边界: 无。
    """
    return {
        "format_score": 0.0,
        "console_errors": 0.0,
        "network_violations": 0.0,
        "a11y_score": 0.0,
        "element_hit_rate": 0.0,
        "ui_spatial_score": 0.0,
        "Color Scheme": 0.0,
        "Visual Style": 0.0,
        "Imagery and Visual Elements": 0.0,
        "Typography and Font Aesthetics": 0.0,
        "Content and Messaging": 0.0,
    }


def _build_request_error_result(*, error_message: str) -> dict[str, Any]:
    """
    输入: 需要透传到下游的错误信息。
    输出: 统一的客户端失败结果。
    边界: 该返回表示“请求未成功完成或响应不可用”，上层不应把它当作服务端业务状态继续分桶。
    """
    return {
        "score": 0.0,
        "overall_status": "request_error",
        "error_message": error_message,
        "reward_signals": _build_default_reward_signals(),
        "render_info": {},
        "judge_info": {},
    }


def _extract_query_from_origin_info(*, origin_info: Any) -> str:
    """
    输入: extra_info 中的 origin_info，可为 dict、JSON 字符串或其他类型。
    输出: 从 origin_info 中提取出的 query 文本; 提取失败时返回空字符串。
    边界: 仅做兼容性兜底，不对 origin_info 的业务结构做严格校验。
    """
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
    """
    输入: 调用方传入的 extra_info，可为 dict、JSON 字符串或 None。
    输出: (规范化后的字典, 错误信息)。
    边界: 失败时返回 ({}, error_message)，上层应直接返回 request_error，不继续发请求。
    """
    if extra_info is None:
        return {}, ""

    normalized_extra_info = extra_info
    if isinstance(normalized_extra_info, str):
        try:
            normalized_extra_info = json.loads(normalized_extra_info)
        except json.JSONDecodeError as error:
            # 这里返回是因为 extra_info 字符串不是合法 JSON。
            # 该返回表示失败; 上层应直接转成 request_error，不能继续请求服务端。
            return {}, f"extra_info is not valid JSON: {error}"

    if not isinstance(normalized_extra_info, dict):
        # 这里返回是因为 extra_info 既不是 dict，也不是可解析成 dict 的 JSON 字符串。
        # 该返回表示失败; 上层应直接转成 request_error，不能继续请求服务端。
        return {}, (
            "extra_info must be a dict or JSON object string, "
            f"got {type(normalized_extra_info).__name__}"
        )

    normalized_extra_info = _convert_to_serializable(obj=normalized_extra_info)
    if not isinstance(normalized_extra_info, dict):
        # 这里返回是因为序列化后结果不再是 dict，无法满足服务端约定。
        # 该返回表示失败; 上层应直接转成 request_error，不能继续请求服务端。
        return {}, "extra_info must serialize to a JSON object"

    if not normalized_extra_info.get("index"):
        normalized_extra_info["index"] = f"client_{uuid.uuid4().hex}"

    if not normalized_extra_info.get("query") and not normalized_extra_info.get("questions"):
        # 兼容历史 parquet: 有些样本把原始题目包在 origin_info.question 里，
        # 这里优先做本地回填，避免还没发请求就被 client 直接判成 request_error。
        fallback_query = _extract_query_from_origin_info(
            origin_info=normalized_extra_info.get("origin_info")
        )
        if fallback_query:
            normalized_extra_info["query"] = fallback_query

    if not normalized_extra_info.get("query") and not normalized_extra_info.get("questions"):
        # 这里返回是因为 V2 服务要求 extra_info 中至少带 query/questions 之一。
        # 该返回表示失败; 上层应直接转成 request_error，不能继续请求服务端。
        return {}, "query or questions must be provided in extra_info"

    # 这里返回是因为 extra_info 已满足 V2 请求约束。
    # 该返回表示成功; 上层可以继续构造 payload 并请求服务端。
    return normalized_extra_info, ""


def _build_payload(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    """
    输入: 规范化后的 reward 请求字段。
    输出: 可直接发送到 /compute_reward_v2 的请求体。
    边界: 当前不在此处做业务校验，假设 extra_info 已由上游校验完成。
    """
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
    """
    输入: 连接池、超时与可选 transport 配置。
    输出: 配置完成的异步 HTTP client。
    边界: 固定 trust_env=False，确保请求不继承本地代理环境变量。
    """
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
    """
    输入: 服务端返回的 JSON 结果。
    输出: 下游稳定可消费的成功结构。
    边界: 若响应不是 dict，则抛出 ValueError，由上层统一降级为 request_error。
    """
    if not isinstance(result, dict):
        raise ValueError("response JSON must be an object")

    raw_signals = result.get("reward_signals", {})
    normalized_signals = _build_default_reward_signals()
    if isinstance(raw_signals, dict):
        # 仅允许服务端已定义字段覆盖默认值，避免脏字段污染下游。
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
    """
    输入: reward 请求四元组与可选 HTTP 配置。
    输出: V2 统一结果字典，包含 score / overall_status / error_message / reward_signals / render_info / judge_info。
    边界: 服务端成功时原样透传其业务状态; 请求失败时统一返回 overall_status=request_error。
    """
    normalized_extra_info, extra_info_error = _normalize_extra_info(extra_info=extra_info)
    if extra_info_error:
        # 这里返回是因为 extra_info 不满足 V2 最低约束。
        # 该返回表示失败; 下游应将其视为 request_error，不再继续按服务端业务状态处理。
        return _build_request_error_result(error_message=extra_info_error)

    # 允许调用方通过 kwargs 透传当前训练步数，便于服务端做分步诊断或策略分流。
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
        # 这里返回是因为客户端请求失败、HTTP 状态码异常，或服务端响应无法解析。
        # 该返回表示失败; 下游应将其视为 request_error，而不是服务端业务状态。
        return _build_request_error_result(error_message=str(error))


def test_compute_score() -> None:
    """
    输入: 无。
    输出: 打印一次手动联调结果。
    边界: 依赖本地 reward server 已启动; 该函数仅用于人工验证，不替代自动化测试。
    """
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
