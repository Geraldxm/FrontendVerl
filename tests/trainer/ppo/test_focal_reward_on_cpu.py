# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for focal reward postprocessing.

These tests focus on the reward aggregation layer only. They construct small
synthetic `DataProto` batches and verify:

1. valid/invalid sample classification from `overall_status`;
2. direct vs focal aggregation outputs;
3. invalid-group compensation behavior;
4. validation-safe scalar field generation.
"""

import unittest

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.focal_reward import postprocess_reward
from verl.trainer.ppo.metric_utils import process_validation_metrics


def _build_reward_signals(
    *,
    format_score: float = 0.0,
    console_errors: float = 0.0,
    network_violations: float = 0.0,
    a11y_score: float = 0.0,
    element_hit_rate: float = 0.0,
    ui_spatial_score: float = 0.0,
    color_scheme: float = 0.0,
    visual_style: float = 0.0,
    imagery_and_visual_elements: float = 0.0,
    typography_and_font_aesthetics: float = 0.0,
    content_and_messaging: float = 0.0,
) -> dict[str, float]:
    """
    输入: 11 个 reward signal 的显式测试值。
    输出: 与 reward client 真实返回格式一致的 `reward_signals` 字典。
    意图: 让测试数据构造和线上字段顺序保持一致。
    """
    return {
        "format_score": format_score,
        "console_errors": console_errors,
        "network_violations": network_violations,
        "a11y_score": a11y_score,
        "element_hit_rate": element_hit_rate,
        "ui_spatial_score": ui_spatial_score,
        "Color Scheme": color_scheme,
        "Visual Style": visual_style,
        "Imagery and Visual Elements": imagery_and_visual_elements,
        "Typography and Font Aesthetics": typography_and_font_aesthetics,
        "Content and Messaging": content_and_messaging,
    }


def _build_batch(uids: list[str]) -> DataProto:
    """
    输入: 每条样本所属的 rollout group uid 列表。
    输出: 最小可运行的 `DataProto` 测试 batch。
    意图: 只保留 reward postprocess 所需字段，降低单测噪声。
    """
    batch_size = len(uids)
    prompts = torch.ones((batch_size, 2), dtype=torch.long)
    responses = torch.ones((batch_size, 3), dtype=torch.long)
    response_mask = torch.ones((batch_size, 3), dtype=torch.long)
    attention_mask = torch.ones((batch_size, 5), dtype=torch.long)
    tensor_batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": attention_mask,
        },
        batch_size=[batch_size],
    )
    return DataProto(
        batch=tensor_batch,
        non_tensor_batch={"uid": np.asarray(uids, dtype=object)},
    )


def _build_reward_extra_infos(
    *,
    overall_statuses: list[str],
    reward_signals: list[dict[str, float]],
    render_statuses: list[str] | None = None,
    judge_statuses: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    输入: 测试用状态列表、reward_signals，以及可选 render/judge 状态。
    输出: 模拟 `extract_reward` 后 reward extra info 的 numpy 字典。
    意图: 复现 reward loop 到 trainer 之间的真实字段形状与类型。
    """
    batch_size = len(overall_statuses)
    if render_statuses is None:
        render_statuses = ["success"] * batch_size
    if judge_statuses is None:
        judge_statuses = ["success"] * batch_size
    return {
        "overall_status": np.asarray(overall_statuses, dtype=object),
        "error_message": np.asarray([""] * batch_size, dtype=object),
        "reward_signals": np.asarray(reward_signals, dtype=object),
        "render_info": np.asarray([{"status": status} for status in render_statuses], dtype=object),
        "judge_info": np.asarray([{"status": status} for status in judge_statuses], dtype=object),
    }


class TestFocalRewardPostprocess(unittest.TestCase):
    def test_llm_generation_error_is_treated_as_valid_sample(self):
        batch = _build_batch(["group_a", "group_a", "group_b", "group_b"])
        raw_reward_tensor = torch.zeros((4, 3), dtype=torch.float32)
        reward_extra_infos = _build_reward_extra_infos(
            overall_statuses=["success", "llm_generation_error", "success", "judge_fail"],
            reward_signals=[
                _build_reward_signals(
                    format_score=10.0,
                    console_errors=10.0,
                    network_violations=10.0,
                    a11y_score=10.0,
                    element_hit_rate=10.0,
                    ui_spatial_score=10.0,
                    color_scheme=10.0,
                    visual_style=10.0,
                    imagery_and_visual_elements=10.0,
                    typography_and_font_aesthetics=10.0,
                    content_and_messaging=10.0,
                ),
                _build_reward_signals(
                    format_score=8.0,
                    console_errors=7.0,
                    network_violations=6.0,
                    a11y_score=5.0,
                    element_hit_rate=4.0,
                ),
                _build_reward_signals(
                    format_score=6.0,
                    console_errors=6.0,
                    network_violations=6.0,
                    a11y_score=6.0,
                    element_hit_rate=6.0,
                    ui_spatial_score=6.0,
                    color_scheme=6.0,
                    visual_style=6.0,
                    imagery_and_visual_elements=6.0,
                    typography_and_font_aesthetics=6.0,
                    content_and_messaging=6.0,
                ),
                _build_reward_signals(
                    format_score=1.0,
                    console_errors=2.0,
                    network_violations=3.0,
                    a11y_score=4.0,
                    element_hit_rate=5.0,
                ),
            ],
            judge_statuses=["success", "skipped", "success", "fail"],
        )

        result = postprocess_reward(
            batch=batch,
            raw_reward_tensor=raw_reward_tensor,
            reward_extra_infos_dict=reward_extra_infos,
            use_focal=False,
            focal_config={"gamma": 3.0, "temperature": 10.0, "epsilon": 0.05, "base_weights": []},
        )

        expected_direct_scores = np.asarray([10.0, 30.0 / 11.0, 6.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(result.derived_extra_info["reward_final_score"], expected_direct_scores, atol=1e-6)
        np.testing.assert_array_equal(result.derived_extra_info["reward_valid_sample"], np.asarray([1, 1, 1, 0]))
        np.testing.assert_array_equal(result.derived_extra_info["reward_is_llm_generation_error"], np.asarray([0, 1, 0, 0]))
        self.assertAlmostEqual(result.derived_extra_info["reward_signal_color_scheme"][1], 0.0)
        self.assertAlmostEqual(
            result.metrics["reward_status/overall_status/llm_generation_error/rate"], 0.25, places=6
        )
        self.assertAlmostEqual(
            result.metrics["reward_signals/color_scheme/mean"], (10.0 + 0.0 + 6.0) / 3.0, places=6
        )
        self.assertIn("reward_weights/color_scheme/mean", result.metrics)
        self.assertIn("reward_others/max/color_scheme", result.metrics)
        self.assertIn("reward_others/min/color_scheme", result.metrics)
        self.assertIn("reward_others/var/color_scheme", result.metrics)
        self.assertIn("reward_signals/color_scheme/focal_weight_mean", result.metrics)
        self.assertNotIn("reward_signals/color_scheme/max", result.metrics)
        self.assertNotIn("reward_signals/color_scheme/min", result.metrics)
        self.assertNotIn("reward_signals/color_scheme/var", result.metrics)
        np.testing.assert_allclose(
            result.reward_tensor.sum(-1).cpu().numpy(),
            expected_direct_scores,
            atol=1e-6,
        )

    def test_invalid_group_receives_batch_level_mean_compensation(self):
        batch = _build_batch(["group_a", "group_a", "group_b", "group_b"])
        raw_reward_tensor = torch.zeros((4, 3), dtype=torch.float32)
        reward_extra_infos = _build_reward_extra_infos(
            overall_statuses=["render_engine_error", "parse_fail", "success", "success"],
            reward_signals=[
                _build_reward_signals(format_score=1.0),
                _build_reward_signals(format_score=2.0),
                _build_reward_signals(
                    format_score=4.0,
                    console_errors=4.0,
                    network_violations=4.0,
                    a11y_score=4.0,
                    element_hit_rate=4.0,
                    ui_spatial_score=4.0,
                    color_scheme=4.0,
                    visual_style=4.0,
                    imagery_and_visual_elements=4.0,
                    typography_and_font_aesthetics=4.0,
                    content_and_messaging=4.0,
                ),
                _build_reward_signals(
                    format_score=8.0,
                    console_errors=8.0,
                    network_violations=8.0,
                    a11y_score=8.0,
                    element_hit_rate=8.0,
                    ui_spatial_score=8.0,
                    color_scheme=8.0,
                    visual_style=8.0,
                    imagery_and_visual_elements=8.0,
                    typography_and_font_aesthetics=8.0,
                    content_and_messaging=8.0,
                ),
            ],
            render_statuses=["fail", "success", "success", "success"],
            judge_statuses=["skipped", "fail", "success", "success"],
        )

        result = postprocess_reward(
            batch=batch,
            raw_reward_tensor=raw_reward_tensor,
            reward_extra_infos_dict=reward_extra_infos,
            use_focal=False,
            focal_config={"base_weights": []},
        )

        np.testing.assert_allclose(
            result.derived_extra_info["reward_final_score"],
            np.asarray([6.0, 6.0, 4.0, 8.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_postprocess_reward_raises_when_no_valid_group_exists(self):
        batch = _build_batch(["group_a", "group_a"])
        raw_reward_tensor = torch.zeros((2, 3), dtype=torch.float32)
        reward_extra_infos = _build_reward_extra_infos(
            overall_statuses=["render_engine_error", "request_error"],
            reward_signals=[_build_reward_signals(), _build_reward_signals()],
            render_statuses=["fail", "fail"],
            judge_statuses=["skipped", "skipped"],
        )

        with self.assertRaisesRegex(ValueError, "No valid focal reward groups found"):
            postprocess_reward(
                batch=batch,
                raw_reward_tensor=raw_reward_tensor,
                reward_extra_infos_dict=reward_extra_infos,
                use_focal=False,
                focal_config={"base_weights": []},
            )

    def test_postprocess_reward_requires_reward_signals(self):
        batch = _build_batch(["group_a"])
        raw_reward_tensor = torch.zeros((1, 3), dtype=torch.float32)
        reward_extra_infos = {
            "overall_status": np.asarray(["success"], dtype=object),
            "error_message": np.asarray([""], dtype=object),
        }

        with self.assertRaisesRegex(ValueError, "reward_signals is required"):
            postprocess_reward(
                batch=batch,
                raw_reward_tensor=raw_reward_tensor,
                reward_extra_infos_dict=reward_extra_infos,
                use_focal=False,
                focal_config={"base_weights": []},
            )

    def test_use_focal_switches_final_score(self):
        batch = _build_batch(["group_a", "group_a"])
        raw_reward_tensor = torch.zeros((2, 3), dtype=torch.float32)
        reward_extra_infos = _build_reward_extra_infos(
            overall_statuses=["success", "success"],
            reward_signals=[
                _build_reward_signals(
                    format_score=10.0,
                    console_errors=10.0,
                    network_violations=10.0,
                    a11y_score=10.0,
                    element_hit_rate=10.0,
                    ui_spatial_score=1.0,
                    color_scheme=1.0,
                    visual_style=1.0,
                    imagery_and_visual_elements=1.0,
                    typography_and_font_aesthetics=1.0,
                    content_and_messaging=1.0,
                ),
                _build_reward_signals(
                    format_score=1.0,
                    console_errors=1.0,
                    network_violations=1.0,
                    a11y_score=1.0,
                    element_hit_rate=1.0,
                    ui_spatial_score=10.0,
                    color_scheme=10.0,
                    visual_style=10.0,
                    imagery_and_visual_elements=10.0,
                    typography_and_font_aesthetics=10.0,
                    content_and_messaging=10.0,
                ),
            ],
        )

        result = postprocess_reward(
            batch=batch,
            raw_reward_tensor=raw_reward_tensor,
            reward_extra_infos_dict=reward_extra_infos,
            use_focal=True,
            focal_config={"gamma": 3.0, "temperature": 10.0, "epsilon": 0.05, "base_weights": []},
        )

        direct_scores = result.derived_extra_info["reward_direct_score"]
        focal_scores = result.derived_extra_info["reward_focal_score"]
        final_scores = result.derived_extra_info["reward_final_score"]

        np.testing.assert_allclose(final_scores, focal_scores, atol=1e-6)
        self.assertFalse(np.allclose(direct_scores, focal_scores))

    def test_postprocess_reward_validates_base_weight_length(self):
        batch = _build_batch(["group_a"])
        raw_reward_tensor = torch.zeros((1, 3), dtype=torch.float32)
        reward_extra_infos = _build_reward_extra_infos(
            overall_statuses=["success"],
            reward_signals=[_build_reward_signals(format_score=1.0)],
        )

        with self.assertRaisesRegex(ValueError, "base_weights length mismatch"):
            postprocess_reward(
                batch=batch,
                raw_reward_tensor=raw_reward_tensor,
                reward_extra_infos_dict=reward_extra_infos,
                use_focal=False,
                focal_config={"base_weights": [1.0, 1.0]},
            )

    def test_validation_extra_info_excludes_raw_nested_fields(self):
        batch = _build_batch(["group_a", "group_a", "group_b", "group_b"])
        raw_reward_tensor = torch.zeros((4, 3), dtype=torch.float32)
        reward_extra_infos = _build_reward_extra_infos(
            overall_statuses=["success", "success", "success", "llm_generation_error"],
            reward_signals=[
                _build_reward_signals(format_score=2.0, color_scheme=1.0),
                _build_reward_signals(format_score=4.0, color_scheme=3.0),
                _build_reward_signals(format_score=6.0, color_scheme=5.0),
                _build_reward_signals(format_score=8.0, color_scheme=0.0),
            ],
            judge_statuses=["success", "success", "success", "skipped"],
        )

        result = postprocess_reward(
            batch=batch,
            raw_reward_tensor=raw_reward_tensor,
            reward_extra_infos_dict=reward_extra_infos,
            use_focal=False,
            focal_config={"base_weights": []},
        )

        self.assertNotIn("reward_signals", result.validation_extra_info)
        self.assertNotIn("render_info", result.validation_extra_info)
        self.assertIn("reward_signals", result.dump_extra_info)
        self.assertIn("reward_weights", result.dump_extra_info)
        self.assertIn("sample_reward_signal", result.dump_extra_info)
        self.assertIn("sample_reward_weight", result.dump_extra_info)
        self.assertIn("group_mean_reward_signal", result.dump_extra_info)
        self.assertIn("group_mean_reward_weight", result.dump_extra_info)
        self.assertIn("reward_others", result.dump_extra_info)
        self.assertIn("render_info", result.dump_extra_info)
        self.assertEqual(result.dump_extra_info["sample_reward_signal"][0]["format_score"], 2.0)
        self.assertEqual(result.dump_extra_info["group_mean_reward_signal"][0]["format_score"], 3.0)
        self.assertEqual(result.dump_extra_info["sample_reward_signal"][1]["format_score"], 4.0)
        self.assertEqual(result.dump_extra_info["group_mean_reward_signal"][1]["format_score"], 3.0)
        self.assertEqual(result.dump_extra_info["reward_signals"][0], result.dump_extra_info["group_mean_reward_signal"][0])
        self.assertEqual(result.dump_extra_info["reward_weights"][0], result.dump_extra_info["group_mean_reward_weight"][0])
        self.assertEqual(
            result.dump_extra_info["sample_reward_weight"][0],
            result.dump_extra_info["group_mean_reward_weight"][0],
        )
        reward_detail = result.dump_extra_info["reward_detail"][0]
        self.assertEqual(reward_detail["sample_reward_signal"]["format_score"], 2.0)
        self.assertEqual(reward_detail["group_mean_reward_signal"]["format_score"], 3.0)

        infos_dict = {"reward": result.reward_tensor.sum(-1).cpu().tolist(), **result.validation_extra_info}
        data_sources = ["source_a"] * 4
        sample_uids = ["group_a", "group_a", "group_b", "group_b"]
        metrics = process_validation_metrics(data_sources, sample_uids, infos_dict, seed=42)

        self.assertIn("source_a", metrics)
        self.assertIn("reward", metrics["source_a"])
        self.assertIn("reward_signal_format_score", metrics["source_a"])
        self.assertIn("mean@2", metrics["source_a"]["reward"])
        self.assertAlmostEqual(metrics["source_a"]["reward_signal_format_score"]["mean@2"], 5.0)


if __name__ == "__main__":
    unittest.main()
