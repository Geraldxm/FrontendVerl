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

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RayTrainerDumpTest(unittest.TestCase):
    def test_dump_generations_uses_compact_schema(self):
        reward_extra_infos = {
            "reward_raw_score": [0.0],
            "reward_direct_score": [5.8181818],
            "reward_focal_score": [2.337951],
            "reward_final_score": [5.8181818],
            "reward_valid_sample": [1],
            "reward_is_llm_generation_error": [0],
            "sample_reward_signal": [
                {
                    "format_score": 10.0,
                    "console_errors": 8.0,
                    "network_violations": 10.0,
                }
            ],
            "sample_reward_weight": [
                {
                    "format_score": 0.1,
                    "console_errors": 0.2,
                    "network_violations": 0.7,
                }
            ],
            "group_mean_reward_signal": [
                {
                    "format_score": 9.0,
                    "console_errors": 7.0,
                    "network_violations": 10.0,
                }
            ],
            "group_mean_reward_weight": [
                {
                    "format_score": 0.1,
                    "console_errors": 0.2,
                    "network_violations": 0.7,
                }
            ],
            "reward_others": [
                {
                    "max": {"format_score": 10.0, "console_errors": 8.0, "network_violations": 10.0},
                    "min": {"format_score": 10.0, "console_errors": 8.0, "network_violations": 10.0},
                    "var": {"format_score": 0.0, "console_errors": 0.0, "network_violations": 0.0},
                }
            ],
            "overall_status": ["success"],
            "error_message": [""],
            "render_info": [{"status": "success", "details": {"desktop": "ok"}}],
            "judge_info": [{"status": "success", "response": "fine"}],
            "reward_signal_format_score": [10.0],
            "reward_detail": [
                {
                    "scores": {"raw": 0.0, "direct": 5.8181818, "focal": 2.337951, "final": 5.8181818},
                    "sample_reward_signal": {"format_score": 10.0},
                    "sample_reward_weight": {"format_score": 0.1},
                    "group_mean_reward_signal": {"format_score": 9.0},
                    "group_mean_reward_weight": {"format_score": 0.1},
                    "reward_others": {"max": {"format_score": 10.0}, "min": {"format_score": 10.0}, "var": {"format_score": 0.0}},
                    "status": {
                        "overall": "success",
                        "error_message": "",
                        "render": {"status": "success"},
                        "judge": {"status": "success"},
                    },
                    "valid_sample": 1,
                    "is_llm_generation_error": 0,
                }
            ],
            "reward": [5.8181818],
            "request_id": ["ignored"],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = SimpleNamespace(global_steps=12)
            RayPPOTrainer._dump_generations(
                trainer,
                inputs=["input text"],
                outputs=["output text"],
                gts=["ground truth"],
                scores=[5.8181818],
                reward_extra_infos_dict=reward_extra_infos,
                dump_path=tmp_dir,
            )

            dump_path = Path(tmp_dir) / "12.jsonl"
            self.assertTrue(dump_path.exists())

            rows = [json.loads(line) for line in dump_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]

            expected_keys = {
                "input",
                "output",
                "gts",
                "step",
                "reward_scores",
                "sample_reward_signal",
                "sample_reward_weight",
                "group_mean_reward_signal",
                "group_mean_reward_weight",
                "reward_others",
                "status",
                "render_info",
                "judge_info",
            }
            self.assertEqual(set(row.keys()), expected_keys)

            self.assertEqual(row["input"], "input text")
            self.assertEqual(row["output"], "output text")
            self.assertEqual(row["gts"], "ground truth")
            self.assertEqual(row["step"], 12)
            self.assertEqual(
                row["reward_scores"],
                {
                    "raw": 0.0,
                    "direct": 5.8181818,
                    "focal": 2.337951,
                    "final": 5.8181818,
                    "valid_sample": 1,
                    "is_llm_generation_error": 0,
                },
            )
            self.assertEqual(
                row["sample_reward_signal"],
                {
                    "format_score": 10.0,
                    "console_errors": 8.0,
                    "network_violations": 10.0,
                },
            )
            self.assertEqual(
                row["sample_reward_weight"],
                {
                    "format_score": 0.1,
                    "console_errors": 0.2,
                    "network_violations": 0.7,
                },
            )
            self.assertEqual(
                row["group_mean_reward_signal"],
                {
                    "format_score": 9.0,
                    "console_errors": 7.0,
                    "network_violations": 10.0,
                },
            )
            self.assertEqual(
                row["group_mean_reward_weight"],
                {
                    "format_score": 0.1,
                    "console_errors": 0.2,
                    "network_violations": 0.7,
                },
            )
            self.assertEqual(
                row["reward_others"],
                {
                    "max": {"format_score": 10.0, "console_errors": 8.0, "network_violations": 10.0},
                    "min": {"format_score": 10.0, "console_errors": 8.0, "network_violations": 10.0},
                    "var": {"format_score": 0.0, "console_errors": 0.0, "network_violations": 0.0},
                },
            )
            self.assertNotIn("reward_signals", row)
            self.assertNotIn("reward_weights", row)
            self.assertNotIn("focal_weights", row)
            self.assertEqual(row["status"], {"overall": "success", "error_message": ""})
            self.assertEqual(row["render_info"]["details"]["desktop"], "ok")
            self.assertEqual(row["judge_info"]["response"], "fine")


if __name__ == "__main__":
    unittest.main()
