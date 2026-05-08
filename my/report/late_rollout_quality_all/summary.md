# 后期 rollout 批量质量分析

## 口径

- 后期窗口只纳入非旧运行尾巴、`valid_rate` 达标、`request_error_rate` 未超阈值的 step。
- rubric 使用 `valid_sample=1` 口径，避免 reward server 失败行 0 分污染。
- `final` 只表示该实验实际训练 reward；baseline 的 `final=direct`，focal 的 `final=focal`，两者不是同一标尺。

## 后期实验汇总

| 实验 | steps | direct | focal | final | subjective | ui | valid_rate | req_err | max_w | top1_same | signflip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-1.7B-Base` | 57,58,59,60,61 | 9.976 | 9.929 | 9.976 | 0.000 | 0.000 | 0.999 | 0.000 | 0.263 | 1.000 | 0.003 |
| `baseline_0414_Qwen3-1.7B` | 54,55,56,57,58 | 8.082 | 5.227 | 8.082 | 0.000 | 7.036 | 0.813 | 0.000 | 0.118 | 0.713 | 0.143 |
| `baseline_0414_Qwen3-1.7B-Base` | 71,72,73,74,75 | 7.954 | 5.048 | 7.954 | 0.000 | 6.575 | 0.941 | 0.000 | 0.134 | 0.706 | 0.151 |
| `baseline_0414_Qwen3-4B` | 1 | 6.834 | 3.984 | 6.834 | 0.000 | 5.808 | 0.938 | 0.000 | 0.105 | 0.750 | 0.119 |
| `baseline_3_Qwen3-4B` | 64,65,66,67,68 | 8.524 | 6.185 | 8.524 | 0.000 | 7.740 | 0.809 | 0.000 | nan | 0.766 | 0.121 |
| `baseline_4_Qwen3-4B` | 21,22,23,24,25 | 7.953 | 4.960 | 7.953 | 0.000 | 6.929 | 0.882 | 0.000 | nan | 0.706 | 0.130 |
| `baseline_Qwen3-1.7B-Base` | 1,2 | 8.630 | 6.486 | 8.630 | 0.000 | 0.000 | 0.983 | 0.000 | 0.690 | 0.953 | 0.057 |
| `baseline_a11y_Qwen3-1.7B-Base` | 23,24,25,26,27 | 6.153 | 2.678 | 6.153 | 0.000 | 5.425 | 0.968 | 0.000 | 0.060 | 0.756 | 0.136 |
| `baseline_v3_Qwen3-1.7B-Base` | 136,137,138,139,140 | 9.115 | 7.775 | 9.115 | 0.000 | 7.815 | 0.985 | 0.000 | 0.439 | 0.891 | 0.088 |
| `baseline_v3_n16_0422_Qwen3-1.7B-Base` | 102,103,104,105,106 | 9.129 | 7.839 | 9.129 | 0.000 | 7.855 | 0.975 | 0.000 | 0.428 | 0.894 | 0.081 |
| `baseline_v3_n16_3e-6_4k_Qwen3-4B` | 42,43,44,45,46 | 9.294 | 8.047 | 9.294 | 8.402 | 8.439 | 0.966 | 0.000 | 0.664 | 0.931 | 0.143 |
| `baseline_v3_n16_3e-6_Qwen3-4B` | 40,41,42,43,44 | 9.354 | 8.215 | 9.354 | 8.522 | 8.621 | 0.979 | 0.000 | 0.662 | 0.938 | 0.162 |
| `baseline_v3_n16_Qwen3-1.7B` | 80,81,82,83,84 | 9.190 | 7.637 | 9.190 | 8.283 | 7.903 | 0.986 | 0.000 | 0.742 | 0.853 | 0.173 |
| `baseline_v3_n16_Qwen3-4B` | 20,21,22,23,24 | 8.843 | 6.737 | 8.843 | 7.562 | 7.035 | 0.991 | 0.000 | 0.761 | 0.772 | 0.170 |
| `focal_0414_Qwen3-1.7B` | 53,54,55,56,57 | 7.963 | 5.376 | 5.376 | 0.000 | 6.866 | 0.712 | 0.000 | 0.135 | 0.653 | 0.149 |
| `focal_0414_Qwen3-1.7B-Base` | 71,72,73,74,75 | 7.898 | 5.253 | 5.253 | 0.000 | 6.284 | 0.851 | 0.000 | 0.165 | 0.709 | 0.142 |
| `focal_0414_Qwen3-4B` | 1 | 6.685 | 3.933 | 3.933 | 0.000 | 5.615 | 0.943 | 0.000 | 0.113 | 0.859 | 0.122 |
| `focal_3_Qwen3-4B` | 42,43,44,45,46 | 7.964 | 5.981 | 5.981 | 0.000 | 7.219 | 0.696 | 0.000 | nan | 0.761 | 0.135 |
| `focal_4_Qwen3-4B` | 20,21,22,23,24 | 7.758 | 5.239 | 5.239 | 0.000 | 7.008 | 0.880 | 0.000 | nan | 0.725 | 0.156 |
| `focal_a11y_Qwen3-1.7B-Base` | 20,21,22,23,24 | 5.925 | 2.636 | 2.636 | 0.000 | 5.121 | 0.960 | 0.000 | 0.066 | 0.784 | 0.167 |
| `focal_v3_Qwen3-1.7B-Base` | 132,133,134,135,136 | 9.062 | 7.739 | 7.739 | 0.000 | 7.696 | 0.981 | 0.000 | 0.445 | 0.822 | 0.117 |
| `focal_v3_e0_t10_g8_Qwen3-1.7B-Base` | 95,96,97,98,99 | 8.995 | 7.402 | 7.402 | 7.949 | 7.648 | 0.988 | 0.000 | 0.712 | 0.766 | 0.197 |
| `focal_v3_e0_t1_g8_Qwen3-1.7B-Base` | 102,103,104,105,106 | 8.953 | 7.533 | 7.533 | 8.082 | 7.754 | 0.983 | 0.000 | 0.727 | 0.681 | 0.261 |
| `focal_v3_n16_e005_t10_g5_Qwen3-1.7B` | 17,18,19,20,21 | 8.537 | 6.580 | 6.580 | 7.279 | 6.800 | 0.992 | 0.000 | 0.567 | 0.650 | 0.183 |
| `focal_v3_n16_e005_t5_g5_Qwen3-1.7B` | 17,18,19,20,21 | 8.553 | 6.584 | 6.584 | 7.310 | 6.785 | 0.992 | 0.000 | 0.574 | 0.653 | 0.186 |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | 26,27,28,29,30 | 9.267 | 8.049 | 8.049 | 8.416 | 8.246 | 0.969 | 0.000 | 0.579 | 0.928 | 0.144 |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | 35,36,37,38,39 | 9.230 | 8.023 | 8.023 | 8.316 | 8.237 | 0.981 | 0.000 | 0.543 | 0.941 | 0.137 |
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` | 15,16,17,18,20 | 9.163 | 7.845 | 7.845 | 8.223 | 7.999 | 0.946 | 0.000 | 0.564 | 0.856 | 0.139 |
| `focal_v3_n16_e0_t1_g8_Qwen3-1.7B` | 78,79,80,81,82 | 9.022 | 7.488 | 7.488 | 8.170 | 7.746 | 0.988 | 0.000 | 0.741 | 0.666 | 0.212 |
| `focal_v3_n16_g2_Qwen3-1.7B-Base` | 102,103,104,105,106 | 9.097 | 7.962 | 7.962 | 0.000 | 7.829 | 0.989 | 0.000 | 0.349 | 0.853 | 0.074 |
| `focal_v3_n16_g3_Qwen3-1.7B-Base` | 90,91,92,93,94 | 9.019 | 7.699 | 7.699 | 0.000 | 7.711 | 0.978 | 0.000 | 0.424 | 0.778 | 0.133 |
| `focal_v3_n8_g2_Qwen3-1.7B-Base` | 110,111,112,113,114 | 9.011 | 7.788 | 7.788 | 0.000 | 7.538 | 0.991 | 0.000 | 0.367 | 0.803 | 0.105 |
| `focal_v4_0421_Qwen3-1.7B-Base` | 112,113,114,115,116 | 9.992 | 9.977 | 9.977 | 0.000 | 0.000 | 1.000 | 0.000 | 0.218 | 1.000 | 0.000 |

## 逐 step 后期明细

| 实验 | step | direct | focal | final | subjective | ui | valid_rate | req_err | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen3-1.7B-Base` | 57 | 9.940 | 9.832 | 9.940 | 0.000 | 0.000 | 1.000 | 0.000 | `{"llm_generation_error": 5, "success": 507}` |
| `Qwen3-1.7B-Base` | 58 | 9.974 | 9.921 | 9.974 | 0.000 | 0.000 | 0.998 | 0.000 | `{"render_engine_error": 1, "success": 511}` |
| `Qwen3-1.7B-Base` | 59 | 9.989 | 9.960 | 9.989 | 0.000 | 0.000 | 1.000 | 0.000 | `{"success": 512}` |
| `Qwen3-1.7B-Base` | 60 | 9.985 | 9.953 | 9.985 | 0.000 | 0.000 | 0.998 | 0.000 | `{"render_engine_error": 1, "success": 511}` |
| `Qwen3-1.7B-Base` | 61 | 9.992 | 9.976 | 9.992 | 0.000 | 0.000 | 0.998 | 0.000 | `{"render_engine_error": 1, "success": 511}` |
| `baseline_0414_Qwen3-1.7B` | 54 | 8.043 | 5.242 | 8.043 | 0.000 | 6.856 | 0.895 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 10, "render_engine_error": 44, "success": 455}` |
| `baseline_0414_Qwen3-1.7B` | 55 | 8.182 | 5.561 | 8.182 | 0.000 | 7.097 | 0.748 | 0.000 | `{"parse_fail": 9, "render_engine_error": 120, "success": 383}` |
| `baseline_0414_Qwen3-1.7B` | 56 | 8.076 | 5.189 | 8.076 | 0.000 | 7.087 | 0.873 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 16, "render_engine_error": 49, "success": 446}` |
| `baseline_0414_Qwen3-1.7B` | 57 | 8.100 | 5.195 | 8.100 | 0.000 | 7.117 | 0.801 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 10, "render_engine_error": 92, "success": 407}` |
| `baseline_0414_Qwen3-1.7B` | 58 | 8.010 | 4.948 | 8.010 | 0.000 | 7.021 | 0.750 | 0.000 | `{"llm_generation_error": 5, "parse_fail": 11, "render_engine_error": 117, "success": 379}` |
| `baseline_0414_Qwen3-1.7B-Base` | 71 | 7.930 | 5.072 | 7.930 | 0.000 | 6.406 | 0.953 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 7, "render_engine_error": 17, "success": 487}` |
| `baseline_0414_Qwen3-1.7B-Base` | 72 | 7.940 | 5.016 | 7.940 | 0.000 | 6.600 | 0.951 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 8, "render_engine_error": 17, "success": 485}` |
| `baseline_0414_Qwen3-1.7B-Base` | 73 | 7.994 | 5.264 | 7.994 | 0.000 | 6.532 | 0.932 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 11, "render_engine_error": 24, "success": 475}` |
| `baseline_0414_Qwen3-1.7B-Base` | 74 | 7.996 | 5.109 | 7.996 | 0.000 | 6.701 | 0.947 | 0.000 | `{"parse_fail": 9, "render_engine_error": 18, "success": 485}` |
| `baseline_0414_Qwen3-1.7B-Base` | 75 | 7.910 | 4.777 | 7.910 | 0.000 | 6.637 | 0.920 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 9, "render_engine_error": 32, "success": 469}` |
| `baseline_0414_Qwen3-4B` | 1 | 6.834 | 3.984 | 6.834 | 0.000 | 5.808 | 0.938 | 0.000 | `{"llm_generation_error": 53, "parse_fail": 9, "render_engine_error": 23, "success": 427}` |
| `baseline_3_Qwen3-4B` | 64 | 8.521 | 6.404 | 8.521 | 0.000 | 7.701 | 0.863 | 0.000 | `{"parse_fail": 16, "render_engine_error": 54, "success": 442}` |
| `baseline_3_Qwen3-4B` | 65 | 8.443 | 6.027 | 8.443 | 0.000 | 7.706 | 0.797 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 24, "render_engine_error": 80, "success": 407}` |
| `baseline_3_Qwen3-4B` | 66 | 8.373 | 5.868 | 8.373 | 0.000 | 7.505 | 0.797 | 0.000 | `{"parse_fail": 18, "render_engine_error": 86, "success": 408}` |
| `baseline_3_Qwen3-4B` | 67 | 8.502 | 6.240 | 8.502 | 0.000 | 7.923 | 0.857 | 0.000 | `{"parse_fail": 24, "render_engine_error": 49, "success": 439}` |
| `baseline_3_Qwen3-4B` | 68 | 8.781 | 6.384 | 8.781 | 0.000 | 7.867 | 0.732 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 31, "render_engine_error": 106, "success": 374}` |
| `baseline_4_Qwen3-4B` | 21 | 7.876 | 4.810 | 7.876 | 0.000 | 6.806 | 0.906 | 0.000 | `{"llm_generation_error": 6, "parse_fail": 10, "render_engine_error": 38, "success": 458}` |
| `baseline_4_Qwen3-4B` | 22 | 7.876 | 4.615 | 7.876 | 0.000 | 6.987 | 0.910 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 12, "render_engine_error": 34, "success": 464}` |
| `baseline_4_Qwen3-4B` | 23 | 8.033 | 5.222 | 8.033 | 0.000 | 7.021 | 0.846 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 18, "render_engine_error": 61, "success": 429}` |
| `baseline_4_Qwen3-4B` | 24 | 8.013 | 5.084 | 8.013 | 0.000 | 6.874 | 0.912 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 12, "render_engine_error": 33, "success": 463}` |
| `baseline_4_Qwen3-4B` | 25 | 7.966 | 5.070 | 7.966 | 0.000 | 6.956 | 0.834 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 16, "render_engine_error": 69, "success": 424}` |
| `baseline_Qwen3-1.7B-Base` | 1 | 8.650 | 6.497 | 8.650 | 0.000 | 0.000 | 0.980 | 0.000 | `{"llm_generation_error": 109, "render_engine_error": 10, "success": 393}` |
| `baseline_Qwen3-1.7B-Base` | 2 | 8.609 | 6.474 | 8.609 | 0.000 | 0.000 | 0.986 | 0.000 | `{"llm_generation_error": 106, "render_engine_error": 7, "success": 399}` |
| `baseline_a11y_Qwen3-1.7B-Base` | 23 | 6.108 | 2.696 | 6.108 | 0.000 | 5.328 | 0.959 | 0.000 | `{"llm_generation_error": 16, "parse_fail": 3, "render_engine_error": 18, "success": 475}` |
| `baseline_a11y_Qwen3-1.7B-Base` | 24 | 6.101 | 2.622 | 6.101 | 0.000 | 5.349 | 0.967 | 0.000 | `{"llm_generation_error": 17, "parse_fail": 5, "render_engine_error": 12, "success": 478}` |
| `baseline_a11y_Qwen3-1.7B-Base` | 25 | 6.154 | 2.649 | 6.154 | 0.000 | 5.584 | 0.967 | 0.000 | `{"llm_generation_error": 14, "parse_fail": 9, "render_engine_error": 8, "success": 481}` |
| `baseline_a11y_Qwen3-1.7B-Base` | 26 | 6.173 | 2.699 | 6.173 | 0.000 | 5.407 | 0.975 | 0.000 | `{"llm_generation_error": 13, "parse_fail": 1, "render_engine_error": 12, "success": 486}` |
| `baseline_a11y_Qwen3-1.7B-Base` | 27 | 6.230 | 2.725 | 6.230 | 0.000 | 5.459 | 0.975 | 0.000 | `{"llm_generation_error": 14, "parse_fail": 3, "render_engine_error": 10, "success": 485}` |
| `baseline_v3_Qwen3-1.7B-Base` | 136 | 9.130 | 7.827 | 9.130 | 0.000 | 7.795 | 0.992 | 0.000 | `{"parse_fail": 4, "success": 508}` |
| `baseline_v3_Qwen3-1.7B-Base` | 137 | 9.104 | 7.727 | 9.104 | 0.000 | 7.725 | 0.994 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 3, "success": 507}` |
| `baseline_v3_Qwen3-1.7B-Base` | 138 | 9.125 | 7.827 | 9.125 | 0.000 | 7.935 | 0.967 | 0.000 | `{"parse_fail": 3, "render_engine_error": 14, "success": 495}` |
| `baseline_v3_Qwen3-1.7B-Base` | 139 | 9.090 | 7.692 | 9.090 | 0.000 | 7.753 | 0.982 | 0.000 | `{"parse_fail": 5, "render_engine_error": 4, "success": 503}` |
| `baseline_v3_Qwen3-1.7B-Base` | 140 | 9.125 | 7.803 | 9.125 | 0.000 | 7.866 | 0.990 | 0.000 | `{"render_engine_error": 5, "success": 507}` |
| `baseline_v3_n16_0422_Qwen3-1.7B-Base` | 102 | 9.112 | 7.760 | 9.112 | 0.000 | 7.885 | 0.971 | 0.000 | `{"parse_fail": 8, "render_engine_error": 22, "success": 994}` |
| `baseline_v3_n16_0422_Qwen3-1.7B-Base` | 103 | 9.141 | 7.858 | 9.141 | 0.000 | 7.800 | 0.987 | 0.000 | `{"parse_fail": 4, "render_engine_error": 9, "success": 1011}` |
| `baseline_v3_n16_0422_Qwen3-1.7B-Base` | 104 | 9.139 | 7.897 | 9.139 | 0.000 | 7.908 | 0.978 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 15, "render_engine_error": 8, "success": 1000}` |
| `baseline_v3_n16_0422_Qwen3-1.7B-Base` | 105 | 9.132 | 7.823 | 9.132 | 0.000 | 7.875 | 0.952 | 0.000 | `{"parse_fail": 16, "render_engine_error": 33, "success": 975}` |
| `baseline_v3_n16_0422_Qwen3-1.7B-Base` | 106 | 9.123 | 7.856 | 9.123 | 0.000 | 7.804 | 0.988 | 0.000 | `{"parse_fail": 3, "render_engine_error": 9, "success": 1012}` |
| `baseline_v3_n16_3e-6_4k_Qwen3-4B` | 42 | 9.279 | 8.029 | 9.279 | 8.333 | 8.512 | 0.977 | 0.000 | `{"parse_fail": 17, "render_engine_error": 7, "success": 1000}` |
| `baseline_v3_n16_3e-6_4k_Qwen3-4B` | 43 | 9.312 | 8.118 | 9.312 | 8.420 | 8.547 | 0.972 | 0.000 | `{"parse_fail": 28, "render_engine_error": 1, "success": 995}` |
| `baseline_v3_n16_3e-6_4k_Qwen3-4B` | 44 | 9.295 | 8.024 | 9.295 | 8.396 | 8.468 | 0.965 | 0.000 | `{"parse_fail": 27, "render_engine_error": 9, "success": 988}` |
| `baseline_v3_n16_3e-6_4k_Qwen3-4B` | 45 | 9.345 | 8.192 | 9.345 | 8.506 | 8.590 | 0.970 | 0.000 | `{"parse_fail": 17, "render_engine_error": 14, "success": 993}` |
| `baseline_v3_n16_3e-6_4k_Qwen3-4B` | 46 | 9.238 | 7.874 | 9.238 | 8.355 | 8.080 | 0.947 | 0.000 | `{"parse_fail": 34, "render_engine_error": 20, "success": 970}` |
| `baseline_v3_n16_3e-6_Qwen3-4B` | 40 | 9.298 | 8.047 | 9.298 | 8.393 | 8.503 | 0.985 | 0.000 | `{"parse_fail": 13, "render_engine_error": 2, "success": 1009}` |
| `baseline_v3_n16_3e-6_Qwen3-4B` | 41 | 9.336 | 8.164 | 9.336 | 8.499 | 8.524 | 0.983 | 0.000 | `{"parse_fail": 16, "render_engine_error": 1, "success": 1007}` |
| `baseline_v3_n16_3e-6_Qwen3-4B` | 42 | 9.375 | 8.280 | 9.375 | 8.563 | 8.685 | 0.979 | 0.000 | `{"parse_fail": 19, "render_engine_error": 3, "success": 1002}` |
| `baseline_v3_n16_3e-6_Qwen3-4B` | 43 | 9.405 | 8.373 | 9.405 | 8.604 | 8.836 | 0.970 | 0.000 | `{"parse_fail": 24, "render_engine_error": 7, "success": 993}` |
| `baseline_v3_n16_3e-6_Qwen3-4B` | 44 | 9.357 | 8.210 | 9.357 | 8.552 | 8.556 | 0.979 | 0.000 | `{"parse_fail": 21, "success": 1003}` |
| `baseline_v3_n16_Qwen3-1.7B` | 80 | 9.176 | 7.539 | 9.176 | 8.285 | 7.771 | 0.991 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 9, "success": 1014}` |
| `baseline_v3_n16_Qwen3-1.7B` | 81 | 9.183 | 7.669 | 9.183 | 8.300 | 7.791 | 0.983 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 17, "success": 1006}` |
| `baseline_v3_n16_Qwen3-1.7B` | 82 | 9.231 | 7.761 | 9.231 | 8.344 | 8.083 | 0.988 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 12, "success": 1011}` |
| `baseline_v3_n16_Qwen3-1.7B` | 83 | 9.209 | 7.727 | 9.209 | 8.282 | 8.048 | 0.982 | 0.000 | `{"parse_fail": 18, "success": 1006}` |
| `baseline_v3_n16_Qwen3-1.7B` | 84 | 9.152 | 7.488 | 9.152 | 8.202 | 7.824 | 0.986 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 13, "render_engine_error": 1, "success": 1008}` |
| `baseline_v3_n16_Qwen3-4B` | 20 | 8.911 | 6.809 | 8.911 | 7.719 | 7.093 | 0.992 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 8, "success": 1014}` |
| `baseline_v3_n16_Qwen3-4B` | 21 | 8.833 | 6.675 | 8.833 | 7.541 | 6.991 | 0.991 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 9, "success": 1011}` |
| `baseline_v3_n16_Qwen3-4B` | 22 | 8.797 | 6.656 | 8.797 | 7.406 | 7.080 | 0.996 | 0.000 | `{"llm_generation_error": 6, "parse_fail": 4, "success": 1014}` |
| `baseline_v3_n16_Qwen3-4B` | 23 | 8.800 | 6.695 | 8.800 | 7.535 | 6.903 | 0.984 | 0.000 | `{"llm_generation_error": 15, "parse_fail": 13, "render_engine_error": 3, "success": 993}` |
| `baseline_v3_n16_Qwen3-4B` | 24 | 8.874 | 6.849 | 8.874 | 7.611 | 7.108 | 0.992 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 8, "success": 1013}` |
| `focal_0414_Qwen3-1.7B` | 53 | 7.960 | 5.373 | 5.373 | 0.000 | 6.885 | 0.746 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 21, "render_engine_error": 109, "success": 378}` |
| `focal_0414_Qwen3-1.7B` | 54 | 7.938 | 5.343 | 5.343 | 0.000 | 6.832 | 0.732 | 0.000 | `{"llm_generation_error": 5, "parse_fail": 12, "render_engine_error": 125, "success": 370}` |
| `focal_0414_Qwen3-1.7B` | 55 | 8.102 | 5.613 | 5.613 | 0.000 | 6.996 | 0.557 | 0.000 | `{"parse_fail": 9, "render_engine_error": 218, "success": 285}` |
| `focal_0414_Qwen3-1.7B` | 56 | 7.909 | 5.232 | 5.232 | 0.000 | 6.813 | 0.764 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 10, "render_engine_error": 111, "success": 389}` |
| `focal_0414_Qwen3-1.7B` | 57 | 7.904 | 5.320 | 5.320 | 0.000 | 6.802 | 0.760 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 6, "render_engine_error": 117, "success": 386}` |
| `focal_0414_Qwen3-1.7B-Base` | 71 | 7.844 | 5.192 | 5.192 | 0.000 | 6.144 | 0.922 | 0.000 | `{"parse_fail": 10, "render_engine_error": 30, "success": 472}` |
| `focal_0414_Qwen3-1.7B-Base` | 72 | 7.876 | 5.198 | 5.198 | 0.000 | 6.271 | 0.922 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 10, "render_engine_error": 30, "success": 471}` |
| `focal_0414_Qwen3-1.7B-Base` | 73 | 7.947 | 5.428 | 5.428 | 0.000 | 6.207 | 0.678 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 7, "render_engine_error": 158, "success": 345}` |
| `focal_0414_Qwen3-1.7B-Base` | 74 | 7.921 | 5.311 | 5.311 | 0.000 | 6.390 | 0.861 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 8, "render_engine_error": 63, "success": 440}` |
| `focal_0414_Qwen3-1.7B-Base` | 75 | 7.898 | 5.138 | 5.138 | 0.000 | 6.407 | 0.873 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 5, "render_engine_error": 60, "success": 445}` |
| `focal_0414_Qwen3-4B` | 1 | 6.685 | 3.933 | 3.933 | 0.000 | 5.615 | 0.943 | 0.000 | `{"llm_generation_error": 67, "parse_fail": 6, "render_engine_error": 23, "success": 416}` |
| `focal_3_Qwen3-4B` | 42 | 7.907 | 6.010 | 6.010 | 0.000 | 7.020 | 0.797 | 0.000 | `{"llm_generation_error": 10, "parse_fail": 23, "render_engine_error": 81, "success": 398}` |
| `focal_3_Qwen3-4B` | 43 | 7.979 | 5.973 | 5.973 | 0.000 | 7.253 | 0.664 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 24, "render_engine_error": 148, "success": 337}` |
| `focal_3_Qwen3-4B` | 44 | 7.976 | 5.995 | 5.995 | 0.000 | 7.321 | 0.621 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 13, "render_engine_error": 181, "success": 314}` |
| `focal_3_Qwen3-4B` | 45 | 7.956 | 5.737 | 5.737 | 0.000 | 7.211 | 0.703 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 25, "render_engine_error": 127, "success": 356}` |
| `focal_3_Qwen3-4B` | 46 | 8.003 | 6.188 | 6.188 | 0.000 | 7.289 | 0.697 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 23, "render_engine_error": 132, "success": 353}` |
| `focal_4_Qwen3-4B` | 20 | 7.780 | 5.276 | 5.276 | 0.000 | 7.106 | 0.922 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 10, "render_engine_error": 30, "success": 470}` |
| `focal_4_Qwen3-4B` | 21 | 7.684 | 5.083 | 5.083 | 0.000 | 6.846 | 0.873 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 8, "render_engine_error": 57, "success": 444}` |
| `focal_4_Qwen3-4B` | 22 | 7.708 | 5.014 | 5.014 | 0.000 | 7.038 | 0.881 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 4, "render_engine_error": 57, "success": 449}` |
| `focal_4_Qwen3-4B` | 23 | 7.843 | 5.452 | 5.452 | 0.000 | 7.108 | 0.885 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 17, "render_engine_error": 42, "success": 452}` |
| `focal_4_Qwen3-4B` | 24 | 7.774 | 5.372 | 5.372 | 0.000 | 6.942 | 0.842 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 8, "render_engine_error": 73, "success": 428}` |
| `focal_a11y_Qwen3-1.7B-Base` | 20 | 5.815 | 2.545 | 2.545 | 0.000 | 5.016 | 0.965 | 0.000 | `{"llm_generation_error": 38, "parse_fail": 4, "render_engine_error": 14, "success": 456}` |
| `focal_a11y_Qwen3-1.7B-Base` | 21 | 5.815 | 2.515 | 2.515 | 0.000 | 5.184 | 0.953 | 0.000 | `{"llm_generation_error": 30, "parse_fail": 1, "render_engine_error": 23, "success": 458}` |
| `focal_a11y_Qwen3-1.7B-Base` | 22 | 5.898 | 2.614 | 2.614 | 0.000 | 5.096 | 0.959 | 0.000 | `{"llm_generation_error": 34, "parse_fail": 3, "render_engine_error": 18, "success": 457}` |
| `focal_a11y_Qwen3-1.7B-Base` | 23 | 6.080 | 2.765 | 2.765 | 0.000 | 5.262 | 0.969 | 0.000 | `{"llm_generation_error": 20, "parse_fail": 3, "render_engine_error": 13, "success": 476}` |
| `focal_a11y_Qwen3-1.7B-Base` | 24 | 6.018 | 2.742 | 2.742 | 0.000 | 5.045 | 0.953 | 0.000 | `{"llm_generation_error": 25, "parse_fail": 2, "render_engine_error": 22, "success": 463}` |
| `focal_v3_Qwen3-1.7B-Base` | 132 | 9.016 | 7.643 | 7.643 | 0.000 | 7.607 | 0.975 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 12, "render_engine_error": 1, "success": 497}` |
| `focal_v3_Qwen3-1.7B-Base` | 133 | 9.023 | 7.576 | 7.576 | 0.000 | 7.694 | 0.982 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 4, "render_engine_error": 5, "success": 502}` |
| `focal_v3_Qwen3-1.7B-Base` | 134 | 9.106 | 7.860 | 7.860 | 0.000 | 7.767 | 0.971 | 0.000 | `{"parse_fail": 5, "render_engine_error": 10, "success": 497}` |
| `focal_v3_Qwen3-1.7B-Base` | 135 | 9.091 | 7.833 | 7.833 | 0.000 | 7.727 | 0.986 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 5, "render_engine_error": 2, "success": 504}` |
| `focal_v3_Qwen3-1.7B-Base` | 136 | 9.075 | 7.782 | 7.782 | 0.000 | 7.684 | 0.990 | 0.000 | `{"parse_fail": 5, "success": 507}` |
| `focal_v3_e0_t10_g8_Qwen3-1.7B-Base` | 95 | 8.983 | 7.374 | 7.374 | 7.925 | 7.651 | 0.984 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 7, "render_engine_error": 9, "success": 1005}` |
| `focal_v3_e0_t10_g8_Qwen3-1.7B-Base` | 96 | 9.002 | 7.364 | 7.364 | 7.939 | 7.628 | 0.986 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 14, "success": 1009}` |
| `focal_v3_e0_t10_g8_Qwen3-1.7B-Base` | 97 | 9.002 | 7.487 | 7.487 | 7.982 | 7.693 | 0.987 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 13, "success": 1010}` |
| `focal_v3_e0_t10_g8_Qwen3-1.7B-Base` | 98 | 9.010 | 7.454 | 7.454 | 7.974 | 7.727 | 0.994 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 4, "render_engine_error": 2, "success": 1014}` |
| `focal_v3_e0_t10_g8_Qwen3-1.7B-Base` | 99 | 8.979 | 7.330 | 7.330 | 7.923 | 7.540 | 0.988 | 0.000 | `{"parse_fail": 10, "render_engine_error": 2, "success": 1012}` |
| `focal_v3_e0_t1_g8_Qwen3-1.7B-Base` | 102 | 8.942 | 7.500 | 7.500 | 8.051 | 7.786 | 0.979 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 9, "render_engine_error": 13, "success": 1001}` |
| `focal_v3_e0_t1_g8_Qwen3-1.7B-Base` | 103 | 8.965 | 7.597 | 7.597 | 8.136 | 7.719 | 0.994 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 2, "render_engine_error": 4, "success": 1017}` |
| `focal_v3_e0_t1_g8_Qwen3-1.7B-Base` | 104 | 8.959 | 7.560 | 7.560 | 8.110 | 7.734 | 0.977 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 9, "render_engine_error": 15, "success": 999}` |
| `focal_v3_e0_t1_g8_Qwen3-1.7B-Base` | 105 | 8.979 | 7.518 | 7.518 | 8.079 | 7.851 | 0.982 | 0.000 | `{"parse_fail": 8, "render_engine_error": 10, "success": 1006}` |
| `focal_v3_e0_t1_g8_Qwen3-1.7B-Base` | 106 | 8.922 | 7.489 | 7.489 | 8.035 | 7.681 | 0.985 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 8, "render_engine_error": 7, "success": 1007}` |
| `focal_v3_n16_e005_t10_g5_Qwen3-1.7B` | 17 | 8.436 | 6.485 | 6.485 | 7.137 | 6.680 | 0.993 | 0.000 | `{"llm_generation_error": 16, "parse_fail": 7, "success": 1001}` |
| `focal_v3_n16_e005_t10_g5_Qwen3-1.7B` | 18 | 8.520 | 6.496 | 6.496 | 7.236 | 6.755 | 0.988 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 12, "success": 1008}` |
| `focal_v3_n16_e005_t10_g5_Qwen3-1.7B` | 19 | 8.553 | 6.533 | 6.533 | 7.245 | 6.926 | 0.993 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 7, "success": 1013}` |
| `focal_v3_n16_e005_t10_g5_Qwen3-1.7B` | 20 | 8.622 | 6.773 | 6.773 | 7.443 | 6.846 | 0.992 | 0.000 | `{"llm_generation_error": 7, "parse_fail": 8, "success": 1009}` |
| `focal_v3_n16_e005_t10_g5_Qwen3-1.7B` | 21 | 8.556 | 6.615 | 6.615 | 7.334 | 6.791 | 0.992 | 0.000 | `{"llm_generation_error": 8, "parse_fail": 8, "success": 1008}` |
| `focal_v3_n16_e005_t5_g5_Qwen3-1.7B` | 17 | 8.488 | 6.515 | 6.515 | 7.189 | 6.705 | 0.995 | 0.000 | `{"llm_generation_error": 10, "parse_fail": 5, "success": 1009}` |
| `focal_v3_n16_e005_t5_g5_Qwen3-1.7B` | 18 | 8.528 | 6.476 | 6.476 | 7.270 | 6.705 | 0.994 | 0.000 | `{"llm_generation_error": 6, "parse_fail": 6, "success": 1012}` |
| `focal_v3_n16_e005_t5_g5_Qwen3-1.7B` | 19 | 8.539 | 6.498 | 6.498 | 7.231 | 6.842 | 0.990 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 10, "success": 1010}` |
| `focal_v3_n16_e005_t5_g5_Qwen3-1.7B` | 20 | 8.639 | 6.775 | 6.775 | 7.495 | 6.867 | 0.986 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 14, "success": 1007}` |
| `focal_v3_n16_e005_t5_g5_Qwen3-1.7B` | 21 | 8.572 | 6.657 | 6.657 | 7.367 | 6.804 | 0.994 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 6, "success": 1016}` |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | 26 | 9.238 | 7.986 | 7.986 | 8.345 | 8.190 | 0.966 | 0.000 | `{"parse_fail": 29, "render_engine_error": 6, "success": 989}` |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | 27 | 9.269 | 8.072 | 8.072 | 8.402 | 8.308 | 0.971 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 20, "render_engine_error": 10, "success": 993}` |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | 28 | 9.272 | 8.072 | 8.072 | 8.429 | 8.211 | 0.970 | 0.000 | `{"parse_fail": 30, "render_engine_error": 1, "success": 993}` |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | 29 | 9.265 | 8.000 | 8.000 | 8.441 | 8.174 | 0.966 | 0.000 | `{"parse_fail": 24, "render_engine_error": 11, "success": 989}` |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | 30 | 9.294 | 8.117 | 8.117 | 8.461 | 8.348 | 0.972 | 0.000 | `{"parse_fail": 27, "render_engine_error": 2, "success": 995}` |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | 35 | 9.224 | 7.935 | 7.935 | 8.289 | 8.234 | 0.984 | 0.000 | `{"parse_fail": 15, "render_engine_error": 1, "success": 1008}` |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | 36 | 9.202 | 7.978 | 7.978 | 8.219 | 8.252 | 0.983 | 0.000 | `{"parse_fail": 17, "success": 1007}` |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | 37 | 9.202 | 7.963 | 7.963 | 8.223 | 8.251 | 0.989 | 0.000 | `{"parse_fail": 11, "success": 1013}` |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | 38 | 9.279 | 8.191 | 8.191 | 8.446 | 8.348 | 0.972 | 0.000 | `{"parse_fail": 27, "render_engine_error": 2, "success": 995}` |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | 39 | 9.242 | 8.046 | 8.046 | 8.402 | 8.100 | 0.978 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 18, "render_engine_error": 5, "success": 1000}` |
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` | 15 | 9.100 | 7.703 | 7.703 | 8.054 | 7.898 | 0.953 | 0.000 | `{"parse_fail": 30, "render_engine_error": 18, "success": 976}` |
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` | 16 | 9.101 | 7.906 | 7.906 | 8.177 | 8.153 | 0.909 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 34, "render_engine_error": 59, "success": 930}` |
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` | 17 | 9.168 | 7.800 | 7.800 | 8.238 | 7.829 | 0.969 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 29, "render_engine_error": 3, "success": 991}` |
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` | 18 | 9.192 | 7.846 | 7.846 | 8.281 | 7.916 | 0.952 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 43, "render_engine_error": 6, "success": 974}` |
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` | 20 | 9.253 | 7.971 | 7.971 | 8.363 | 8.203 | 0.944 | 0.000 | `{"parse_fail": 35, "render_engine_error": 22, "success": 967}` |
| `focal_v3_n16_e0_t1_g8_Qwen3-1.7B` | 78 | 9.023 | 7.476 | 7.476 | 8.173 | 7.797 | 0.990 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 9, "render_engine_error": 1, "success": 1010}` |
| `focal_v3_n16_e0_t1_g8_Qwen3-1.7B` | 79 | 8.966 | 7.267 | 7.267 | 8.067 | 7.615 | 0.994 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 6, "success": 1017}` |
| `focal_v3_n16_e0_t1_g8_Qwen3-1.7B` | 80 | 8.989 | 7.387 | 7.387 | 8.139 | 7.585 | 0.988 | 0.000 | `{"llm_generation_error": 5, "parse_fail": 12, "success": 1007}` |
| `focal_v3_n16_e0_t1_g8_Qwen3-1.7B` | 81 | 9.026 | 7.602 | 7.602 | 8.179 | 7.745 | 0.979 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 15, "render_engine_error": 6, "success": 1001}` |
| `focal_v3_n16_e0_t1_g8_Qwen3-1.7B` | 82 | 9.107 | 7.706 | 7.706 | 8.292 | 7.990 | 0.987 | 0.000 | `{"parse_fail": 12, "render_engine_error": 1, "success": 1011}` |
| `focal_v3_n16_g2_Qwen3-1.7B-Base` | 102 | 9.087 | 7.936 | 7.936 | 0.000 | 7.840 | 0.987 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 12, "render_engine_error": 1, "success": 1009}` |
| `focal_v3_n16_g2_Qwen3-1.7B-Base` | 103 | 9.109 | 7.985 | 7.985 | 0.000 | 7.848 | 0.987 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 9, "render_engine_error": 4, "success": 1010}` |
| `focal_v3_n16_g2_Qwen3-1.7B-Base` | 104 | 9.099 | 7.958 | 7.958 | 0.000 | 7.781 | 0.992 | 0.000 | `{"parse_fail": 7, "render_engine_error": 1, "success": 1016}` |
| `focal_v3_n16_g2_Qwen3-1.7B-Base` | 105 | 9.096 | 7.946 | 7.946 | 0.000 | 7.881 | 0.987 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 13, "success": 1010}` |
| `focal_v3_n16_g2_Qwen3-1.7B-Base` | 106 | 9.094 | 7.983 | 7.983 | 0.000 | 7.797 | 0.989 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 2, "render_engine_error": 9, "success": 1012}` |
| `focal_v3_n16_g3_Qwen3-1.7B-Base` | 90 | 9.030 | 7.796 | 7.796 | 0.000 | 7.773 | 0.988 | 0.000 | `{"parse_fail": 10, "render_engine_error": 2, "success": 1012}` |
| `focal_v3_n16_g3_Qwen3-1.7B-Base` | 91 | 8.975 | 7.596 | 7.596 | 0.000 | 7.761 | 0.956 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 9, "render_engine_error": 36, "success": 978}` |
| `focal_v3_n16_g3_Qwen3-1.7B-Base` | 92 | 9.002 | 7.589 | 7.589 | 0.000 | 7.776 | 0.985 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 7, "render_engine_error": 8, "success": 1008}` |
| `focal_v3_n16_g3_Qwen3-1.7B-Base` | 93 | 9.036 | 7.708 | 7.708 | 0.000 | 7.575 | 0.975 | 0.000 | `{"parse_fail": 11, "render_engine_error": 15, "success": 998}` |
| `focal_v3_n16_g3_Qwen3-1.7B-Base` | 94 | 9.051 | 7.804 | 7.804 | 0.000 | 7.670 | 0.983 | 0.000 | `{"llm_generation_error": 1, "parse_fail": 11, "render_engine_error": 6, "success": 1006}` |
| `focal_v3_n8_g2_Qwen3-1.7B-Base` | 110 | 9.010 | 7.803 | 7.803 | 0.000 | 7.431 | 0.988 | 0.000 | `{"parse_fail": 6, "success": 506}` |
| `focal_v3_n8_g2_Qwen3-1.7B-Base` | 111 | 9.001 | 7.767 | 7.767 | 0.000 | 7.523 | 0.990 | 0.000 | `{"parse_fail": 5, "success": 507}` |
| `focal_v3_n8_g2_Qwen3-1.7B-Base` | 112 | 9.008 | 7.804 | 7.804 | 0.000 | 7.684 | 0.988 | 0.000 | `{"parse_fail": 6, "success": 506}` |
| `focal_v3_n8_g2_Qwen3-1.7B-Base` | 113 | 9.004 | 7.781 | 7.781 | 0.000 | 7.449 | 0.992 | 0.000 | `{"parse_fail": 4, "success": 508}` |
| `focal_v3_n8_g2_Qwen3-1.7B-Base` | 114 | 9.030 | 7.784 | 7.784 | 0.000 | 7.604 | 0.996 | 0.000 | `{"parse_fail": 2, "success": 510}` |
| `focal_v4_0421_Qwen3-1.7B-Base` | 112 | 9.998 | 9.996 | 9.996 | 0.000 | 0.000 | 1.000 | 0.000 | `{"success": 512}` |
| `focal_v4_0421_Qwen3-1.7B-Base` | 113 | 9.989 | 9.973 | 9.973 | 0.000 | 0.000 | 1.000 | 0.000 | `{"llm_generation_error": 1, "success": 511}` |
| `focal_v4_0421_Qwen3-1.7B-Base` | 114 | 9.987 | 9.953 | 9.953 | 0.000 | 0.000 | 1.000 | 0.000 | `{"success": 512}` |
| `focal_v4_0421_Qwen3-1.7B-Base` | 115 | 9.998 | 9.995 | 9.995 | 0.000 | 0.000 | 1.000 | 0.000 | `{"success": 512}` |
| `focal_v4_0421_Qwen3-1.7B-Base` | 116 | 9.988 | 9.967 | 9.967 | 0.000 | 0.000 | 1.000 | 0.000 | `{"llm_generation_error": 1, "success": 511}` |
