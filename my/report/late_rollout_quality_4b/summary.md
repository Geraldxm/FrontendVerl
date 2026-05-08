# 后期 rollout 批量质量分析

## 口径

- 后期窗口只纳入非旧运行尾巴、`valid_rate` 达标、`request_error_rate` 未超阈值的 step。
- rubric 使用 `valid_sample=1` 口径，避免 reward server 失败行 0 分污染。
- `final` 只表示该实验实际训练 reward；baseline 的 `final=direct`，focal 的 `final=focal`，两者不是同一标尺。

## 后期实验汇总

| 实验 | steps | direct | focal | final | subjective | ui | valid_rate | req_err | max_w | top1_same | signflip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v3_n16_3e-6_4k_Qwen3-4B` | 42,43,44,45,46 | 9.294 | 8.047 | 9.294 | 8.402 | 8.439 | 0.966 | 0.000 | 0.664 | 0.931 | 0.143 |
| `baseline_v3_n16_3e-6_Qwen3-4B` | 40,41,42,43,44 | 9.354 | 8.215 | 9.354 | 8.522 | 8.621 | 0.979 | 0.000 | 0.662 | 0.938 | 0.162 |
| `baseline_v3_n16_Qwen3-4B` | 20,21,22,23,24 | 8.843 | 6.737 | 8.843 | 7.562 | 7.035 | 0.991 | 0.000 | 0.761 | 0.772 | 0.170 |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | 26,27,28,29,30 | 9.267 | 8.049 | 8.049 | 8.416 | 8.246 | 0.969 | 0.000 | 0.579 | 0.928 | 0.144 |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | 35,36,37,38,39 | 9.230 | 8.023 | 8.023 | 8.316 | 8.237 | 0.981 | 0.000 | 0.543 | 0.941 | 0.137 |
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` | 15,16,17,18,20 | 9.163 | 7.845 | 7.845 | 8.223 | 7.999 | 0.946 | 0.000 | 0.564 | 0.856 | 0.139 |

## 逐 step 后期明细

| 实验 | step | direct | focal | final | subjective | ui | valid_rate | req_err | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
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
| `baseline_v3_n16_Qwen3-4B` | 20 | 8.911 | 6.809 | 8.911 | 7.719 | 7.093 | 0.992 | 0.000 | `{"llm_generation_error": 2, "parse_fail": 8, "success": 1014}` |
| `baseline_v3_n16_Qwen3-4B` | 21 | 8.833 | 6.675 | 8.833 | 7.541 | 6.991 | 0.991 | 0.000 | `{"llm_generation_error": 4, "parse_fail": 9, "success": 1011}` |
| `baseline_v3_n16_Qwen3-4B` | 22 | 8.797 | 6.656 | 8.797 | 7.406 | 7.080 | 0.996 | 0.000 | `{"llm_generation_error": 6, "parse_fail": 4, "success": 1014}` |
| `baseline_v3_n16_Qwen3-4B` | 23 | 8.800 | 6.695 | 8.800 | 7.535 | 6.903 | 0.984 | 0.000 | `{"llm_generation_error": 15, "parse_fail": 13, "render_engine_error": 3, "success": 993}` |
| `baseline_v3_n16_Qwen3-4B` | 24 | 8.874 | 6.849 | 8.874 | 7.611 | 7.108 | 0.992 | 0.000 | `{"llm_generation_error": 3, "parse_fail": 8, "success": 1013}` |
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
