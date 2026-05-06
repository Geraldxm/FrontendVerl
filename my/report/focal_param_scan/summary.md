# Focal 参数网格扫描总结

- Rollout 目录: `rollouts/baseline_v3_n16_0422_Qwen3-1.7B-Base`
- 解析 step 数: 1
- 重点 step: 1
- temperatures: `0.1,0.5,1,3,5,10`
- gammas: `0:8:0.5`
- epsilons: `0,0.01,0.02,0.05,0.1,0.2`

## 解析数据规模

- 总行数: 1024
- 总组数: 64
- 当前排序对比组数（direct vs 已存 focal）: 64
- 网格 step 点位数: 612

## 一致性检查

- group signal 最大绝对误差: 0
- focal 回放最大绝对误差: 1.77636e-15
- 重点 step 1 行数: 1024
- 重点 step 1 组数: 64
- 重点 step 1 组大小分布: `{"16": 64}`
- 重点 step 1 group_signal 最大误差: 0
- 重点 step 1 focal 回放最大误差: 1.77636e-15

## 缺失字段统计

- `missing/judge_info.parsed_response`: 205
- `missing/subjective/color_harmony_and_theme_fit`: 211
- `missing/subjective/first_view_content_messaging`: 211
- `missing/subjective/typography_rhythm_and_readability`: 211
- `missing/subjective_scores`: 211
- `missing/ui_issue_types_key`: 205
- `sample_signal_source/reconstructed`: 1024

## 参数候选 Top-K

| 排名 | temp | gamma | epsilon | kendall_mean | signflip_mean | kendall_p90_mean | signflip_p90_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1 | 8 | 0 | 0.173503 | 0.227539 | 0.269583 | 0.375 |
| 2 | 0.1 | 8 | 0.01 | 0.172721 | 0.227539 | 0.266667 | 0.375 |
| 3 | 0.1 | 8 | 0.02 | 0.17194 | 0.227539 | 0.266667 | 0.375 |
| 4 | 0.1 | 7.5 | 0 | 0.17168 | 0.226562 | 0.266667 | 0.375 |
| 5 | 0.1 | 7.5 | 0.01 | 0.171159 | 0.227539 | 0.266667 | 0.375 |
| 6 | 0.1 | 7.5 | 0.02 | 0.171029 | 0.226562 | 0.265417 | 0.375 |
| 7 | 0.1 | 7 | 0 | 0.170768 | 0.226562 | 0.258333 | 0.375 |
| 8 | 0.1 | 7 | 0.01 | 0.170638 | 0.225586 | 0.258333 | 0.375 |
| 9 | 0.1 | 8 | 0.05 | 0.169792 | 0.224609 | 0.266667 | 0.375 |
| 10 | 0.1 | 7 | 0.02 | 0.169466 | 0.224609 | 0.258333 | 0.375 |

## 输出文件

- `group_metrics_current.csv`
- `grid_group_metrics.csv`
- `grid_step_metrics.csv`
- `heatmaps/*.png`
- `summary.md`
