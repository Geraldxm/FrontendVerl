# Focal vs Direct Reward Ranking Analysis

- Rollout dir: `rollouts/focal_v3_n16_g3_Qwen3-1.7B-Base`
- Steps parsed: 94
- Groups parsed: 6016
- Group key: `sha256(input)`
- Ranking scores: `reward_scores.direct` vs `reward_scores.focal`

## Overall Result

- Mean normalized Kendall distance: 0.120608
- Mean z-advantage correlation: 0.884058
- Mean z-advantage sign flip rate: 0.146121
- Mean top-1 same rate: 0.736037
- Lowest mean Kendall distance step: 2 (0.0965495)
- Highest mean Kendall distance step: 13 (0.146224)

## Consistency Check

- Top-level `reward_signals` and `reward_weights` are group-level views, not per-sample raw signals.
- Row-level `calc_direct_score/calc_focal_score` therefore may differ from the row's `reward_scores.direct/focal`.
- Group-level checks compare those calculated scores with each group's mean direct/focal scores.
- Max group direct abs error: 1.77636e-15
- Max group focal abs error: 1.77636e-15

## Focus Step 58

- Rows: 1024
- Groups: 64
- Group size counts: `{"16": 64}`
- Status counts: `{"llm_generation_error": 1, "parse_fail": 13, "render_engine_error": 4, "success": 1006}`
- Row direct abs error mean/max: 0.286737 / 5.41635
- Row focal abs error mean/max: 0.667137 / 7.34144
- Group direct abs error max: 1.77636e-15
- Group focal abs error max: 8.88178e-16
- Mean normalized Kendall distance: 0.13444
- Mean z-advantage correlation: 0.876364
- Mean z-advantage sign flip rate: 0.150391
- Top-1 same rate: 0.75

## Group Size Anomalies

- None. Every parsed step has 64 groups of size 16.

## Interpretation

- The ranking metrics show that focal changes the ordering, but the average change is modest.
- GRPO-style z-advantage correlation remains high, so the resulting advantage signal is largely aligned.
- The sign flip rate is the most direct check for whether samples cross the group mean after focal reweighting.

## Output Files

- `step_metrics.csv`
- `group_metrics.csv`
- `ranking_diff_curve.png`
