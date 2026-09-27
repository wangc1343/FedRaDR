# Reported result records

These CSV files are machine-readable copies of the records reported in the
revised manuscript and supplied independent-run records. They preserve the
supplied values and their stated aggregation level.

- `independent_rtt.csv`: Appendix B first-hit round and accuracy for every
  setting, method, and seed in the independent repeated campaign.
- `controlled_participation.csv`: fixed-client-population participation study.
- `balanced_forgetting.csv`: balanced three-seed MFG/HFR endpoint records.
- `missing_class.csv`: full-client missing-class summary statistics.
- `jetson_tinyimagenet_resource.csv`: repeated resource summary values.

They are not full per-round training logs. A complete public artifact must also
include `metrics.jsonl` (or equivalent raw stdout logs) for every method,
setting, and seed. Do not rename these aggregate records as raw trajectories.
