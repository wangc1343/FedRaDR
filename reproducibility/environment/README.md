# Environment records

`local-audit.json` records the machine used to validate this repository in the
current workspace. It is a CPU-only environment and is not the environment that
produced the manuscript's RTX 4090 or Jetson measurements.

Before release, run `scripts/capture_environment.py` on every experiment host
and retain the generated JSON beside the corresponding raw logs. Add precision,
batch size, cache policy, local-stage count, and any power-limit or clock
settings that are not discoverable automatically.

