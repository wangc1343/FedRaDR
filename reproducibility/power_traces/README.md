# Raw power traces

File naming:

```text
power_rtx4090_{dataset}_{setting}_{method}_seed{seed}.samples.csv
power_jetson_orin_{dataset}_{setting}_{method}_seed{seed}.samples.csv
```

Normalized columns:

```csv
timestamp_ms,power_watts,power_scope,gpu_util_percent,memory_util_percent,memory_used_mib,temperature_c
```

For RTX 4090, `power_scope` is `gpu_board` and values come from `nvidia-smi`.
For Jetson AGX Orin, record whether the value is `board_total` or a named power
rail from `tegrastats`; do not label board power as GPU-only power. A 100 ms
polling interval is requested, but the sensor's internal update rate may be
slower, so repeated consecutive values must be retained rather than removed.

Each trace must be accompanied by a JSON summary containing the exact command,
return code, sampling interval, device identity, start/end timestamps, elapsed
time, sample count, and trapezoidally integrated energy in joules.

