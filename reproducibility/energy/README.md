# Energy measurement protocol

The manuscript's GPU energy values must be supported by raw power traces. Use
`scripts/measure_nvidia_energy.py` for NVIDIA desktop/server GPUs.

## RTX 4090 protocol

1. Fix the GPU index, model, precision, batch size, local stages, data subset,
   teacher-cache mode, and normalization mode.
2. Run one unreported warm-up workload.
3. Run each measured workload at least three times with seeds 20, 21, and 22.
4. Sample `nvidia-smi` power draw at 100 ms or faster where supported.
5. Integrate the sampled power by the trapezoidal rule. Retain both the CSV
   trace and JSON summary.
6. Report mean and sample standard deviation over the three independent
   measurements. Do not infer energy from elapsed time alone.

Example:

```bash
python scripts/measure_nvidia_energy.py \
  --gpu-index 0 --interval-ms 100 \
  --output-prefix reproducibility/energy/rtx4090-fedradr-seed20 \
  -- python scripts/run_fedradr.py <arguments>
```

The wrapper records the exact child command, return code, timestamps, GPU
identity, driver information, raw power samples, elapsed time, and integrated
joules.

## Jetson AGX Orin

Use `tegrastats` (or an equivalent board telemetry interface) and retain its raw
timestamped output. Board-level power and GPU-rail power are different
quantities and must be labeled explicitly. The currently reported single
Jetson observation is qualitative; it cannot be presented with an error bar
until repeated measurements are available.

## Missing historical evidence

No raw RTX 4090 or Jetson power traces underlying the manuscript values were
found in the local repository during the 2026-09-25 audit. Recover those traces
from the measurement host or rerun the protocol before claiming that raw energy
records have been released.

