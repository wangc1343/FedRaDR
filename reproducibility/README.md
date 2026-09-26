# Reproducibility package

This directory is the release checklist for the experiments reported in
"FedRaDR: Reliability-Aware Distillation and Regularization for Non-IID
Federated Learning." The implementation is derived from the official FedDC
repository. The FedRaDR code is in `fedradr/` and
`utils_methods_FedRaDR.py`.

## Ready in this repository

- runnable FedRaDR training and smoke tests;
- deterministic seed handling and fixed client-schedule generation;
- shared and FedRaDR-specific configuration records;
- the reported controlled-participation, forgetting, and missing-class
  aggregate records used in Appendix C;
- partition identifiers and expected SHA-256 prefixes stated in the paper;
- environment capture and NVIDIA power-sampling utilities;
- an explicit artifact manifest with verification status.

## Required before the public release is claimed as complete

The files below were not found in the local repository on 2026-09-25. They
must be recovered from the machines that produced the paper results or rerun
under the published protocol. Do not mark them as released until their hashes
and contents have been verified.

1. Serialized historical partition files matching the identifiers in
   `partitions/manifest.json`.
2. Full per-round JSONL/stdout logs for every reported method, setting, and
   seed. Appendix-level first-hit and aggregate records are not substitutes
   for full training logs.
3. Raw RTX 4090 and Jetson power traces underlying the reported energy values.
4. The exact baseline tuning grids and selection records beyond the final
   author-provided coefficients.
5. A public repository owned by the authors. The current Git remote points to
   the upstream FedDC repository and must not be cited as the FedRaDR artifact.

## Verification

Run:

```bash
python -m pytest -q
python scripts/smoke_fedradr.py --output smoke_output
python scripts/capture_environment.py --output reproducibility/environment/local.json
```

Generate or verify schedules with:

```bash
python -m scripts.make_reproducibility_assets \
  --clients 100 --clients-per-round 10 --rounds 1000 --seed 20 \
  --output reproducibility/schedules/c100-p10-r1000-seed20.json
```

Every public run directory should contain `config.json`,
`client_schedule.json`, `metrics.jsonl`, and the relevant partition hash.
