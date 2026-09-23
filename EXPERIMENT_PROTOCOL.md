# FedRaDR experiment protocol

This protocol is the reproducibility record for the experiments added to the
official FedDC code base. It is intentionally limited to experimental
execution and reporting; it does not replace the mathematical description in
the paper.

## Primary matrix

- Datasets: CIFAR-10, CIFAR-100, and Tiny-ImageNet where the paper reports it.
- Models: ResNet-18 with GN; MobileNetV2 only for the stated transfer checks.
- Partitions: the same serialized Dirichlet/pathological partitions are reused
  by every method and every seed.
- Seeds: 20, 21, and 22 for every primary method/setting pair.
- Participation: C100-P10 and C500-P2 for the CIFAR comparisons.
- Metrics: final accuracy, target round, single-device aggregate compute time,
  communication payload, MFG, HFR, missing-class retention, peak memory, and
  energy when a hardware counter is available.

## Execution rules

1. Generate one client schedule per setting with
   `scripts/make_reproducibility_assets.py` and reuse it for all methods.
2. Record the partition hash, schedule hash, seed, commit, device, precision,
   batch size, local steps, optimizer, and all method-specific hyperparameters
   in `config.json` for every run.
3. Keep the exact client-mean normalization as the primary result. Mini-batch
   and EMA normalization are explicitly labeled lower-cost variants.
4. Save per-round metrics as JSONL and retain raw logs; aggregate only after all
   three seeds finish.
5. Treat a target not reached within the round budget as censored. Any speedup
   computed from `>R_max` is a strict lower bound, never an exact speedup.
6. Report single-device aggregate compute time as simulator cost. Do not call it
   parallel distributed wall-clock time unless a real multi-device experiment is
   run.

The helper `utils_methods_FedRaDR.py` follows the data-object conventions of
FedDC and can be called from the original FedDC experiment scripts.
