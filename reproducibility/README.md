# Reproducibility package

The unified build workflow is documented in [`BUILD_GUIDE.md`](BUILD_GUIDE.md)
and implemented by `scripts/build_reproducibility_bundle.py`. It records
generated rerun assets separately from recovered historical evidence.

This directory is the release checklist for the experiments reported in
"FedRaDR: Reliability-Aware Distillation and Regularization for Non-IID
Federated Learning." The implementation is derived from the official FedDC
repository. The FedRaDR code is in `fedradr/` and
`utils_methods_FedRaDR.py`.

## Ready in this repository

- runnable FedRaDR training and smoke tests;
- deterministic D1/D2/P1/P2 partition generation, serialization, and validation;
- deterministic seed handling and fixed client-schedule generation;
- shared and FedRaDR-specific configuration records;
- the independent per-seed RTT records used in Appendix B and the reported
  controlled-participation, forgetting, and missing-class records in Appendix C;
- partition identifiers and expected SHA-256 prefixes stated in the paper;
- environment capture plus RTX 4090 and Jetson power-sampling utilities;
- raw-log CSV export and release-artifact audit utilities;
- an explicit artifact manifest with verification status.
