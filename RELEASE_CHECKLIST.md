# Public release checklist

- [x] FedRaDR implementation integrated into the FedDC framework.
- [x] Unit and smoke tests.
- [x] Deterministic schedule generator.
- [x] Shared protocol and FedRaDR hyperparameters.
- [x] FedRaDR tuning candidates and selection record.
- [x] Appendix C machine-readable aggregate records.
- [x] Environment capture script.
- [x] NVIDIA raw power-trace sampler and measurement protocol.
- [x] Jetson raw power-trace sampler and timestamp-aware hardware summary.
- [x] Strict supplementary exporter with schema checks and SHA-256 manifest.
- [x] Create an author-owned public GitHub repository.
- [x] Set `origin` to `https://github.com/wangc1343/FedRaDR`.
- [ ] Recover and commit exact partition files with full SHA-256 hashes.
- [ ] Record whether `client_indices` refer to raw dataset order or the exact
      FedDC seed-shuffled order, and test the corresponding loader.
- [ ] Recover full per-round logs for every paper result.
- [ ] Recover raw RTX 4090 and Jetson power traces, or rerun the measurements.
- [x] Record the author-provided final baseline coefficients.
- [ ] Recover the baseline tuning grids and selection records, or state
      explicitly that original-paper defaults were used without a new sweep.
- [x] Add the final public URL to the manuscript and response letter.
- [ ] Tag the submitted version, for example `neucom-r2-artifact-v1`.
