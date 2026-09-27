# Per-round training logs

Use one CSV per dataset, setting, partition, method, and seed:

```text
log_{dataset}_N{clients}P{participation_percent}_{partition}_{method}_seed{seed}.csv
```

Examples:

```text
log_cifar10_N100P10_D1_FedRaDR_seed20.csv
log_cifar10_N500P2_P1_FedDyn_seed22.csv
log_tinyimagenet_N100P10_D1_FedSKC_seed21.csv
```

Required columns:

```csv
round,test_accuracy,train_objective,mfg,hfr_tau,downlink_bytes,uplink_bytes,cumulative_serialized_bytes,single_device_elapsed_seconds
```

- `test_accuracy` is a fraction in `[0,1]`.
- `train_objective` is the complete optimized local objective aggregated with
  sample weights; it is not cross-entropy alone.
- `hfr_tau` uses the preregistered fixed threshold for the matched dataset and
  partition. Leave it empty when the diagnostic is not evaluated.
- byte counts are measured serialized payloads, not parameter-count estimates.
- `single_device_elapsed_seconds` is serialized simulator compute time. Do not
  call it distributed wall-clock time.

Every CSV must have a sidecar `config.json` recording the code commit,
partition SHA-256, schedule SHA-256, model, optimizer, all hyperparameters,
device, precision, software versions, and target-accuracy definition.