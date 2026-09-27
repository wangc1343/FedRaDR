$ErrorActionPreference = 'Stop'
# Requires labels/<dataset>_train_labels.npy
python scripts/build_reproducibility_bundle.py generate-partitions --root reproducibility --labels-root labels
