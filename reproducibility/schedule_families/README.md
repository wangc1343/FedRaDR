# Controlled schedule families

The JSON schedules under `../schedules/` predate the family metadata fields and
are retained byte-for-byte for provenance. The `c100-r1000-seed*.permutation_hashes.json`
sidecars are deterministic hashes of the shared round permutations regenerated
from the same NumPy seed. They document the intended C100-P10/P5/P2 nesting;
they do not turn an old schedule into recovered historical evidence. To create
new files with the family fields embedded, use:

```text
python scripts/make_controlled_schedules.py --clients 100 \
  --participants 10 5 2 --rounds 1000 --seed 20 \
  --output-dir reproducibility/schedules --name-prefix c100
```
