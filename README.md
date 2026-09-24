# FedRaDR

Reproducible experiment code for **FedRaDR: Reliability-Aware Distillation and
Regularization for Non-IID Federated Learning**. This repository is derived
from the official MIT-licensed [FedDC implementation](https://github.com/gaoliang13/FedDC).

The original FedDC scripts remain at the repository root for provenance. New
FedRaDR components live in `fedradr/` and use current PyTorch APIs.

## Implemented experimental requirements

- reliability-aware weighted distillation with detached sample weights;
- exact client mean, mini-batch mean, and EMA forgetting normalization;
- deterministic seeds and reusable client-participation schedules;
- mean forgetting gap (MFG) and high-forgetting ratio (HFR);
- locally absent-class probability-mass retention;
- exact serialized model payload measurement;
- elapsed-time and peak CUDA-memory profiling hooks;
- structured JSONL logs and unit tests.

The repository intentionally does not implement or claim the manuscript's
theoretical convergence results. It covers the experimental revision only.

## Installation

```bash
python -m pip install -e ".[test]"
pytest
```

For the current dependency set without installing the package first:

```bash
python -m pip install -r requirements-fedradr.txt
python scripts/smoke_fedradr.py --output smoke_output
```

The smoke command executes the complete FedDC state update, FedRaDR local
objective, server aggregation, evaluation, JSONL logging, and checkpoint save.

## Full experiment entry point

`scripts/run_fedradr.py` consumes a compressed NPZ containing `clnt_x`,
`clnt_y`, `tst_x`, `tst_y`, and `dataset`. It supports the paper's
ResNet-18-GN and MobileNetV2-GN configurations:

```bash
python scripts/run_fedradr.py \
  --data partitions/cifar10-d1.npz \
  --output results/CIFAR10/D1/FedRaDR/20 \
  --model resnet18 --rounds 1000 --clients-per-round 10 \
  --seed 20 --normalization exact
```

The runner saves the fixed client schedule, complete configuration, per-round
metrics, and the latest global model. Full FedDC drift and gradient state is
large for ResNet-18 and is therefore opt-in via `--checkpoint-every N`; these
histories remain in CPU memory during training rather than consuming GPU memory.

CUDA, driver, PyTorch, torchvision, device precision, batch size, and profiler
settings must be recorded with every reported resource result. Energy remains
blank unless a real hardware counter is available; simulated energy numbers
must not be reported as device measurements.

## Reproducibility assets

Generate one client schedule and reuse the exact JSON file for every method:

```bash
python -m scripts.make_reproducibility_assets \
  --clients 100 --clients-per-round 10 --rounds 1000 --seed 20 \
  --output reproducibility/schedules/c100-p10-seed20.json
```

Use seeds `20`, `21`, and `22` for every compared method. Store raw records in
`results/<dataset>/<partition>/<method>/<seed>/metrics.jsonl`; do not commit
datasets, checkpoints, or large raw logs.

## Normalization guidance

| Mode | Use case | Trade-off |
|---|---|---|
| `exact` | Moderate local datasets and accuracy-first evaluation | Full-client student pre-pass per local stage; highest cost |
| `minibatch` | Memory- or time-constrained clients | No full-client reference; noisier normalization |
| `ema` | Throughput-oriented or edge evaluation | Lowest reference cost; depends on registered EMA decay |

For persistent teacher caching, the dataset must expose stable sample indices.
Caching by shuffled batch position is incorrect and is deliberately not used.

## Required reporting

Each experiment must publish the partition file/hash, client schedule, complete
configuration, code commit, per-seed logs, hardware description, and raw rounds
to target. If a baseline does not reach the target within the budget, speedup is
reported as a lower bound (for example, `>5.26x`).

## FedDC upstream documentation

We provide code to run FedDC, FedAvg, 
[FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w), 
[Scaffold](https://openreview.net/pdf?id=B7v4QMR6Z9w), and [FedProx](https://arxiv.org/abs/1812.06127) methods.


## Prerequisite
* Install the libraries listed in requirements.txt
    ```
    pip install -r requirements.txt
    ```

## Datasets preparation
**We give datasets for the benchmark, including CIFAR10, CIFAR100, MNIST, EMNIST-L and the synthetic dataset.**




You can obtain the datasets when you first time run the code on CIFAR10, CIFAR100, MNIST, synthetic datasets.
EMNIST needs to be downloaded from this [link](https://www.nist.gov/itl/products-and-services/emnist-dataset).


For example, you can follow the following steps to run the experiments:

```python example_code_mnist.py```
```python example_code_cifar10.py```
```python example_code_cifar100.py```

1. Run the following script to run experiments on the MNIST dataset for all above methods:
    ```
    python example_code_mnist.py
    ```
2. Run the following script to run experiments on CIFAR10 for all above methods:
    ```
    python example_code_cifar10.py
    ```
3. Run the following script to run experiments on CIFAR100 for all above methods:
    ```
    python example_code_cifar10.py
    ```
4. To show the convergence plots, we use the tensorboardX package. As an example to show the results which stored in "./Folder/Runs/CIFAR100_100_23_iid_":
    ```
    tensorboard --logdir=./Folder/Runs/CIFAR10_100_23_iid
    ```
5. Get the url, and then enter the url in to the web browser, for example "http://localhost:6006/".

   
## Generate IID and Dirichlet distributions:
Modify the DatasetObject() function in the example code.
CIFAR-10 IID, 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=17, rule='iid', unbalanced_sgm=0, data_path=data_path)
```
CIFAR-10 Dirichlet (0.3), 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=47, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
```

    
## FedDC 
The FedDC method is implemented in ```utils_methods_FedDC.py```. The baseline methods are stored in ```utils_methods.py```.

### Citation

```
@inproceedings{
gao2022federated,
title={FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction},
author={Liang Gao and Huazhu Fu and Li Li and Yingwen Chen and Ming Xu and Cheng-Zhong Xu},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
year={2022}
}
```
