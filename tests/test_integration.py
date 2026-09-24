import json

import torch

from fedradr.objective import FedRaDRConfig
from scripts.smoke_fedradr import TinyClassifier, make_data
from utils_methods_FedRaDR import train_FedRaDR


def test_feddc_state_pipeline_writes_replayable_artifacts(tmp_path):
    torch.manual_seed(11)
    model = train_FedRaDR(
        make_data(),
        TinyClassifier,
        TinyClassifier(),
        client_schedule=[[0, 1], [2, 3]],
        output_dir=tmp_path,
        config=FedRaDRConfig(normalization="exact"),
        learning_rate=0.05,
        batch_size=8,
        local_epochs=1,
        alpha=0.01,
        weight_decay=0.0,
        seed=20,
        device="cpu",
        checkpoint_every=2,
    )

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    checkpoint = torch.load(tmp_path / "state_round_2.pt", map_location="cpu", weights_only=False)
    assert len(records) == 2
    assert checkpoint["round"] == 1
    assert checkpoint["parameter_drifts"].shape[0] == 4
    assert checkpoint["gradient_states"].shape[0] == 5
    assert torch.isfinite(checkpoint["cloud_params"]).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
