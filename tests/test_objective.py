import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedradr.diagnostics import serialized_state_dict_bytes
from fedradr.objective import (
    FedRaDRConfig,
    FedRaDRLocalTrainer,
    normalized_gate,
    stage_schedule,
    teacher_confidence,
)
from fedradr.reproducibility import make_client_schedule


def test_reliability_components_are_bounded():
    logits = torch.tensor([[9.0, -1.0], [0.0, 0.0]])
    confidence = teacher_confidence(logits, temperature=1.0)
    gate = normalized_gate(torch.tensor([0.0, 2.0]), torch.tensor(1.0), 1e-8)
    assert torch.all((0 <= confidence) & (confidence <= 1))
    assert torch.all((0 <= gate) & (gate <= 1))
    assert 0.5 <= stage_schedule(2, 10) <= 1.0


def test_schedule_is_reproducible_and_balanced_per_round():
    first = make_client_schedule(20, 4, 5, seed=20)
    second = make_client_schedule(20, 4, 5, seed=20)
    assert first == second
    assert all(len(set(round_clients)) == 4 for round_clients in first)


def test_local_trainer_runs_all_normalization_modes():
    inputs = torch.randn(12, 4)
    targets = torch.randint(0, 3, (12,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    for mode in ("exact", "minibatch", "ema"):
        torch.manual_seed(7)
        global_model = nn.Linear(4, 3)
        local_model = nn.Linear(4, 3)
        local_model.load_state_dict(global_model.state_dict())
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.05)
        trainer = FedRaDRLocalTrainer(FedRaDRConfig(normalization=mode))
        _, metrics = trainer.train(
            local_model,
            global_model,
            loader,
            round_index=0,
            total_rounds=5,
            local_epochs=1,
            optimizer=optimizer,
        )
        assert len(metrics) == 3
        assert all(0 <= row["mean_gate"] <= 1 for row in metrics)
        assert serialized_state_dict_bytes(local_model) > 0
