"""FedRaDR experiment components built on the official FedDC codebase."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .objective import FedRaDRConfig, FedRaDRLocalTrainer

__all__ = ["FedRaDRConfig", "FedRaDRLocalTrainer"]


def __getattr__(name: str):
    if name in __all__:
        from .objective import FedRaDRConfig, FedRaDRLocalTrainer

        return {"FedRaDRConfig": FedRaDRConfig, "FedRaDRLocalTrainer": FedRaDRLocalTrainer}[name]
    raise AttributeError(name)
