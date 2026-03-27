import torch
from nca.data import make_seed


class SamplePool:
    def __init__(self, n_channels: int, size: int, pool_size: int = 1024):
        self.n_channels = n_channels
        self.size = size
        self.pool = make_seed(n_channels, size, batch_size=pool_size)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(len(self.pool))[:batch_size]
        return self.pool[indices].clone(), indices

    def update(self, indices: torch.Tensor, new_states: torch.Tensor) -> None:
        self.pool[indices] = new_states.detach().cpu()

    def replace_worst(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        worst = losses.argmax().item()
        self.pool[indices[worst]] = make_seed(self.n_channels, self.size, batch_size=1)[0]
