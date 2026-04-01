import numpy as np
import torch
from nca.data import make_seed


class SamplePool:
    def __init__(self, n_channels: int, size: int, pool_size: int = 1024):
        self.n_channels = n_channels
        self.size = size
        self.pool = make_seed(n_channels, size, batch_size=pool_size)

    def sample(self, batch_size: int) -> tuple[np.ndarray, torch.Tensor]:
        indices = np.random.randint(len(self.pool), size=batch_size)
        return indices, self.pool[indices].clone()

    def commit(self, indices: np.ndarray, states: torch.Tensor, losses: torch.Tensor) -> None:
        # replace worst sample with a fresh seed
        worst = torch.argsort(losses, descending=True)[:max(1, int(0.15 * len(losses)))]
        states[worst] = make_seed(self.n_channels, self.size, batch_size=len(worst)).to(states.device)
        self.pool[indices] = states.detach().cpu()
