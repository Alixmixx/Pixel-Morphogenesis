import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NCA(nn.Module):
    def __init__(self, n_channels=32, hidden_size=128, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.dropout = dropout

        # perceive + update combined in one sequential block
        self.update = nn.Sequential(
            nn.Conv2d(n_channels, 3 * n_channels, 3, padding=1, groups=n_channels, bias=False),
            nn.Conv2d(3 * n_channels, hidden_size, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, n_channels, 1),
        )
        # zero-init last layer so initial updates do nothing
        self.update[-1].weight.data *= 0

        # fix first layer to Sobel filters
        identity = np.outer([0, 1, 0], [0, 1, 0])
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
        kernel = np.stack([identity, dx, dx.T], axis=0)
        kernel = np.tile(kernel, [n_channels, 1, 1])
        self.update[0].weight.data[...] = torch.from_numpy(kernel).float()[:, None, :, :]
        self.update[0].weight.requires_grad = False

    def alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x: torch.Tensor, steps: int = 1, training: bool = True,
                seed_loc: tuple[int, int] | None = None) -> torch.Tensor:
        for _ in range(steps):
            pre_alive = self.alive_mask(x)

            update_mask = torch.rand(*x.shape, device=x.device) > self.dropout
            x = x + update_mask.float() * self.update(x)

            post_alive = self.alive_mask(x)
            x = x * pre_alive.float() * post_alive.float()

            # keep the seed pixel alive — prevents collapse
            if seed_loc is not None:
                x[..., 3, seed_loc[0], seed_loc[1]] = 1.0

        return x
