import torch
import torch.nn as nn
import torch.nn.functional as F


class NCA(nn.Module):
    def __init__(self, n_channels=16, hidden_size=128):
        super().__init__()
        self.n_channels = n_channels

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.T
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        kernel = torch.stack([identity, sobel_x, sobel_y]).unsqueeze(1)
        kernel = kernel.repeat(n_channels, 1, 1, 1)
        self.perception_kernel: torch.Tensor
        self.register_buffer("perception_kernel", kernel)

        self.fc1 = nn.Conv2d(n_channels * 3, hidden_size, 1)
        self.fc2 = nn.Conv2d(hidden_size, n_channels, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)  # type: ignore[arg-type]

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.perception_kernel, padding=1, groups=self.n_channels)

    def update_rule(self, perception: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(perception)))

    def alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x: torch.Tensor, steps: int = 1, training: bool = True) -> torch.Tensor:
        for _ in range(steps):
            pre_alive = self.alive_mask(x)

            perception = self.perceive(x)
            update = self.update_rule(perception)

            if training:
                mask = (torch.rand(x[:, :1].shape, device=x.device) < 0.5).float()
                update = update * mask

            x = x + update

            post_alive = self.alive_mask(x)
            x = x * (pre_alive & post_alive).float()

        return x
