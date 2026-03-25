import torch
import torch.nn as nn
import torch.nn.functional as F


class NCA(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        self.n_channels = n_channels

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.T
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        # stack into [3*C, 1, 3, 3] for depthwise conv — one filter per channel
        kernel = torch.stack([identity, sobel_x, sobel_y]).unsqueeze(1)  # [3, 1, 3, 3]
        kernel = kernel.repeat(n_channels, 1, 1, 1)                      # [3*C, 1, 3, 3]
        self.perception_kernel: torch.Tensor
        self.register_buffer("perception_kernel", kernel)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise conv: each channel filtered independently
        return F.conv2d(x, self.perception_kernel, padding=1, groups=self.n_channels)
