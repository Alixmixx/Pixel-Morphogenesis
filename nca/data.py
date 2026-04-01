import numpy as np
from PIL import Image
import torch


def load_target(path, target_size=40, pad=8):
    """Load RGBA image, premultiply alpha, pad to create breathing room."""
    img = Image.open(path).convert("RGBA").resize((target_size, target_size), Image.Resampling.LANCZOS)
    img = np.float32(np.array(img)) / 255.0
    img[..., :3] *= img[..., 3:]  # premultiply RGB by alpha
    img = img.transpose(2, 0, 1)  # [4, H, W]
    img = np.pad(img, ((0, 0), (pad, pad), (pad, pad)))  # [4, H+2*pad, W+2*pad]
    return torch.from_numpy(img)


def make_seed(n_channels, size, batch_size=1):
    seed = torch.zeros(batch_size, n_channels, size, size)
    mid = size // 2
    seed[:, 3:, mid, mid] = 1.0  # alpha + all hidden channels = 1 at center
    return seed
