from PIL import Image
import torch
import torchvision.transforms as T


def load_target(path, size=40):
    img = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    return T.ToTensor()(img)  # [4, H, W], values in [0, 1]


def make_seed(n_channels, size, batch_size=1):
    seed = torch.zeros(batch_size, n_channels, size, size)
    # single alive pixel in the center — alpha channel is index 3
    seed[:, 3, size // 2, size // 2] = 1.0
    return seed
