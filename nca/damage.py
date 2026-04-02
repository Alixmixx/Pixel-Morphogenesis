import torch
import numpy as np


def apply_damage(state, fraction_range=(0.25, 0.5)):
    """Zero out a random rectangle on each sample in the batch."""
    B, C, H, W = state.shape
    damaged = state.clone()

    for i in range(B):
        frac = np.random.uniform(*fraction_range)
        h = int(H * np.sqrt(frac))
        w = int(W * np.sqrt(frac))
        y = np.random.randint(0, H - h + 1)
        x = np.random.randint(0, W - w + 1)
        damaged[i, :, y:y+h, x:x+w] = 0.0

    return damaged
