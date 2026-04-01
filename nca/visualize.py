import torch
import numpy as np
from PIL import Image
from nca.data import make_seed


def to_rgb(x):
    """Composite RGBA onto white background."""
    rgb, a = x[:3], x[3:4].clamp(0, 1)
    return (1.0 - a + rgb).clamp(0, 1)


def generate_growth_frames(model, n_channels, size, n_steps=200, device="cpu",
                            seed_loc=None):
    state = make_seed(n_channels, size, batch_size=1).to(device)
    frames = []
    with torch.no_grad():
        for _ in range(n_steps):
            state = model(state, steps=1, training=False, seed_loc=seed_loc)
            rgb = to_rgb(state[0]).permute(1, 2, 0).cpu().numpy()
            frames.append((rgb * 255).astype(np.uint8))
    return frames


def save_gif(frames, path, fps=30, scale=4):
    images = []
    for f in frames:
        img = Image.fromarray(f)
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        images.append(img)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
