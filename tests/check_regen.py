import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from nca.model import NCA
from nca.damage import apply_damage
from nca.visualize import generate_growth_frames, to_rgb, save_gif
from nca.data import make_seed
import numpy as np

N_CHANNELS = 32
GRID_SIZE = 56
SEED_LOC = (28, 28)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = NCA(n_channels=N_CHANNELS).to(device)
model.load_state_dict(torch.load("outputs/nca_pool.pt", map_location=device))
model.eval()

state = make_seed(N_CHANNELS, GRID_SIZE, batch_size=1).to(device)
frames = []

with torch.no_grad():
    # grow for 150 steps
    for _ in range(150):
        state = model(state, steps=1, training=False, seed_loc=SEED_LOC)
        rgb = to_rgb(state[0]).permute(1, 2, 0).cpu().numpy()
        frames.append((rgb * 255).astype(np.uint8))

    # damage the center quarter
    state = apply_damage(state, fraction_range=(0.4, 0.5))
    rgb = to_rgb(state[0]).permute(1, 2, 0).cpu().numpy()
    for _ in range(10):  # hold the damaged frame for a beat
        frames.append((rgb * 255).astype(np.uint8))

    # regenerate for 100 steps
    for _ in range(100):
        state = model(state, steps=1, training=False, seed_loc=SEED_LOC)
        rgb = to_rgb(state[0]).permute(1, 2, 0).cpu().numpy()
        frames.append((rgb * 255).astype(np.uint8))

print(f"got {len(frames)} frames")
save_gif(frames, "outputs/regeneration.gif", fps=30, scale=4)
print("saved outputs/regeneration.gif")
