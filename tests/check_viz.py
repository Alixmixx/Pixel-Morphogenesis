import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from nca.model import NCA
from nca.visualize import generate_growth_frames, save_gif

N_CHANNELS = 32
GRID_SIZE = 56
SEED_LOC = (28, 28)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = NCA(n_channels=N_CHANNELS).to(device)
model.load_state_dict(torch.load("outputs/nca_pool.pt", map_location=device))
model.eval()

print("generating frames...")
frames = generate_growth_frames(model, n_channels=N_CHANNELS, size=GRID_SIZE,
                                 n_steps=200, device=device, seed_loc=SEED_LOC)
print(f"got {len(frames)} frames, shape {frames[0].shape}")

save_gif(frames, "outputs/growth.gif", fps=30, scale=4)
print("saved outputs/growth.gif")
