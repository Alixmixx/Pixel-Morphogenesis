"""
Generate all skull assets after training:
  - outputs/skull_result.png
  - outputs/growth.gif
  - outputs/regeneration.gif
  - assets/skull_result.png
  - assets/growth.gif
  - assets/regeneration.gif
  - docs/models/skull.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from nca.model import NCA
from nca.data import make_seed
from nca.visualize import generate_growth_frames, save_gif, to_rgb
from nca.damage import apply_damage
from export import export_model, validate_export

MODEL_PATH  = "outputs/nca_pool.pt"
TARGET_PATH = "targets/skull.png"
N_CHANNELS  = 32
HIDDEN_SIZE = 128
GRID_SIZE   = 56
SEED_LOC    = (GRID_SIZE // 2, GRID_SIZE // 2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

model = NCA.from_pretrained(MODEL_PATH, n_channels=N_CHANNELS, hidden_size=HIDDEN_SIZE)
model = model.to(device)
model.eval()

Path("outputs").mkdir(exist_ok=True)
Path("assets").mkdir(exist_ok=True)

# --- result snapshot ---
with torch.no_grad():
    seed = make_seed(N_CHANNELS, GRID_SIZE, 1).to(device)
    out = model(seed, steps=160, training=False, seed_loc=SEED_LOC)

rgba = out[0, :4].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
plt.imsave("outputs/skull_result.png", rgba)
plt.imsave("assets/skull_result.png", rgba)
print("saved skull_result.png")

# --- growth GIF (200 steps) ---
print("generating growth GIF...")
growth_frames = generate_growth_frames(
    model, N_CHANNELS, GRID_SIZE, n_steps=200, device=device, seed_loc=SEED_LOC
)
save_gif(growth_frames, "outputs/growth.gif", fps=30, scale=4)
save_gif(growth_frames, "assets/growth.gif",  fps=30, scale=4)
print("saved growth.gif")

# --- regeneration GIF: grow → damage → heal ---
print("generating regeneration GIF...")
regen_frames = []

with torch.no_grad():
    state = make_seed(N_CHANNELS, GRID_SIZE, 1).to(device)

    # grow for 150 steps
    for _ in range(150):
        state = model(state, steps=1, training=False, seed_loc=SEED_LOC)
        rgb = to_rgb(state[0]).permute(1, 2, 0).cpu().numpy()
        regen_frames.append((rgb * 255).astype(np.uint8))

    # damage: zero out a central rectangle
    damaged = state.clone()
    h, w = GRID_SIZE, GRID_SIZE
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    damaged[:, :, y0:y1, x0:x1] = 0.0
    state = damaged

    # hold damage frame for 20 frames
    rgb = to_rgb(state[0]).permute(1, 2, 0).cpu().numpy()
    for _ in range(20):
        regen_frames.append((rgb * 255).astype(np.uint8))

    # regenerate for 150 steps
    for _ in range(150):
        state = model(state, steps=1, training=False, seed_loc=SEED_LOC)
        rgb = to_rgb(state[0]).permute(1, 2, 0).cpu().numpy()
        regen_frames.append((rgb * 255).astype(np.uint8))

save_gif(regen_frames, "outputs/regeneration.gif", fps=30, scale=4)
save_gif(regen_frames, "assets/regeneration.gif",  fps=30, scale=4)
print("saved regeneration.gif")

# --- export model weights for browser ---
export_model(MODEL_PATH, "docs/models/skull.json", N_CHANNELS, HIDDEN_SIZE, GRID_SIZE)
validate_export(MODEL_PATH, "docs/models/skull.json", N_CHANNELS, HIDDEN_SIZE)

print("\nAll assets generated.")
