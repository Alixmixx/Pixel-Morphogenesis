import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
from nca.model import NCA
from nca.data import make_seed

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = NCA(n_channels=16).to(device)
model.load_state_dict(torch.load("outputs/nca_basic.pt", map_location=device))
model.eval()

seed = make_seed(16, 40, batch_size=1).to(device)
state = seed.clone()

snapshots = []
with torch.no_grad():
    for step in range(200):
        state = model(state, steps=1, training=False)
        if step in (31, 63, 95, 127, 199):
            rgba = state[0, :4].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            snapshots.append((step + 1, rgba))

fig, axes = plt.subplots(1, len(snapshots), figsize=(14, 3))
for ax, (step, rgba) in zip(axes, snapshots):
    ax.imshow(rgba)
    ax.set_title(f"step {step}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/check_stability.png", dpi=100)
plt.show()
print("saved to outputs/check_stability.png")
