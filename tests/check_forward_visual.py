import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
from nca.model import NCA
from nca.data import make_seed

model = NCA(n_channels=16)

# temporarily use random weights so something actually happens
torch.nn.init.normal_(model.fc2.weight, std=0.1)
torch.nn.init.zeros_(model.fc2.bias)

seed = make_seed(n_channels=16, size=40, batch_size=1)
state = seed.clone()

snapshots = []
alive_counts = []

with torch.no_grad():
    for step in range(100):
        state = model(state, steps=1, training=False)
        alive = (state[0, 3] > 0.1).sum().item()
        alive_counts.append(alive)
        if step in (0, 9, 24, 49, 99):
            snapshots.append((step + 1, state[0, 3].numpy().copy()))

fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(14, 3))

# alive cell count over time
axes[0].plot(alive_counts)
axes[0].set_title("alive cells over steps")
axes[0].set_xlabel("step")
axes[0].set_ylabel("count")

for ax, (step, alpha) in zip(axes[1:], snapshots):
    ax.imshow(alpha, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"step {step}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/check_forward_visual.png", dpi=100)
plt.show()
print("saved to outputs/check_forward_visual.png")
