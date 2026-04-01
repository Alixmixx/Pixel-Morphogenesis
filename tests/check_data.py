import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from nca.data import load_target, make_seed

if len(sys.argv) < 2:
    print("usage: python check_data.py targets/your_image.png")
    sys.exit(1)

target = load_target(sys.argv[1], size=40)
print(f"target shape: {target.shape}, min: {target.min():.3f}, max: {target.max():.3f}")

seed = make_seed(n_channels=16, size=40)
print(f"seed shape:   {seed.shape}")
print(f"alive cells:  {(seed[0, 3] > 0).sum().item()} (should be 1)")

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].imshow(target.permute(1, 2, 0).numpy())
axes[0].set_title("target (RGBA)")
axes[0].axis("off")

axes[1].imshow(target[:3].permute(1, 2, 0).numpy())
axes[1].set_title("target (RGB only)")
axes[1].axis("off")

# seed alpha channel — should be one white dot in the center
axes[2].imshow(seed[0, 3].numpy(), cmap="gray", vmin=0, vmax=1)
axes[2].set_title("seed (alpha channel)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("outputs/check_data.png", dpi=100)
plt.show()
print("saved to outputs/check_data.png")
