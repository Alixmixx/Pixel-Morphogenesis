import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
from nca.model import NCA
from nca.data import load_target

if len(sys.argv) < 2:
    print("usage: python tests/check_perception.py targets/your_image.png")
    sys.exit(1)

target = load_target(sys.argv[1], size=40)  # [4, 40, 40]
x = target.unsqueeze(0)                     # [1, 4, 40, 40]

model = NCA(n_channels=4)
perception = model.perceive(x)             # [1, 12, 40, 40]
print(f"input shape:      {x.shape}")
print(f"perception shape: {perception.shape}")  # should be [1, 12, 40, 40]

# channels 0-3: identity (should look like original)
# channels 4-7: sobel_x (horizontal edges)
# channels 8-11: sobel_y (vertical edges)
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

titles = ["identity", "sobel_x", "sobel_y"]
for row, (title, offset) in enumerate(zip(titles, [0, 4, 8])):
    for col in range(4):
        ch = perception[0, offset + col].detach().numpy()
        axes[row, col].imshow(ch, cmap="RdBu", vmin=-1, vmax=1)
        axes[row, col].set_title(f"{title} ch{col}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("outputs/check_perception.png", dpi=100)
plt.show()
print("saved to outputs/check_perception.png")
