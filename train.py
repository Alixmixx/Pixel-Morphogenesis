import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import matplotlib.pyplot as plt
from nca.model import NCA
from nca.data import load_target, make_seed

TARGET_PATH = "targets/emoji.png"
GRID_SIZE = 40
N_CHANNELS = 16
BATCH_SIZE = 8
N_STEPS = 64
LR = 2e-3
TOTAL_STEPS = 2000
SAVE_EVERY = 500

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

target = load_target(TARGET_PATH, size=GRID_SIZE).to(device)  # [4, H, W]
model = NCA(n_channels=N_CHANNELS).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)

Path("outputs").mkdir(exist_ok=True)

for step in range(1, TOTAL_STEPS + 1):
    seed = make_seed(N_CHANNELS, GRID_SIZE, batch_size=BATCH_SIZE).to(device)
    out = model(seed, steps=N_STEPS, training=True)
    loss = ((out[:, :4] - target) ** 2).mean()

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    if step % 100 == 0:
        print(f"step {step:>5}  loss {loss.item():.4f}")

    if step % SAVE_EVERY == 0:
        with torch.no_grad():
            sample = model(
                make_seed(N_CHANNELS, GRID_SIZE, 1).to(device),
                steps=N_STEPS,
                training=False,
            )
        rgba = sample[0, :4].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        plt.imsave(f"outputs/step_{step:05d}.png", rgba)

torch.save(model.state_dict(), "outputs/nca_basic.pt")
print("done — model saved to outputs/nca_basic.pt")
