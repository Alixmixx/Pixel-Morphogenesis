import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from nca.model import NCA
from nca.data import load_target, make_seed
from nca.pool import SamplePool

TARGET_PATH  = "targets/skull.png"
TARGET_SIZE  = 40
PAD          = 8
GRID_SIZE    = TARGET_SIZE + 2 * PAD  # 56
N_CHANNELS   = 32
HIDDEN_SIZE  = 128
BATCH_SIZE   = 8
POOL_SIZE    = 1024
LR           = 2e-3
WEIGHT_DECAY = 3e-5
MILESTONES   = [3000, 6000, 9000]
GAMMA        = 0.2
TOTAL_STEPS  = 12000
SAVE_EVERY   = 500

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

target = load_target(TARGET_PATH, target_size=TARGET_SIZE, pad=PAD).to(device)
seed_loc = (GRID_SIZE // 2, GRID_SIZE // 2)
print(f"grid: {GRID_SIZE}x{GRID_SIZE}, target: {TARGET_SIZE}x{TARGET_SIZE}, seed: {seed_loc}")

model = NCA(n_channels=N_CHANNELS, hidden_size=HIDDEN_SIZE).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = MultiStepLR(optim, milestones=MILESTONES, gamma=GAMMA)
pool  = SamplePool(n_channels=N_CHANNELS, size=GRID_SIZE, pool_size=POOL_SIZE)

Path("outputs").mkdir(exist_ok=True)

for step in range(1, TOTAL_STEPS + 1):
    indices, batch = pool.sample(BATCH_SIZE)
    batch = batch.to(device)

    n_steps = np.random.randint(64, 108)
    out = model(batch, steps=n_steps, training=True, seed_loc=seed_loc)

    per_sample_loss = ((out[:, :4] - target) ** 2).view(BATCH_SIZE, -1).mean(-1)
    loss = per_sample_loss.mean()

    loss.backward()

    # gradient normalization to stabilize training
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data /= p.grad.data.norm() + 1e-8

    optim.step()
    optim.zero_grad()
    sched.step()

    pool.commit(indices, out, per_sample_loss)

    if step % 100 == 0:
        lr = optim.param_groups[0]["lr"]
        print(f"step {step:>5}  loss {loss.item():.4f}  lr {lr:.2e}")

    if step % SAVE_EVERY == 0:
        with torch.no_grad():
            seed = make_seed(N_CHANNELS, GRID_SIZE, 1).to(device)
            sample = model(seed, steps=80, training=False, seed_loc=seed_loc)
        rgba = sample[0, :4].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        plt.imsave(f"outputs/pool_step_{step:05d}.png", rgba)

torch.save(model.state_dict(), "outputs/nca_pool.pt")
print("done — model saved to outputs/nca_pool.pt")
