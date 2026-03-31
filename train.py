import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import matplotlib.pyplot as plt
from nca.model import NCA
from nca.data import load_target, make_seed
from nca.pool import SamplePool

TARGET_PATH = "targets/skull.png"
GRID_SIZE = 40
N_CHANNELS = 16
BATCH_SIZE = 8
POOL_SIZE = 1024
LR = 2e-3
TOTAL_STEPS = 8000
SAVE_EVERY = 500

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

target = load_target(TARGET_PATH, size=GRID_SIZE).to(device)
model = NCA(n_channels=N_CHANNELS).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.3)
pool = SamplePool(n_channels=N_CHANNELS, size=GRID_SIZE, pool_size=POOL_SIZE)

Path("outputs").mkdir(exist_ok=True)

for step in range(1, TOTAL_STEPS + 1):
    batch, indices = pool.sample(BATCH_SIZE)
    batch = batch.to(device)

    n_steps = torch.randint(64, 96, (1,)).item()
    out = model(batch, steps=n_steps, training=True)

    per_sample_loss = ((out[:, :4] - target) ** 2).mean(dim=[1, 2, 3])
    loss = per_sample_loss.mean()

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    sched.step()

    pool.update(indices, out)
    pool.replace_worst(indices, per_sample_loss)

    if step % 100 == 0:
        lr = sched.get_last_lr()[0]
        print(f"step {step:>5}  loss {loss.item():.4f}  lr {lr:.2e}")

    if step % SAVE_EVERY == 0:
        with torch.no_grad():
            sample = model(make_seed(N_CHANNELS, GRID_SIZE, 1).to(device), steps=64, training=False)
        rgba = sample[0, :4].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        plt.imsave(f"outputs/pool_step_{step:05d}.png", rgba)

torch.save(model.state_dict(), "outputs/nca_pool.pt")
print("done — model saved to outputs/nca_pool.pt")
