import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from nca.model import NCA
from nca.data import make_seed

model = NCA(n_channels=16)
seed = make_seed(n_channels=16, size=40, batch_size=1)

out = model(seed, steps=10, training=False)

print(f"output shape: {out.shape}")                                    # [1, 16, 40, 40]
print(f"alive cells:  {(out[0, 3] > 0.1).sum().item()}")              # should be > 0, likely small
print(f"center changed: {not torch.equal(seed[0, :, 20, 20], out[0, :, 20, 20])}")  # True
print(f"any NaN:      {torch.isnan(out).any().item()}")                # False
