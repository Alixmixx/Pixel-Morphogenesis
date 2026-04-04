import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from nca.model import NCA


def export_model(pt_path, json_path, n_channels=32, hidden_size=128, grid_size=56):
    model = NCA.from_pretrained(pt_path, n_channels=n_channels, hidden_size=hidden_size)
    sd = model.state_dict()

    # only export the learned layers (update.1 and update.3)
    # update.0 is the fixed Sobel perception layer — hardcoded in JS
    data = {
        "n_channels": n_channels,
        "hidden_size": hidden_size,
        "grid_size": grid_size,
        "fc1_weight": {
            "shape": list(sd["update.1.weight"].shape[:2]),
            "data": sd["update.1.weight"].squeeze().flatten().tolist(),
        },
        "fc1_bias": {
            "shape": [sd["update.1.bias"].shape[0]],
            "data": sd["update.1.bias"].tolist(),
        },
        "fc2_weight": {
            "shape": list(sd["update.3.weight"].shape[:2]),
            "data": sd["update.3.weight"].squeeze().flatten().tolist(),
        },
        "fc2_bias": {
            "shape": [sd["update.3.bias"].shape[0]],
            "data": sd["update.3.bias"].tolist(),
        },
    }

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f)

    size_kb = Path(json_path).stat().st_size / 1024
    print(f"exported {pt_path} -> {json_path} ({size_kb:.0f} KB)")


def validate_export(pt_path, json_path, n_channels=32, hidden_size=128):
    """Compare one forward step between PyTorch and reconstructed weights."""
    model = NCA.from_pretrained(pt_path, n_channels=n_channels, hidden_size=hidden_size)

    with open(json_path) as f:
        data = json.load(f)

    # reconstruct and compare
    fc1_w = torch.tensor(data["fc1_weight"]["data"]).reshape(data["fc1_weight"]["shape"][0], data["fc1_weight"]["shape"][1], 1, 1)
    fc1_b = torch.tensor(data["fc1_bias"]["data"])
    fc2_w = torch.tensor(data["fc2_weight"]["data"]).reshape(data["fc2_weight"]["shape"][0], data["fc2_weight"]["shape"][1], 1, 1)
    fc2_b = torch.tensor(data["fc2_bias"]["data"])

    assert torch.allclose(model.update[1].weight, fc1_w), "fc1 weight mismatch"
    assert torch.allclose(model.update[1].bias, fc1_b), "fc1 bias mismatch"
    assert torch.allclose(model.update[3].weight, fc2_w), "fc2 weight mismatch"
    assert torch.allclose(model.update[3].bias, fc2_b), "fc2 bias mismatch"
    print("validation passed — weights match exactly")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/nca_pool.pt")
    parser.add_argument("--output", default="docs/models/skull.json")
    args = parser.parse_args()

    export_model(args.input, args.output)
    validate_export(args.input, args.output)
