
# Minimal tests for SimCLRModel output shape and NTXentLoss similarity effect
import os
import torch
import types
import pytest

# Import SimCLRModel and NTXentLoss from main.py
import importlib.util
CURRENT_DIR = os.path.dirname(__file__)
MAIN_PATH = os.path.join(CURRENT_DIR, "main.py")
spec = importlib.util.spec_from_file_location("simclr_main", MAIN_PATH)
main_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_mod)
SimCLRModel = main_mod.SimCLRModel
NTXentLoss = main_mod.NTXentLoss

class DummyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.linear = torch.nn.Linear(8, hidden_size)
    def forward(self, pixel_values):
        # pixel_values: [B, C, H, W] -> flatten to [B, 8]
        B = pixel_values.shape[0]
        flat = pixel_values.view(B, -1)
        out = self.linear(flat[:, :8])
        return types.SimpleNamespace(last_hidden_state=out.unsqueeze(1))

def test_simclr_model_output_shape():
    backbone = DummyBackbone(hidden_size=16)
    model = SimCLRModel(backbone, projection_dim=10)
    x = torch.randn(4, 3, 2, 2)  # [batch, C, H, W] -> 4,3,2,2
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 10), f"Expected output shape (4, 10), got {out.shape}"

def test_ntxentloss_similarity():
    device = "cpu"
    batch_size = 4
    dim = 8
    # Similar vectors (identical)
    z1 = torch.randn(batch_size, dim)
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = z1.clone()
    loss_similar = NTXentLoss(device, batch_size, temperature=0.5)(z1, z2)
    # Dissimilar vectors (random)
    z3 = torch.randn(batch_size, dim)
    z3 = torch.nn.functional.normalize(z3, dim=1)
    loss_dissimilar = NTXentLoss(device, batch_size, temperature=0.5)(z1, z3)
    assert loss_similar < loss_dissimilar, f"Expected lower loss for similar vectors, got {loss_similar} !< {loss_dissimilar}"

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))