import pytest
import torch

from config import FEATURE_DIM, LATENT_DIM, N_QUERIES, N_TOKENS
from models.latent_action import LatentActionEncoder, kl_divergence
from models.world_model import WorldModel
from train import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    def build():
        model = torch.nn.ModuleDict({"encoder": LatentActionEncoder(), "world_model": WorldModel()})
        return model, torch.optim.AdamW(model.parameters())

    model, optimizer = build()
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    save_checkpoint(tmp_path / "c.pt", model, optimizer, 3, 40, 0.5, torch.Generator().manual_seed(0), "cpu")

    fresh, fresh_optimizer = build()
    assert load_checkpoint(tmp_path / "c.pt", fresh, fresh_optimizer, torch.Generator(), "cpu") == (4, 40, 0.5)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fresh.state_dict().values()))
    assert len(fresh_optimizer.state_dict()["state"]) == len(optimizer.state_dict()["state"]) > 0


def test_latent_action_shape_and_kl():
    frame = torch.randn(2, N_TOKENS, FEATURE_DIM)
    mean, logvar = LatentActionEncoder()(frame, frame)
    assert mean.shape == logvar.shape == (2, N_QUERIES, LATENT_DIM)
    assert torch.isfinite(kl_divergence(mean, logvar))


@pytest.mark.parametrize("mode", ["cross", "concat"])
def test_world_model_output_shape(mode, monkeypatch):
    monkeypatch.setattr("models.world_model.CONDITIONING", mode)
    frame = torch.randn(2, N_TOKENS, FEATURE_DIM)
    z = torch.randn(2, N_QUERIES, LATENT_DIM)
    assert WorldModel()(frame, z).shape == frame.shape
