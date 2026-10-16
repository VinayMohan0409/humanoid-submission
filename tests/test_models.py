"""Shape, KL and checkpoint round-trip tests for the latent action model."""
import torch

from src.config import CHUNK, FEATURE_DIM, LATENT_DIM, N_QUERIES, N_TOKENS
from src.models.latent_action import LatentActionEncoder, encode_chunk, kl_divergence
from src.models.world_model import WorldModel
from src.training.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    def build():
        """Return a fresh encoder and world model with their optimizer."""
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


def test_chunk_encoder_shape_and_kl():
    frames = torch.randn(2, CHUNK + 1, N_TOKENS, FEATURE_DIM)
    mean, logvar = encode_chunk(LatentActionEncoder(), frames)
    assert mean.shape == logvar.shape == (2, CHUNK, N_QUERIES, LATENT_DIM)
    assert torch.isfinite(kl_divergence(mean, logvar))


def test_world_model_shape_and_causality():
    model = WorldModel().eval()
    for p in model.out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    frame, z = torch.randn(2, N_TOKENS, FEATURE_DIM), torch.randn(2, CHUNK, N_QUERIES, LATENT_DIM)
    out = model(frame, z)
    assert out.shape == (2, CHUNK, N_TOKENS, FEATURE_DIM)
    for j in range(CHUNK):
        changed = z.clone()
        changed[:, j] += 1
        diff = (model(frame, changed) - out).abs().amax(dim=(0, 2, 3))
        assert (diff[:j] == 0).all() and (diff[j:] > 0).all()
