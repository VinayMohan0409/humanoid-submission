"""Per-layer activation and gradient statistics, and a one-word health verdict."""
import json
import statistics

from torch import nn


class Diagnostics:
    """Collects activation stats with forward hooks during the real training step and gradient stats after backward."""

    def __init__(self, model):
        self.active = False
        self.activations = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.GELU)):
                module.register_forward_hook(self._record(name))

    def _record(self, name):
        """Return a hook that stores mean, std and, for GELU outputs, the fraction of dead units."""

        def hook(module, inputs, output):
            if self.active:
                out = output.detach().float()
                dead = None
                if isinstance(module, nn.GELU):
                    dead = (out.abs().amax(dim=(0, 1)) < 1e-3).float().mean().item()
                self.activations[name] = {"mean": out.mean().item(), "std": out.std().item(), "dead_fraction": dead}

        return hook

    def gradient_stats(self, model):
        """Return mean, std and norm of every weight matrix's gradient. Call after backward and before clipping."""
        return {
            name: {"mean": p.grad.mean().item(), "std": p.grad.std().item(), "norm": p.grad.norm().item()}
            for name, p in model.named_parameters()
            if p.ndim >= 2 and p.grad is not None
        }

    def diagnose(self, gradients):
        """Return 'dead_neurons', 'exploding_gradients', 'vanishing_gradients' or 'healthy', checked in that order."""
        if any(a["dead_fraction"] is not None and a["dead_fraction"] > 0.5 for a in self.activations.values()):
            return "dead_neurons"
        norms = [g["norm"] for g in gradients.values()]
        if any(n != n or n > 1000 for n in norms):
            return "exploding_gradients"
        if any(n < 1e-3 * statistics.median(norms) for n in norms):
            return "vanishing_gradients"
        return "healthy"

    def log(self, model, step, path):
        """Append the full per-layer record to a jsonl file and print one summary line."""
        gradients = self.gradient_stats(model)
        verdict = self.diagnose(gradients)
        record = {"step": step, "verdict": verdict, "activations": self.activations, "gradients": gradients}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"step {step}: {verdict}")
