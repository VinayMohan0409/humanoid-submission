"""Collect the ablation runs' epoch-10 validation and training numbers into ablations/results.csv and results.md."""
import csv
import json

from src.config import ROOT

METRICS_DIR = ROOT / "ablations" / "metrics"
EPOCH = 9  # the 10th epoch, so runs of different lengths compare at the same point
RUNS = [  # (label, run folder, queries, dims, dropout)
    ("4x8 baseline", "det_b32", 4, 8, 0.0),
    ("2x8", "bn_q2_d8", 2, 8, 0.0),
    ("4x4", "bn_q4_d4", 4, 4, 0.0),
    ("4x2", "bn_q4_d2", 4, 2, 0.0),
    ("2x8, weight decay 0.1", "q2d8_wd0.1", 2, 8, 0.0),
    ("2x8, 1 layer", "q2d8_L1", 2, 8, 0.0),
    ("2x8, 1 layer, weight decay 0.1", "q2d8_L1_wd0.1", 2, 8, 0.0),
    ("2x8, dropout 0.2", "q2d8_drop0.2", 2, 8, 0.2),
    ("2x8, dropout 0.3", "q2d8_drop0.3", 2, 8, 0.3),
    ("2x8, VAE beta 1e-6", "q2d8_vae1e-6", 2, 8, 0.0),
    ("2x8, VAE beta 3e-6, 3-epoch KL warm-up", "q2d8_vae3e-6_wu3", 2, 8, 0.0),
    ("2x8, VAE beta 1e-5", "q2d8_vae1e-5", 2, 8, 0.0),
    ("2x8, VAE beta 1e-4", "q2d8_vae1e-4", 2, 8, 0.0),
]
FIELDS = ["config", "numbers_per_step", "dropout", "val_mse", "copy_mse", "val_over_copy", "train_over_copy", "gap", "shuffled_over_copy"]
NOTE = """Frozen DINOv3 ViT-S/16 patch tokens, 5 predicted frames 12 apart, deterministic z (except the VAE rows), batch 32, 10 epochs, single seed.
`val_mse` is the training loss (MSE over tokens and features) on held-out clips, averaged over the 5 steps; `copy_mse` is the same for
predicting F_0 unchanged. `*_over_copy` is the model's squared error divided by the copy baseline's (1 = no better than copy), averaged
over steps; `gap` = val - train of that ratio; `shuffled_over_copy` uses another sample's latents (above 1 means the latents matter).
The 4x8 baseline is epoch 10 of a 20-epoch run, with MSE rebuilt from the ratios and the fixed copy MSE."""


def load_epoch(run):
    """Return the metrics record of the comparison epoch, or None if the run is missing or shorter."""
    path = METRICS_DIR / f"{run}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.open()]
    return rows[EPOCH] if len(rows) > EPOCH else None


def make_row(label, record, copy, queries, dims, dropout):
    """Return one table row of means over the 5 steps."""
    mean = lambda values: sum(values) / len(values)
    mse = record.get("mse") or [ratio * c for ratio, c in zip(record["model"], copy)]
    val, train = mean(record["model"]), mean(record["train_model"])
    return {
        "config": label, "numbers_per_step": queries * dims, "dropout": dropout, "val_mse": mean(mse), "copy_mse": mean(copy),
        "val_over_copy": val, "train_over_copy": train, "gap": val - train, "shuffled_over_copy": mean(record["shuffled"]),
    }


def main():
    """Write results.csv and results.md for every ablation run that has finished 10 epochs."""
    records = {run: load_epoch(run) for _, run, *_ in RUNS}
    copy = next(r["mse_copy"] for r in records.values() if r and "mse_copy" in r)
    rows = [make_row(label, records[run], copy, q, d, p) for label, run, q, d, p in RUNS if records[run]]
    out = ROOT / "ablations"
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    table = ["| " + " | ".join(FIELDS) + " |", "|" + "---|" * len(FIELDS)]
    table += ["| " + " | ".join(f"{row[k]:.4f}" if isinstance(row[k], float) and k != "dropout" else str(row[k]) for k in FIELDS) + " |" for row in rows]
    (out / "results.md").write_text("# Latent bottleneck, regularisation and VAE ablation\n\n" + "\n".join(table) + "\n\n" + NOTE + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
