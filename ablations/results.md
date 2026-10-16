# Latent bottleneck, regularisation and VAE ablation

| config | numbers_per_step | dropout | val_mse | copy_mse | val_over_copy | train_over_copy | gap | shuffled_over_copy |
|---|---|---|---|---|---|---|---|---|
| 4x8 baseline | 32 | 0.0 | 0.0124 | 0.0251 | 0.5122 | 0.3014 | 0.2107 | 1.3931 |
| 2x8 | 16 | 0.0 | 0.0121 | 0.0251 | 0.5004 | 0.2974 | 0.2030 | 1.3936 |
| 4x4 | 16 | 0.0 | 0.0127 | 0.0251 | 0.5239 | 0.3071 | 0.2168 | 1.4310 |
| 4x2 | 8 | 0.0 | 0.0141 | 0.0251 | 0.5825 | 0.3200 | 0.2625 | 1.2910 |
| 2x8, weight decay 0.1 | 16 | 0.0 | 0.0119 | 0.0251 | 0.4951 | 0.2966 | 0.1985 | 1.3990 |
| 2x8, 1 layer | 16 | 0.0 | 0.0124 | 0.0251 | 0.5126 | 0.3387 | 0.1738 | 1.2456 |
| 2x8, 1 layer, weight decay 0.1 | 16 | 0.0 | 0.0123 | 0.0251 | 0.5096 | 0.3390 | 0.1706 | 1.2436 |
| 2x8, dropout 0.2 | 16 | 0.2 | 0.0123 | 0.0251 | 0.5077 | 0.3165 | 0.1912 | 1.2342 |
| 2x8, dropout 0.3 | 16 | 0.3 | 0.0123 | 0.0251 | 0.5082 | 0.3288 | 0.1794 | 1.2346 |
| 2x8, VAE beta 1e-6 | 16 | 0.0 | 0.0132 | 0.0251 | 0.5462 | 0.3114 | 0.2348 | 1.3890 |
| 2x8, VAE beta 3e-6, 3-epoch KL warm-up | 16 | 0.0 | 0.0129 | 0.0251 | 0.5308 | 0.3057 | 0.2251 | 1.3132 |
| 2x8, VAE beta 1e-5 | 16 | 0.0 | 0.0171 | 0.0251 | 0.7009 | 0.3398 | 0.3611 | 1.1212 |
| 2x8, VAE beta 1e-4 | 16 | 0.0 | 0.0218 | 0.0251 | 0.8772 | 0.3945 | 0.4827 | 0.8835 |

Frozen DINOv3 ViT-S/16 patch tokens, 5 predicted frames 12 apart, deterministic z (except the VAE rows), batch 32, 10 epochs, single seed.
`val_mse` is the training loss (MSE over tokens and features) on held-out clips, averaged over the 5 steps; `copy_mse` is the same for
predicting F_0 unchanged. `*_over_copy` is the model's squared error divided by the copy baseline's (1 = no better than copy), averaged
over steps; `gap` = val - train of that ratio; `shuffled_over_copy` uses another sample's latents (above 1 means the latents matter).
The 4x8 baseline is epoch 10 of a 20-epoch run, with MSE rebuilt from the ratios and the fixed copy MSE.
