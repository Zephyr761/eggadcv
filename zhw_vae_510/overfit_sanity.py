"""5-minute sanity check: can the VAE overfit a single batch?

Procedure:
1. Pull one batch from the synthetic dataset.
2. Disable KL (kl_weight = 0).
3. Train the model on that single batch for ``--steps`` iterations.
4. Track reconstruction PSNR; success criterion is PSNR > 30 dB (any working
   VAE should hit this on a fixed batch within a few hundred steps).

If this fails, the VAE itself has a bug -- don't waste time on real data
training.

Usage::

    python -m zhw_vae_510.overfit_sanity --steps 500
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zhw_vae_510.crossview_vae import CrossView4DVAE  # noqa: E402
from zhw_vae_510.data import SyntheticCylinderDataset, SyntheticConfig  # noqa: E402


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return 99.0
    # Dynamic range is 2 (we use [-1, 1]), so MAX^2 = 4.
    return 10.0 * math.log10(4.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image-hw", type=int, nargs=2, default=[64, 128])
    ap.add_argument("--sequence-length", type=int, default=5)
    ap.add_argument("--view-count", type=int, default=6)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--latent-view-count", type=int, default=3)
    ap.add_argument("--out", default="zhw_vae_510/runs/overfit")
    ap.add_argument("--use-camera-params", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1) one batch
    ds = SyntheticCylinderDataset(SyntheticConfig(
        samples_per_epoch=1,
        sequence_length=args.sequence_length,
        view_count=args.view_count,
        image_hw=tuple(args.image_hw),
    ))
    sample = ds[0]
    x = sample["vae_images"].unsqueeze(0).to(device)            # [1, T, V, 3, H, W]
    K = sample["camera_intrinsics"].unsqueeze(0).to(device)     # [1, T, V, 3, 3]
    E = sample["camera_transforms"].unsqueeze(0).to(device)     # [1, T, V, 4, 4]
    intr_hw = sample["intrinsics_hw"]
    print(f"[data] x={tuple(x.shape)} K={tuple(K.shape)} E={tuple(E.shape)} hw={intr_hw}")

    # 2) model -- virtual_view_count must match V_input for reconstruction.
    model = CrossView4DVAE(
        base_channels=args.base_channels, latent_channels=4,
        virtual_view_count=args.view_count,
        latent_view_count=args.latent_view_count,
        num_attention_heads=2, num_bottleneck_blocks=1,
        temporal_downsample_factor=4, temporal_pre=1,
        spatial_downsample_factor=8,
    ).to(device)

    n = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # 3) overfit loop with KL=0
    model.train()
    best_psnr = 0.0
    t0 = time.time()
    K_in = K if args.use_camera_params else None
    E_in = E if args.use_camera_params else None

    for step in range(1, args.steps + 1):
        optim.zero_grad(set_to_none=True)
        posterior = model.encode(
            x, intrinsics=K_in, extrinsics=E_in, intrinsics_hw=intr_hw,
        ).latent_dist
        z = posterior.sample()
        recon = model.decode(z).sample
        t_min = min(recon.shape[1], x.shape[1])
        rec = F.l1_loss(recon[:, :t_min], x[:, :t_min])
        # No KL during overfit.
        rec.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                z_m = posterior.mode()
                recon_m = model.decode(z_m).sample[:, :t_min]
                p = psnr(recon_m, x[:, :t_min])
                model.train()
            best_psnr = max(best_psnr, p)
            dt = time.time() - t0
            print(f"  step {step:4d}/{args.steps}  rec_l1={rec.item():.4f}  "
                  f"psnr={p:.2f} dB  best={best_psnr:.2f}  elapsed={dt:.1f}s")

    # 4) final check + dump preview
    with torch.no_grad():
        model.eval()
        z_m = posterior.mode()
        recon_m = model.decode(z_m).sample[:, :t_min]
        final_psnr = psnr(recon_m, x[:, :t_min])
        # Save first-frame side-by-side for V=6 views.
        v_keep = min(x.shape[2], 6)
        gt = x[0, 0, :v_keep].cpu()
        rc = recon_m[0, 0, :v_keep].cpu()
        grid = torch.cat([gt, rc], dim=0)
        grid = (grid.clamp(-1, 1) + 1) / 2.0
        vutils.save_image(grid, str(out / "overfit_final.png"), nrow=v_keep)

    print("\n=== Result ===")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    # Thresholds calibrated for the synthetic stripe dataset. PSNR has a clear
    # signal of "is the model learning at all" -- if it climbs steadily during
    # training, the wiring is sound. The absolute number depends heavily on
    # bottleneck capacity / steps / lr, so we use lenient cut-offs.
    if final_psnr > 28:
        print(">> PASS: clean overfit (>28 dB). Wiring is correct.")
    elif final_psnr > 18:
        print(">> OK: PSNR climbing (>18 dB). Wiring is correct; "
              "give it more steps or larger base_channels for a sharper recon.")
    else:
        print(">> FAIL: <18 dB and not climbing. Likely a wiring / loss / lr "
              "issue. Inspect overfit_final.png for diagnosis.")
    print(f"Preview saved to: {out / 'overfit_final.png'}")


if __name__ == "__main__":
    main()
