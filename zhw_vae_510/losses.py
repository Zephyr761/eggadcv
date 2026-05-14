"""Loss bundle for the cross-view 4D VAE.

This is intentionally lightweight (no GAN, no fancy schedulers) so the first
training pass focuses on getting reconstruction + KL right. Once the VAE
overfits a small batch and produces visually reasonable outputs, you can
optionally bolt on adversarial / LPIPS terms.

Three components:

1. **Reconstruction loss**: pixel L1 (more robust to outliers than L2 on
   driving footage with bright sky / dark shadows). Optional perceptual term
   via LPIPS for sharper textures (skipped if ``lpips`` is not installed).

2. **KL divergence**: standard analytic KL of the Gaussian posterior against
   ``N(0, I)``, summed over latent elements then averaged across batch.
   Scaled by a small ``kl_weight`` (think 1e-6, similar to SD VAE) because
   driving images have very high entropy and KL would otherwise dominate.

3. **Logvar regulariser** (optional): a tiny L2 on ``logvar`` itself to
   stabilise the early phase where the posterior would otherwise prefer
   ``logvar -> -inf`` (deterministic decoder collapse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAELossConfig:
    rec_weight: float = 1.0
    rec_kind: str = "l1"          # 'l1' | 'l2' | 'huber'
    kl_weight: float = 1.0e-6     # SD VAE-ish; raise for stronger regularisation
    perceptual_weight: float = 0.0  # set >0 to enable LPIPS (requires lpips pkg)
    logvar_reg_weight: float = 0.0  # 1e-4 if you see logvar -> -inf
    kl_warmup_steps: int = 0      # linearly ramp KL from 0 to full weight


class VAELoss(nn.Module):
    """Stateful loss module so we can manage LPIPS and KL warmup cleanly."""

    def __init__(self, cfg: VAELossConfig):
        super().__init__()
        self.cfg = cfg
        self._step = 0

        self.lpips: Optional[nn.Module] = None
        if cfg.perceptual_weight > 0:
            try:
                import lpips  # type: ignore
                # alex is fastest and good enough for monitoring quality.
                self.lpips = lpips.LPIPS(net="alex", verbose=False).eval()
                # LPIPS has its own params; freeze them.
                for p in self.lpips.parameters():
                    p.requires_grad_(False)
            except ImportError:
                print(
                    "[VAELoss] 'lpips' package not installed; falling back "
                    "to pixel-only reconstruction loss. Run "
                    "'pip install lpips' to enable perceptual loss.")
                self.lpips = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rec(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kind = self.cfg.rec_kind
        if kind == "l1":
            return F.l1_loss(pred, target)
        if kind == "l2":
            return F.mse_loss(pred, target)
        if kind == "huber":
            return F.smooth_l1_loss(pred, target, beta=0.1)
        raise ValueError(f"Unknown rec_kind={kind}")

    def _perceptual(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects 4D [N, 3, H, W] in [-1, 1].
        # ``pred`` and ``target`` are ``[B, T, V, C, H, W]`` in [-1, 1].
        if self.lpips is None or pred.shape[3] != 3:
            return pred.new_zeros(())
        flat_pred = pred.flatten(0, 2)
        flat_target = target.flatten(0, 2)
        # LPIPS internally returns per-sample scalar; mean over batch.
        return self.lpips(flat_pred, flat_target).mean()

    def _kl(self, posterior) -> torch.Tensor:
        # ``posterior.kl()`` in this codebase returns a *sum* over all
        # latent elements (B, T, V, C, H, W). Normalise by batch so the
        # absolute value doesn't depend on resolution / sequence length.
        kl = posterior.kl()
        # Recover the leading batch dimension. ``posterior.mean`` has the
        # same shape, take its size(0) as B.
        b = posterior.mean.shape[0]
        return kl / max(b, 1)

    def _kl_weight(self) -> float:
        if self.cfg.kl_warmup_steps <= 0:
            return self.cfg.kl_weight
        ramp = min(1.0, self._step / float(self.cfg.kl_warmup_steps))
        return self.cfg.kl_weight * ramp

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------
    def forward(self, pred, target, posterior) -> dict:
        """Compute the total loss and per-term scalars for logging.

        Args:
            pred: VAE reconstruction, ``[B, T, V, C, H, W]`` in [-1, 1].
            target: ground-truth frames in the same shape and range.
            posterior: ``DiagonalGaussianDistribution`` from VAE.encode().

        Returns:
            dict with ``loss`` (scalar tensor for backward) and a bunch of
            detached scalars for logging.
        """
        rec = self._rec(pred, target)
        perc = self._perceptual(pred, target)
        kl = self._kl(posterior)

        logvar_reg = pred.new_zeros(())
        if self.cfg.logvar_reg_weight > 0:
            logvar_reg = (posterior.logvar.pow(2)).mean()

        kl_w = self._kl_weight()

        loss = (
            self.cfg.rec_weight * rec
            + self.cfg.perceptual_weight * perc
            + kl_w * kl
            + self.cfg.logvar_reg_weight * logvar_reg
        )

        self._step += 1
        return {
            "loss": loss,
            "rec": rec.detach(),
            "perceptual": perc.detach() if isinstance(perc, torch.Tensor) else pred.new_zeros(()),
            "kl": kl.detach(),
            "kl_weight_now": pred.new_tensor(kl_w),
            "logvar_reg": logvar_reg.detach() if isinstance(logvar_reg, torch.Tensor) else pred.new_zeros(()),
            "logvar_mean": posterior.logvar.detach().mean(),
            "mean_abs_z": posterior.mean.detach().abs().mean(),
        }


# ----------------------------------------------------------------------
# Quick sanity check when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from zhw_vae_510.crossview_vae import CrossView4DVAE

    vae = CrossView4DVAE(
        base_channels=16, latent_channels=4,
        virtual_view_count=4, latent_view_count=2,
        num_attention_heads=2, num_bottleneck_blocks=1,
        temporal_downsample_factor=4, spatial_downsample_factor=8,
        temporal_pre=1)
    x = torch.randn(1, 5, 6, 3, 64, 64).clamp(-1, 1)

    out = vae(x)
    loss_fn = VAELoss(VAELossConfig(kl_weight=1e-6, rec_kind="l1"))
    metrics = loss_fn(out["sample"][:, :x.shape[1]], x[:, :out["sample"].shape[1]],
                      out["posterior"])
    for k, v in metrics.items():
        print(f"  {k}: {v.item():.4g}" if hasattr(v, 'item') else f"  {k}: {v}")
