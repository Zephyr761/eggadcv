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
   ``N(0, I)``. The reduction can either match the original implementation
   (sum over latent elements, average over batch) or average over all latent
   elements for easier comparisons across different bottleneck sizes.

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
    kl_reduction: str = "batch"   # 'batch' | 'mean'
    perceptual_weight: float = 0.0  # set >0 to enable LPIPS (requires lpips pkg)
    perceptual_batch_size: int = 4  # LPIPS chunk size over flattened frames/views
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
        # Flatten all leading dims, e.g. [B,T,V,C,H,W] -> [B*T*V,C,H,W].
        if self.lpips is None or pred.ndim < 4 or pred.shape[-3] != 3:
            return pred.new_zeros(())
        flat_pred = pred.reshape(-1, *pred.shape[-3:]).float()
        flat_target = target.reshape(-1, *target.shape[-3:]).float()
        chunk = max(int(self.cfg.perceptual_batch_size), 1)
        total = flat_pred.new_zeros(())
        count = 0
        for start in range(0, flat_pred.shape[0], chunk):
            end = min(start + chunk, flat_pred.shape[0])
            score = self.lpips(flat_pred[start:end], flat_target[start:end])
            total = total + score.sum()
            count += score.numel()
        return total / max(count, 1)

    def _kl(self, posterior) -> torch.Tensor:
        # ``posterior.kl()`` in this codebase returns a single scalar summed
        # over every latent element. Keep the old batch-normalised behavior
        # for compatibility, but allow per-element KL when comparing different
        # temporal/view/spatial compression settings.
        kl = posterior.kl()
        reduction = self.cfg.kl_reduction
        if reduction == "batch":
            return kl / max(posterior.mean.shape[0], 1)
        if reduction == "mean":
            return kl / max(posterior.mean.numel(), 1)
        raise ValueError(f"Unknown kl_reduction={reduction}")

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
