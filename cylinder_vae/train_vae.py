"""Standalone training entry point for ``CrossView4DVAE``.

Designed to be lean: pure PyTorch, no Lightning / Accelerate, so the bug
surface is small and debugging is straightforward. All the moving parts
(model, data, loss) are imported from this folder.

Usage examples
--------------

# Synthetic data, single GPU, default config (good for first sanity-train).
python -m step_1.train_vae --data synthetic --steps 2000 --image-hw 64 128

# Multi-GPU DDP. Per-rank batch size is still --batch-size.
torchrun --nproc_per_node=4 -m step_1.train_vae --data synthetic \
    --steps 2000 --image-hw 64 128 --batch-size 2

# Real nuPlan data (paths must match those in dwm.tools.dataset_nus).
python -m step_1.train_vae --data nuplan --steps 50000 --image-hw 256 448 \
    --batch-size 1 --base-channels 64

# Resume from a checkpoint.
python -m step_1.train_vae --data nuplan --resume step_1/runs/last.pt

The script intentionally writes nothing outside ``step_1/`` so it's safe
to run repeatedly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# Make ``dwm`` importable for the dataset adapters that pull in
# dwm.datasets.nuscenes / nuplan.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The VAE model itself lives next to this script.
try:
    from .crossview_vae import CrossView4DVAE  # noqa: E402
    from .data import (  # noqa: E402
        MultiViewVAEAdapter,
        SyntheticCylinderDataset,
        SyntheticConfig,
        make_nuscenes_base,
        make_nuscenes_scene_folder_base,
        vae_collate,
    )
    from .losses import VAELoss, VAELossConfig  # noqa: E402
except ImportError:
    from crossview_vae import CrossView4DVAE  # noqa: E402
    from data import (  # noqa: E402
        MultiViewVAEAdapter,
        SyntheticCylinderDataset,
        SyntheticConfig,
        make_nuscenes_base,
        make_nuscenes_scene_folder_base,
        vae_collate,
    )
    from losses import VAELoss, VAELossConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "nuplan", "nus", "nuscenes", "nuscenes_scene"],
                   default="synthetic", help="Dataset source.")
    p.add_argument("--out", default="cylinder_vae/runs",
                   help="Where to save checkpoints / previews / logs.")
    p.add_argument("--resume", default="",
                   help="Path to a checkpoint .pt to resume from.")

    # data
    p.add_argument("--image-hw", type=int, nargs=2, default=[64, 128])
    p.add_argument("--sequence-length", type=int, default=5,
                   help="Must satisfy T = temporal_pre + k * tdf.")
    p.add_argument("--view-count", type=int, default=6,
                   help="Number of input cameras (used by synthetic; nuScenes / "
                        "nuPlan are read straight from the underlying dataset).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--overfit-samples", type=int, default=0,
                   help="Debug mode: train on a fixed contiguous subset. "
                        "Use 1 for true single-sample overfit.")
    p.add_argument("--overfit-sample-index", type=int, default=0,
                   help="First dataset index used by --overfit-samples.")

    # nuScenes-specific paths
    p.add_argument("--nusc-data-root", default="",
                   help="(nuScenes) directory holding the extracted blobs / "
                        "metadata json. Required when --data=nuscenes.")
    p.add_argument("--nusc-cache-root", default="",
                   help="(nuScenes) optional pre-generated image cache dir.")
    p.add_argument("--nusc-dataset-name", default="v1.0-trainval",
                   help="(nuScenes) metadata sub-directory; e.g. "
                        "'v1.0-trainval' or 'interp_12Hz_trainval'.")
    p.add_argument("--nusc-split", default="train",
                   help="(nuScenes) split: train / val / mini_train / mini_val.")
    p.add_argument("--nusc-fps", type=int, default=2)
    p.add_argument("--nusc-stride", type=float, default=0.5)
    p.add_argument("--nusc-keyframe-only", action="store_true", default=True)

    # model (mirrors CrossView4DVAE config keys)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--virtual-view-count", type=int, default=-1,
                   help="Number of virtual cameras the cylinder projects to. "
                        "Defaults to -1 which means 'match the input view "
                        "count'. With --use-camera-params this can differ "
                        "from the physical camera count because the decoded "
                        "cylinder is rendered back to camera views for loss.")
    p.add_argument("--latent-view-count", type=int, default=3,
                   help="Number of latent views (V_lat). Must be "
                        "<= virtual_view_count. 3 covers 360deg in three "
                        "120deg sectors which keeps decent horizontal "
                        "angular resolution while still halving the view "
                        "dim of a 6-camera ring.")
    p.add_argument("--latent-time-count", type=int, default=2,
                   help="Default latent time-query count used when near/far "
                        "time budgets are not specified.")
    p.add_argument("--near-latent-time-count", type=int, default=-1,
                   help="Latent time-query budget for the near cylinder layer.")
    p.add_argument("--far-latent-time-count", type=int, default=-1,
                   help="Latent time-query budget for the far cylinder layer.")
    p.add_argument("--near-latent-view-count", type=int, default=-1,
                   help="Latent view budget for the near cylinder layer. "
                        "Default -1 uses --latent-view-count.")
    p.add_argument("--far-latent-view-count", type=int, default=-1,
                   help="Latent view budget for the far cylinder layer. "
                        "Default -1 uses roughly half --latent-view-count.")
    p.add_argument("--latent-spatial-downsample-factor", type=int, default=1,
                   help="Default extra latent spatial downsample factor inside "
                        "the learned T/V/S compressor.")
    p.add_argument("--near-latent-spatial-downsample-factor", type=int, default=-1,
                   help="Near-layer extra latent spatial downsample. Default "
                        "-1 uses --latent-spatial-downsample-factor.")
    p.add_argument("--far-latent-spatial-downsample-factor", type=int, default=-1,
                   help="Far-layer extra latent spatial downsample. Default "
                        "-1 uses 2x the base factor, so far tokens are "
                        "spatially cheaper than near tokens.")
    p.add_argument("--num-attention-heads", type=int, default=2)
    p.add_argument("--num-bottleneck-blocks", type=int, default=1)
    p.add_argument("--temporal-downsample-factor", type=int, default=4)
    p.add_argument("--temporal-pre", type=int, default=1)
    p.add_argument("--spatial-downsample-factor", type=int, default=8)
    p.add_argument("--cylinder-radii", type=float, nargs="+", default=[4.0, 20.0],
                   help="True cylinder layer radii in metres. Two values "
                        "enable near/far layered cylinders.")
    p.add_argument("--cylinder-height-scale", type=float, default=1.0,
                   help="Vertical range scale for the canonical cylinder. In "
                        "camera_fov mode, values above 1 expand the camera "
                        "render coverage and can remove top/bottom gray arcs; "
                        "values below 1 crop the band. "
                        "In aspect mode this keeps the old multiplier.")
    p.add_argument("--cylinder-vertical-mode",
                   choices=["aspect", "camera_fov"], default="camera_fov",
                   help="How to set the cylinder vertical range. camera_fov "
                        "uses camera intrinsics to avoid unobserved gray "
                        "top/bottom bands; aspect keeps the old aspect-ratio "
                        "heuristic.")
    p.add_argument("--projector-edge-feather", type=float, default=0.05,
                   help="Soft fade width near source-image borders in "
                        "normalized grid coordinates. This reduces hard "
                        "camera mosaic seams in the projected cylinder.")
    p.add_argument("--projector-angular-power", type=float, default=0.5,
                   help="Power applied to angular visibility weights. Values "
                        "below 1 soften source-camera transitions.")
    p.add_argument("--projector-blend-mode",
                   choices=["soft", "best_camera"], default="soft",
                   help="How to aggregate overlapping source cameras on the "
                        "cylinder. soft averages by visibility weights; "
                        "best_camera picks one dominant physical camera per "
                        "virtual view to avoid parallax ghosting in RGB "
                        "targets.")
    p.add_argument("--ego-coordinate-mode", choices=["nuscenes", "synthetic"],
                   default="nuscenes",
                   help="Ego axes used by the geometry projector. nuScenes "
                        "uses x/y ground plane and z up; synthetic uses x/z "
                        "ground plane and y up.")
    p.add_argument("--decode-camera-from-cylinder", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--use-view-residual", action="store_true",
                   help=argparse.SUPPRESS)

    # loss
    p.add_argument("--rec-kind", default="l1", choices=["l1", "l2", "huber"])
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--kl-reduction", default="batch", choices=["batch", "mean"],
                   help="KL reduction. 'batch' keeps the original summed "
                        "latent KL divided by B; 'mean' reports KL per latent "
                        "element, which is easier to compare across "
                        "bottleneck sizes.")
    p.add_argument("--kl-warmup", type=int, default=2000)
    p.add_argument("--perceptual-weight", type=float, default=0.0,
                   help="Set >0 to enable LPIPS (requires `pip install lpips`).")
    p.add_argument("--perceptual-batch-size", type=int, default=4,
                   help="LPIPS chunk size after flattening B*T*V frames. "
                        "Lower this if perceptual loss runs out of memory.")
    p.add_argument("--edge-loss-weight", type=float, default=0.0,
                   help="Camera-domain first-order image-gradient loss weight.")
    p.add_argument("--cylinder-edge-loss-weight", type=float, default=0.0,
                   help="Cylinder-domain first-order image-gradient loss "
                        "weight. Uses the cylinder mask when "
                        "--masked-cylinder-loss is enabled.")
    p.add_argument("--logvar-reg-weight", type=float, default=0.0,
                   help="Tiny L2 penalty on posterior logvar, e.g. 1e-4.")
    p.add_argument("--reconstruction-domain",
                   choices=["camera", "cylinder", "both"], default="camera",
                   help="Where to apply reconstruction supervision. camera "
                        "matches the original closed-loop objective; cylinder "
                        "supervises the canonical projected output directly; "
                        "both combines them.")
    p.add_argument("--camera-loss-weight", type=float, default=1.0,
                   help="Weight for camera-domain reconstruction when "
                        "--reconstruction-domain is camera/both.")
    p.add_argument("--cylinder-loss-weight", type=float, default=1.0,
                   help="Weight for cylinder-domain reconstruction when "
                        "--reconstruction-domain is cylinder/both.")
    p.add_argument("--masked-cylinder-loss", action="store_true",
                   help="Apply cylinder reconstruction only where the "
                        "camera-to-cylinder projector has valid source "
                        "coverage.")
    p.add_argument("--cylinder-mask-threshold", type=float, default=1e-4,
                   help="Coverage threshold used by --masked-cylinder-loss.")
    p.add_argument("--cylinder-seam-weight", type=float, default=0.0,
                   help="Optional continuity penalty between neighboring "
                        "virtual cylinder slices.")
    p.add_argument("--view-residual-penalty-weight", type=float, default=0.0,
                   help=argparse.SUPPRESS)
    p.add_argument("--sample-posterior", action="store_true",
                   help="Sample z from the posterior during training. By "
                        "default training uses posterior.mode() for a "
                        "deterministic reconstruction sanity path.")

    # optim / sched
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)

    # mixed precision
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    p.add_argument("--ddp-backend", default="",
                   help="Distributed backend. Empty means nccl on CUDA, else gloo.")
    p.add_argument("--find-unused-parameters", action="store_true",
                   help="Pass find_unused_parameters=True to DDP.")
    p.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int,
                   default=int(os.environ.get("LOCAL_RANK", "0")),
                   help=argparse.SUPPRESS)

    # logging
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--preview-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--use-camera-params", action="store_true",
                   help="Use K/E for the full closed-loop camera->cylinder->"
                        "latent->cylinder->camera path. Otherwise the VAE "
                        "uses the fallback fixed-view path.")
    p.add_argument("--gradient-checkpoint", action="store_true",
                   help="CrossView4DVAE: checkpoint encode+decode (less VRAM, slower).")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed(args):
    """Initialize torch.distributed when launched by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device_of(),
        }

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = args.ddp_backend or ("gloo" if os.name == "nt" else "nccl")
    else:
        device = torch.device("cpu")
        backend = args.ddp_backend or "gloo"

    dist.init_process_group(backend=backend)
    return {
        "distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def is_main_process(ddp_info) -> bool:
    return ddp_info["rank"] == 0


def rank0_print(ddp_info, *args, **kwargs):
    if is_main_process(ddp_info):
        print(*args, **kwargs)


def cleanup_distributed(ddp_info):
    if ddp_info.get("distributed") and dist.is_initialized():
        dist.destroy_process_group()


def amp_dtype(name: str):
    return {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def validate_temporal_config(args):
    delta = args.sequence_length - args.temporal_pre
    if delta < 0 or delta % args.temporal_downsample_factor != 0:
        raise ValueError(
            "Invalid sequence_length for this VAE temporal stride: "
            f"sequence_length={args.sequence_length}, "
            f"temporal_pre={args.temporal_pre}, "
            f"temporal_downsample_factor={args.temporal_downsample_factor}. "
            "Expected T = temporal_pre + k * temporal_downsample_factor."
        )


def lr_lambda(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    if step < warmup:
        return step / float(warmup)
    return 1.0


def make_dataset(args, ddp_info=None):
    if args.data == "synthetic":
        cfg = SyntheticConfig(
            samples_per_epoch=10_000,
            sequence_length=args.sequence_length,
            view_count=args.view_count,
            image_hw=tuple(args.image_hw),
        )
        return SyntheticCylinderDataset(cfg)

    if args.data in ("nus", "nuscenes"):
        if not args.nusc_data_root:
            raise ValueError("--nusc-data-root is required when --data=nus/nuscenes")
        base = make_nuscenes_base(
            data_root=args.nusc_data_root,
            cache_root=args.nusc_cache_root or None,
            dataset_name=args.nusc_dataset_name,
            split=args.nusc_split,
            sequence_length=args.sequence_length,
            fps=args.nusc_fps,
            stride=args.nusc_stride,
            keyframe_only=args.nusc_keyframe_only,
        )
        if ddp_info is None or is_main_process(ddp_info):
            print(f"[data] nuScenes: {len(base)} clips "
                  f"(dataset_name={args.nusc_dataset_name}, split={args.nusc_split}, "
                  f"fps={args.nusc_fps}, T={args.sequence_length})")
        return MultiViewVAEAdapter(
            base, sequence_length=args.sequence_length,
            image_hw=tuple(args.image_hw))

    if args.data == "nuscenes_scene":
        if not args.nusc_data_root:
            raise ValueError("--nusc-data-root is required when --data=nuscenes_scene")
        base = make_nuscenes_scene_folder_base(
            data_root=args.nusc_data_root,
            split=args.nusc_split,
            sequence_length=args.sequence_length,
        )
        if ddp_info is None or is_main_process(ddp_info):
            print(f"[data] nuScenes scene-folder: {len(base)} clips "
                  f"(root={args.nusc_data_root}, split={args.nusc_split}, "
                  f"T={args.sequence_length})")
        return MultiViewVAEAdapter(
            base, sequence_length=args.sequence_length,
            image_hw=tuple(args.image_hw))

    # nuplan: import locally so synthetic runs have no nuplan deps.
    from dwm.tools.dataset_nus import make_base_ds
    base = make_base_ds(train=True)
    return MultiViewVAEAdapter(
        base, sequence_length=args.sequence_length,
        image_hw=tuple(args.image_hw))


def maybe_make_overfit_subset(dataset, args, ddp_info):
    if args.overfit_samples <= 0:
        return dataset
    if args.overfit_sample_index < 0:
        raise ValueError("--overfit-sample-index must be >= 0")
    if args.overfit_samples < 1:
        raise ValueError("--overfit-samples must be >= 1 when enabled")

    end = args.overfit_sample_index + args.overfit_samples
    if end > len(dataset):
        raise ValueError(
            "Requested overfit subset is outside the dataset: "
            f"start={args.overfit_sample_index}, "
            f"count={args.overfit_samples}, len={len(dataset)}"
        )

    indices = list(range(args.overfit_sample_index, end))
    rank0_print(
        ddp_info,
        f"[data] overfit subset enabled: indices "
        f"{indices[0]}..{indices[-1]} ({len(indices)} sample(s)); "
        "shuffle disabled"
    )
    return Subset(dataset, indices)


def save_preview(images: torch.Tensor, recon: torch.Tensor,
                 path: str, max_views: int = 6):
    """Save a side-by-side grid of first-frame views: GT row over recon row."""
    # images / recon: [B, T, V, C, H, W] in [-1, 1]
    b, t, v, c, h, w = images.shape
    v_keep = min(v, max_views)
    gt = images[0, 0, :v_keep].cpu()
    rc = recon[0, 0, :v_keep].cpu()
    grid = torch.cat([gt, rc], dim=0)             # 2*V_keep
    grid = (grid.clamp(-1, 1) + 1) / 2.0
    vutils.save_image(grid, path, nrow=v_keep)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                        kind: str) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "l2":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target, beta=0.1)
    raise ValueError(f"Unknown rec_kind={kind}")


def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    if mask.ndim != pred.ndim:
        raise ValueError(
            f"Mask rank must match prediction rank: pred={tuple(pred.shape)} "
            f"mask={tuple(mask.shape)}.")
    if mask.shape[:-3] != pred.shape[:-3] or mask.shape[-2:] != pred.shape[-2:]:
        raise ValueError(
            "Mask shape must match prediction non-channel and spatial dims: "
            f"pred={tuple(pred.shape)} mask={tuple(mask.shape)}.")
    if mask.shape[-3] == 1 and pred.shape[-3] != 1:
        mask = mask.expand(*pred.shape[:-3], pred.shape[-3], *pred.shape[-2:])
    weight = mask.to(dtype=pred.dtype).clamp(0, 1)
    if kind == "l1":
        loss_map = (pred - target).abs()
    elif kind == "l2":
        loss_map = (pred - target).square()
    elif kind == "huber":
        loss_map = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none")
    else:
        raise ValueError(f"Unknown rec_kind={kind}")
    return (loss_map * weight).sum() / weight.sum().clamp(min=1.0)


def _expand_mask_channels(mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if mask.ndim != pred.ndim:
        raise ValueError(
            f"Mask rank must match prediction rank: pred={tuple(pred.shape)} "
            f"mask={tuple(mask.shape)}.")
    if mask.shape[:-3] != pred.shape[:-3] or mask.shape[-2:] != pred.shape[-2:]:
        raise ValueError(
            "Mask shape must match prediction non-channel and spatial dims: "
            f"pred={tuple(pred.shape)} mask={tuple(mask.shape)}.")
    if mask.shape[-3] == 1 and pred.shape[-3] != 1:
        mask = mask.expand(*pred.shape[:-3], pred.shape[-3], *pred.shape[-2:])
    elif mask.shape[-3] != pred.shape[-3]:
        raise ValueError(
            "Mask channel dim must be 1 or match prediction channels: "
            f"pred={tuple(pred.shape)} mask={tuple(mask.shape)}.")
    return mask.to(dtype=pred.dtype).clamp(0, 1)


def gradient_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """First-order spatial gradient L1 loss over the last H/W dimensions."""
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_target = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_target = target[..., 1:, :] - target[..., :-1, :]
    dx_loss = (dx_pred - dx_target).abs()
    dy_loss = (dy_pred - dy_target).abs()
    if mask is None:
        return 0.5 * (dx_loss.mean() + dy_loss.mean())

    weight = _expand_mask_channels(mask, pred)
    dx_weight = weight[..., :, 1:] * weight[..., :, :-1]
    dy_weight = weight[..., 1:, :] * weight[..., :-1, :]
    dx = (dx_loss * dx_weight).sum() / dx_weight.sum().clamp(min=1.0)
    dy = (dy_loss * dy_weight).sum() / dy_weight.sum().clamp(min=1.0)
    return 0.5 * (dx + dy)


def cylinder_seam_loss(x: torch.Tensor) -> torch.Tensor:
    view_dim = 3 if x.ndim == 7 else 2
    if x.shape[view_dim] <= 1:
        return x.new_zeros(())
    if x.ndim == 7:
        internal = (
            x[:, :, :, :-1, :, :, -1] -
            x[:, :, :, 1:, :, :, 0]
        ).abs().mean()
        wrap = (
            x[:, :, :, -1, :, :, -1] -
            x[:, :, :, 0, :, :, 0]
        ).abs().mean()
    else:
        internal = (x[:, :, :-1, :, :, -1] - x[:, :, 1:, :, :, 0]).abs().mean()
        wrap = (x[:, :, -1, :, :, -1] - x[:, :, 0, :, :, 0]).abs().mean()
    return 0.5 * (internal + wrap)


def log_input_probe(ddp_info, sample: dict, out_dir: Path, max_views: int = 6):
    if not is_main_process(ddp_info):
        return
    imgs = sample["vae_images"]
    view_means = imgs.mean(dim=(0, 2, 3, 4))
    means_str = ", ".join(f"{m:.3f}" for m in view_means.tolist())
    print(
        f"[data] probe vae_images={tuple(imgs.shape)} "
        f"range=[{imgs.min().item():.3f}, {imgs.max().item():.3f}] "
        f"view_mean=[{means_str}]"
    )
    v_keep = min(imgs.shape[1], max_views)
    grid = (imgs[0, :v_keep].clamp(-1, 1) + 1) / 2.0
    vutils.save_image(grid, str(out_dir / "input_probe.png"), nrow=v_keep)


def save_checkpoint(model, optim, scheduler, step, args, path):
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "args": vars(args),
    }
    torch.save(payload, path)


def append_metrics(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_model_state(model, state_dict, strict: bool = False):
    """Load checkpoints saved with or without a DDP ``module.`` prefix."""
    try:
        return model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            return model.load_state_dict(state_dict, strict=strict)
        raise


def reduce_metric_dict(metrics, ddp_info):
    reduced = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            value = v.detach().to(ddp_info["device"])
        else:
            value = torch.tensor(float(v), device=ddp_info["device"])
        if value.ndim != 0:
            value = value.mean()
        if ddp_info["distributed"]:
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= ddp_info["world_size"]
        reduced[k] = value
    return reduced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    if args.decode_camera_from_cylinder and not args.use_camera_params:
        args.use_camera_params = True
    args.decode_camera_from_cylinder = args.use_camera_params
    validate_temporal_config(args)
    ddp_info = init_distributed(args)
    try:
        device = ddp_info["device"]

        out_dir = Path(args.out)
        preview_dir = out_dir / "previews"
        ckpt_dir = out_dir / "ckpts"
        metrics_path = out_dir / "metrics.jsonl"
        if is_main_process(ddp_info):
            out_dir.mkdir(parents=True, exist_ok=True)
            preview_dir.mkdir(exist_ok=True)
            ckpt_dir.mkdir(exist_ok=True)
        if ddp_info["distributed"]:
            if device.type == "cuda":
                dist.barrier(device_ids=[ddp_info["local_rank"]])
            else:
                dist.barrier()

        rank0_print(
            ddp_info,
            f"[setup] device={device}, amp={args.amp}, "
            f"distributed={ddp_info['distributed']}, "
            f"world_size={ddp_info['world_size']}, "
            f"posterior={'sample' if args.sample_posterior else 'mode'}"
        )

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        dataset = make_dataset(args, ddp_info)
        dataset = maybe_make_overfit_subset(dataset, args, ddp_info)
        overfit_active = args.overfit_samples > 0
        drop_last = not overfit_active
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_info["world_size"],
            rank=ddp_info["rank"],
            shuffle=not overfit_active,
            drop_last=drop_last,
        ) if ddp_info["distributed"] else None
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None and not overfit_active),
            sampler=train_sampler,
            num_workers=args.num_workers, collate_fn=vae_collate,
            drop_last=drop_last, pin_memory=device.type == "cuda")

        # Probe one sample to learn V_input. Reconstruction loss requires
        # ``virtual_view_count == V_input`` so VAE output and GT have matching
        # view dim.
        probe = dataset[0]
        log_input_probe(ddp_info, probe, out_dir)
        v_input = int(probe["vae_images"].shape[1])
        if args.virtual_view_count <= 0:
            args.virtual_view_count = v_input
            rank0_print(ddp_info, f"[setup] auto virtual_view_count = V_input = {v_input}")
        elif (
            args.virtual_view_count != v_input
            and not args.use_camera_params
        ):
            rank0_print(
                ddp_info,
                f"[warn] virtual_view_count ({args.virtual_view_count}) != "
                f"V_input ({v_input}). Reconstruction loss is going to fail "
                "unless you use --use-camera-params for closed-loop "
                "cylinder rendering. Aborting.")
            raise SystemExit(1)
        elif args.virtual_view_count != v_input:
            rank0_print(
                ddp_info,
                f"[setup] closed-loop cylinder mode: V_virtual="
                f"{args.virtual_view_count}, V_camera={v_input}")
        if args.latent_view_count > args.virtual_view_count:
            rank0_print(
                ddp_info,
                f"[warn] latent_view_count ({args.latent_view_count}) must be "
                f"<= virtual_view_count ({args.virtual_view_count}). Aborting.")
            raise SystemExit(1)
        near_v = (
            args.near_latent_view_count
            if args.near_latent_view_count > 0
            else args.latent_view_count
        )
        far_v = (
            args.far_latent_view_count
            if args.far_latent_view_count > 0
            else max(1, args.latent_view_count // 2)
        )
        near_t = (
            args.near_latent_time_count
            if args.near_latent_time_count > 0
            else args.latent_time_count
        )
        far_t = (
            args.far_latent_time_count
            if args.far_latent_time_count > 0
            else max(1, args.latent_time_count // 2)
        )
        if args.latent_spatial_downsample_factor < 1:
            rank0_print(ddp_info, "[warn] latent_spatial_downsample_factor must be >= 1.")
            raise SystemExit(1)
        near_s = (
            args.near_latent_spatial_downsample_factor
            if args.near_latent_spatial_downsample_factor > 0
            else args.latent_spatial_downsample_factor
        )
        far_s = (
            args.far_latent_spatial_downsample_factor
            if args.far_latent_spatial_downsample_factor > 0
            else max(1, args.latent_spatial_downsample_factor * 2)
        )
        if near_v > args.virtual_view_count or far_v > args.virtual_view_count:
            rank0_print(
                ddp_info,
                f"[warn] layer latent views must be <= virtual_view_count "
                f"({args.virtual_view_count}); near={near_v}, far={far_v}.")
            raise SystemExit(1)
        if near_t > args.sequence_length or far_t > args.sequence_length:
            rank0_print(
                ddp_info,
                f"[warn] layer latent times must be <= sequence_length "
                f"({args.sequence_length}); near={near_t}, far={far_t}.")
            raise SystemExit(1)
        rank0_print(
            ddp_info,
            f"[setup] cylinder layers={len(args.cylinder_radii)} "
            f"radii={args.cylinder_radii} "
            f"latent T/V/S near={near_t}x{near_v}/s{near_s} "
            f"far={far_t}x{far_v}/s{far_s}")

        # Log args after auto-filled values (e.g. virtual_view_count) are resolved.
        if is_main_process(ddp_info):
            with open(out_dir / "args.json", "w") as f:
                json.dump(vars(args), f, indent=2)
            if not args.resume and metrics_path.exists():
                metrics_path.unlink()

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        model = CrossView4DVAE(
            base_channels=args.base_channels,
            latent_channels=args.latent_channels,
            sequence_length=args.sequence_length,
            virtual_view_count=args.virtual_view_count,
            latent_view_count=args.latent_view_count,
            latent_time_count=args.latent_time_count,
            image_height=args.image_hw[0],
            image_width=args.image_hw[1],
            near_latent_time_count=(
                None if args.near_latent_time_count <= 0
                else args.near_latent_time_count),
            far_latent_time_count=(
                None if args.far_latent_time_count <= 0
                else args.far_latent_time_count),
            near_latent_view_count=(
                None if args.near_latent_view_count <= 0
                else args.near_latent_view_count),
            far_latent_view_count=(
                None if args.far_latent_view_count <= 0
                else args.far_latent_view_count),
            latent_spatial_downsample_factor=args.latent_spatial_downsample_factor,
            near_latent_spatial_downsample_factor=(
                None if args.near_latent_spatial_downsample_factor <= 0
                else args.near_latent_spatial_downsample_factor),
            far_latent_spatial_downsample_factor=(
                None if args.far_latent_spatial_downsample_factor <= 0
                else args.far_latent_spatial_downsample_factor),
            num_attention_heads=args.num_attention_heads,
            num_bottleneck_blocks=args.num_bottleneck_blocks,
            temporal_downsample_factor=args.temporal_downsample_factor,
            temporal_pre=args.temporal_pre,
            spatial_downsample_factor=args.spatial_downsample_factor,
            cylinder_radii=tuple(args.cylinder_radii),
            cylinder_height_scale=args.cylinder_height_scale,
            cylinder_vertical_mode=args.cylinder_vertical_mode,
            projector_edge_feather=args.projector_edge_feather,
            projector_angular_power=args.projector_angular_power,
            projector_blend_mode=args.projector_blend_mode,
            ego_coordinate_mode=args.ego_coordinate_mode,
            decode_camera_from_cylinder=args.decode_camera_from_cylinder,
            use_view_residual=False,
            gradient_checkpoint_encode=args.gradient_checkpoint,
            gradient_checkpoint_decode=args.gradient_checkpoint,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        rank0_print(ddp_info, f"[model] CrossView4DVAE: {n_params/1e6:.2f}M params")

        # ------------------------------------------------------------------
        # Loss / optim
        # ------------------------------------------------------------------
        loss_fn = VAELoss(VAELossConfig(
            rec_kind=args.rec_kind,
            kl_weight=args.kl_weight,
            kl_reduction=args.kl_reduction,
            kl_warmup_steps=args.kl_warmup,
            perceptual_weight=args.perceptual_weight,
            perceptual_batch_size=args.perceptual_batch_size,
            logvar_reg_weight=args.logvar_reg_weight,
        )).to(device)
        optim = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: lr_lambda(s, args.warmup_steps))

        # ------------------------------------------------------------------
        # Resume
        # ------------------------------------------------------------------
        start_step = 0
        if args.resume and os.path.exists(args.resume):
            rank0_print(ddp_info, f"[resume] loading {args.resume}")
            ck = torch.load(args.resume, map_location="cpu")
            load_model_state(model, ck["model"], strict=False)
            if ck.get("optimizer") is not None:
                optim.load_state_dict(ck["optimizer"])
            if ck.get("scheduler") is not None:
                scheduler.load_state_dict(ck["scheduler"])
            start_step = ck.get("step", 0)
            loss_fn._step = start_step

        if ddp_info["distributed"]:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[ddp_info["local_rank"]] if device.type == "cuda" else None,
                output_device=ddp_info["local_rank"] if device.type == "cuda" else None,
                find_unused_parameters=args.find_unused_parameters,
            )

        # ------------------------------------------------------------------
        # Train loop
        # ------------------------------------------------------------------
        autocast_dtype = amp_dtype(args.amp)
        use_amp = autocast_dtype is not None
        scaler = torch.cuda.amp.GradScaler() if (
            use_amp and autocast_dtype == torch.float16 and device.type == "cuda") else None

        model.train()
        step = start_step
        epoch = start_step // max(len(loader), 1)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = iter(loader)
        log_buf = {}

        t_last = time.time()
        while step < args.steps:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)

            x = batch["vae_images"].to(device, non_blocking=True)  # [B, T, V, 3, H, W]
            K = batch.get("camera_intrinsics")
            E = batch.get("camera_transforms")
            intr_hw = batch.get("intrinsics_hw")
            if args.use_camera_params and K is not None and E is not None:
                K = K.to(device, non_blocking=True)
                E = E.to(device, non_blocking=True)
            else:
                K = None
                E = None

            optim.zero_grad(set_to_none=True)

            ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if use_amp else torch.enable_grad()
            )
            with ctx:
                render_camera = args.reconstruction_domain in ("camera", "both")
                output = model(
                    x,
                    sample_posterior=args.sample_posterior,
                    intrinsics=K,
                    extrinsics=E,
                    intrinsics_hw=intr_hw,
                    render_camera=render_camera,
                )
                posterior = output["posterior"]
                recon_camera = output["sample"] if render_camera else None
                pred_cylinder = output["cylinder_sample"]
                if render_camera and recon_camera.shape != x.shape:
                    raise RuntimeError(
                        "VAE reconstruction shape does not match input: "
                        f"recon={tuple(recon_camera.shape)} input={tuple(x.shape)}. "
                        "Check sequence_length, view counts, and spatial "
                        "downsample settings."
                    )

                zero = pred_cylinder.new_zeros(())
                camera_rec = zero
                cylinder_rec = zero
                camera_edge = zero
                cylinder_edge = zero
                shared_camera_rec = zero
                cylinder_mask_mean = zero
                cylinder_mask_zero = zero
                seam_loss = zero
                perceptual = zero

                if args.reconstruction_domain in ("camera", "both"):
                    camera_rec = reconstruction_loss(
                        recon_camera, x, args.rec_kind)
                    shared_sample = output.get("shared_sample")
                    if shared_sample is not None and shared_sample.shape == x.shape:
                        shared_camera_rec = reconstruction_loss(
                            shared_sample, x, args.rec_kind)
                    if args.perceptual_weight > 0:
                        perceptual = loss_fn._perceptual(recon_camera, x)
                    if args.edge_loss_weight > 0:
                        camera_edge = gradient_reconstruction_loss(
                            recon_camera, x)

                if args.reconstruction_domain in ("cylinder", "both"):
                    raw_model = model.module if hasattr(model, "module") else model
                    with torch.no_grad():
                        target_cylinder = raw_model.virtual_projector.project_layers(
                            x, K, E, intr_hw)
                        if args.masked_cylinder_loss:
                            cylinder_mask = raw_model.virtual_projector.project_layers(
                                torch.ones_like(x[:, :, :, :1]), K, E, intr_hw)
                            cylinder_mask = (
                                cylinder_mask > args.cylinder_mask_threshold
                            ).to(dtype=pred_cylinder.dtype)
                        else:
                            cylinder_mask = None
                    if pred_cylinder.shape != target_cylinder.shape:
                        raise RuntimeError(
                            "Cylinder reconstruction shape does not match "
                            "projected target: "
                            f"pred={tuple(pred_cylinder.shape)} "
                            f"target={tuple(target_cylinder.shape)}."
                        )
                    if cylinder_mask is None:
                        cylinder_rec = reconstruction_loss(
                            pred_cylinder, target_cylinder, args.rec_kind)
                    else:
                        cylinder_rec = masked_reconstruction_loss(
                            pred_cylinder, target_cylinder, cylinder_mask,
                            args.rec_kind)
                        cylinder_mask_mean = cylinder_mask.detach().mean()
                        cylinder_mask_zero = (
                            cylinder_mask.detach() < 0.5).float().mean()
                    if args.cylinder_edge_loss_weight > 0:
                        cylinder_edge = gradient_reconstruction_loss(
                            pred_cylinder,
                            target_cylinder,
                            cylinder_mask if args.masked_cylinder_loss else None,
                        )
                    if args.cylinder_seam_weight > 0:
                        seam_loss = cylinder_seam_loss(pred_cylinder)

                rec = (
                    args.camera_loss_weight * camera_rec
                    + args.cylinder_loss_weight * cylinder_rec
                )
                kl = loss_fn._kl(posterior)
                kl_w = loss_fn._kl_weight()
                logvar_reg = zero
                if args.logvar_reg_weight > 0:
                    logvar_reg = posterior.logvar.pow(2).mean()
                loss = (
                    rec
                    + args.camera_loss_weight *
                    args.perceptual_weight * perceptual
                    + args.camera_loss_weight *
                    args.edge_loss_weight * camera_edge
                    + args.cylinder_loss_weight *
                    args.cylinder_edge_loss_weight * cylinder_edge
                    + kl_w * kl
                    + args.logvar_reg_weight * logvar_reg
                    + args.cylinder_seam_weight * seam_loss
                )
                loss_fn._step += 1
                metrics = {
                    "loss": loss,
                    "rec": rec.detach(),
                    "camera_rec": camera_rec.detach(),
                    "camera_edge": camera_edge.detach(),
                    "shared_camera_rec": shared_camera_rec.detach(),
                    "cylinder_rec": cylinder_rec.detach(),
                    "cylinder_edge": cylinder_edge.detach(),
                    "cylinder_mask_mean": cylinder_mask_mean.detach(),
                    "cylinder_mask_zero": cylinder_mask_zero.detach(),
                    "cylinder_seam": seam_loss.detach(),
                    "perceptual": perceptual.detach(),
                    "kl": kl.detach(),
                    "kl_weight_now": pred_cylinder.new_tensor(kl_w),
                    "logvar_reg": logvar_reg.detach(),
                    "logvar_mean": posterior.logvar.detach().mean(),
                    "mean_abs_z": posterior.mean.detach().abs().mean(),
                }

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
            scheduler.step()
            step += 1

            # ---- logging ----
            reduced_metrics = reduce_metric_dict(metrics, ddp_info)
            for k, v in reduced_metrics.items():
                log_buf.setdefault(k, []).append(float(v.item()) if hasattr(v, "item") else float(v))

            if step % args.log_every == 0:
                if is_main_process(ddp_info):
                    avg = {k: sum(vs) / len(vs) for k, vs in log_buf.items()}
                    dt = time.time() - t_last
                    t_last = time.time()
                    lr_now = optim.param_groups[0]["lr"]
                    it_s = args.log_every / max(dt, 1e-6)
                    append_metrics(metrics_path, {
                        "step": step,
                        "steps_total": args.steps,
                        "loss": avg["loss"],
                        "rec": avg["rec"],
                        "camera_rec": avg.get("camera_rec", 0.0),
                        "camera_edge": avg.get("camera_edge", 0.0),
                        "shared_camera_rec": avg.get("shared_camera_rec", 0.0),
                        "cylinder_rec": avg.get("cylinder_rec", 0.0),
                        "cylinder_edge": avg.get("cylinder_edge", 0.0),
                        "cylinder_mask_mean": avg.get("cylinder_mask_mean", 0.0),
                        "cylinder_mask_zero": avg.get("cylinder_mask_zero", 0.0),
                        "cylinder_seam": avg.get("cylinder_seam", 0.0),
                        "perceptual": avg.get("perceptual", 0.0),
                        "kl": avg["kl"],
                        "kl_weight_now": avg["kl_weight_now"],
                        "weighted_kl": avg["kl"] * avg["kl_weight_now"],
                        "logvar_reg": avg.get("logvar_reg", 0.0),
                        "mean_abs_z": avg["mean_abs_z"],
                        "logvar_mean": avg["logvar_mean"],
                        "lr": lr_now,
                        "it_s": it_s,
                    })
                    print(
                        f"[step {step:6d}/{args.steps}] "
                        f"loss={avg['loss']:.4f} rec={avg['rec']:.4f} "
                        f"cam={avg.get('camera_rec', 0.0):.4f} "
                        f"cyl={avg.get('cylinder_rec', 0.0):.4f} "
                        f"edge={avg.get('camera_edge', 0.0):.4f}/"
                        f"{avg.get('cylinder_edge', 0.0):.4f} "
                        f"perc={avg.get('perceptual', 0.0):.4f} "
                        f"mask0={avg.get('cylinder_mask_zero', 0.0):.2f} "
                        f"seam={avg.get('cylinder_seam', 0.0):.4f} "
                        f"kl={avg['kl']:.2f} (w={avg['kl_weight_now']:.1e}, "
                        f"wkl={avg['kl'] * avg['kl_weight_now']:.4f}) "
                        f"|z|={avg['mean_abs_z']:.3f} logvar~{avg['logvar_mean']:.2f} "
                        f"lr={lr_now:.2e}  ({it_s:.1f} it/s)"
                    )
                log_buf.clear()

            # ---- preview ----
            if is_main_process(ddp_info) and step % args.preview_every == 0:
                with torch.no_grad():
                    model.eval()
                    raw_model = model.module if hasattr(model, "module") else model
                    preview_out = raw_model(
                        x,
                        sample_posterior=False,
                        intrinsics=K,
                        extrinsics=E,
                        intrinsics_hw=intr_hw,
                        render_camera=True,
                    )
                    recon_eval = preview_out["sample"]
                    save_preview(
                        x.detach(), recon_eval.detach(),
                        str(preview_dir / f"step_{step:06d}.png"))
                    model.train()

            # ---- ckpt ----
            if is_main_process(ddp_info) and (
                step % args.ckpt_every == 0 or step == args.steps
            ):
                save_checkpoint(
                    model, optim, scheduler, step, args,
                    str(ckpt_dir / f"step_{step:06d}.pt"))
                save_checkpoint(
                    model, optim, scheduler, step, args,
                    str(ckpt_dir / "last.pt"))

        rank0_print(ddp_info, "[done] training finished")
    finally:
        cleanup_distributed(ddp_info)


if __name__ == "__main__":
    main()
