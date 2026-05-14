"""Standalone training entry point for ``CrossView4DVAE``.

Designed to be lean: pure PyTorch, no Lightning / Accelerate, so the bug
surface is small and debugging is straightforward. All the moving parts
(model, data, loss) are imported from this folder.

Usage examples
--------------

# Synthetic data, single GPU, default config (good for first sanity-train).
python -m zhw_vae_510.train_vae --data synthetic --steps 2000 --image-hw 64 128

# Multi-GPU DDP. Per-rank batch size is still --batch-size.
torchrun --nproc_per_node=4 -m zhw_vae_510.train_vae --data synthetic \
    --steps 2000 --image-hw 64 128 --batch-size 2

# Real nuPlan data (paths must match those in dwm.tools.dataset_nus).
python -m zhw_vae_510.train_vae --data nuplan --steps 50000 --image-hw 256 448 \
    --batch-size 1 --base-channels 64

# Resume from a checkpoint.
python -m zhw_vae_510.train_vae --data nuplan --resume zhw_vae_510/runs/last.pt

The script intentionally writes nothing outside ``zhw_vae_510/`` so it's safe
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
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Make ``dwm`` importable for the dataset adapters that pull in
# dwm.datasets.nuscenes / nuplan.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The VAE model itself lives next to this script.
from zhw_vae_510.crossview_vae import CrossView4DVAE  # noqa: E402
from zhw_vae_510.data import (  # noqa: E402
    MultiViewVAEAdapter,
    SyntheticCylinderDataset,
    SyntheticConfig,
    make_nuscenes_base,
    vae_collate,
)
from zhw_vae_510.losses import VAELoss, VAELossConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "nuplan", "nuscenes"],
                   default="synthetic", help="Dataset source.")
    p.add_argument("--out", default="zhw_vae_510/runs",
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
                        "count'. Pass an explicit value (e.g. 4) ONLY if "
                        "you know how to provide a target with that many "
                        "views (e.g. by pre-projecting GT yourself).")
    p.add_argument("--latent-view-count", type=int, default=3,
                   help="Number of latent views (V_lat). Must be "
                        "<= virtual_view_count. 3 covers 360deg in three "
                        "120deg sectors which keeps decent horizontal "
                        "angular resolution while still halving the view "
                        "dim of a 6-camera ring.")
    p.add_argument("--num-attention-heads", type=int, default=2)
    p.add_argument("--num-bottleneck-blocks", type=int, default=1)
    p.add_argument("--temporal-downsample-factor", type=int, default=4)
    p.add_argument("--temporal-pre", type=int, default=1)
    p.add_argument("--spatial-downsample-factor", type=int, default=8)
    p.add_argument("--cylinder-radii", type=float, nargs="+", default=[10.0])

    # loss
    p.add_argument("--rec-kind", default="l1", choices=["l1", "l2", "huber"])
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--kl-warmup", type=int, default=2000)
    p.add_argument("--perceptual-weight", type=float, default=0.0,
                   help="Set >0 to enable LPIPS (requires `pip install lpips`).")

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
                   help="Pass intrinsics/extrinsics to encode() (geometry-aware "
                        "projection). Otherwise the VAE uses its fallback.")
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

    if args.data == "nuscenes":
        if not args.nusc_data_root:
            raise ValueError("--nusc-data-root is required when --data=nuscenes")
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

    # nuplan: import locally so synthetic runs have no nuplan deps.
    from dwm.tools.dataset_nus import make_base_ds
    base = make_base_ds(train=True)
    return MultiViewVAEAdapter(
        base, sequence_length=args.sequence_length,
        image_hw=tuple(args.image_hw))


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
    ddp_info = init_distributed(args)
    try:
        device = ddp_info["device"]

        out_dir = Path(args.out)
        preview_dir = out_dir / "previews"
        ckpt_dir = out_dir / "ckpts"
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
            f"world_size={ddp_info['world_size']}"
        )

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        dataset = make_dataset(args, ddp_info)
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_info["world_size"],
            rank=ddp_info["rank"],
            shuffle=True,
            drop_last=True,
        ) if ddp_info["distributed"] else None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers, collate_fn=vae_collate,
            drop_last=True, pin_memory=device.type == "cuda")

        # Probe one sample to learn V_input. Reconstruction loss requires
        # ``virtual_view_count == V_input`` so VAE output and GT have matching
        # view dim.
        probe = dataset[0]
        v_input = int(probe["vae_images"].shape[1])
        if args.virtual_view_count <= 0:
            args.virtual_view_count = v_input
            rank0_print(ddp_info, f"[setup] auto virtual_view_count = V_input = {v_input}")
        elif args.virtual_view_count != v_input:
            rank0_print(
                ddp_info,
                f"[warn] virtual_view_count ({args.virtual_view_count}) != "
                f"V_input ({v_input}). Reconstruction loss is going to fail "
                f"unless you pre-project GT yourself. Aborting.")
            raise SystemExit(1)
        if args.latent_view_count > args.virtual_view_count:
            rank0_print(
                ddp_info,
                f"[warn] latent_view_count ({args.latent_view_count}) must be "
                f"<= virtual_view_count ({args.virtual_view_count}). Aborting.")
            raise SystemExit(1)

        # Log args after auto-filled values (e.g. virtual_view_count) are resolved.
        if is_main_process(ddp_info):
            with open(out_dir / "args.json", "w") as f:
                json.dump(vars(args), f, indent=2)

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        model = CrossView4DVAE(
            base_channels=args.base_channels,
            latent_channels=args.latent_channels,
            virtual_view_count=args.virtual_view_count,
            latent_view_count=args.latent_view_count,
            num_attention_heads=args.num_attention_heads,
            num_bottleneck_blocks=args.num_bottleneck_blocks,
            temporal_downsample_factor=args.temporal_downsample_factor,
            temporal_pre=args.temporal_pre,
            spatial_downsample_factor=args.spatial_downsample_factor,
            cylinder_radii=tuple(args.cylinder_radii),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        rank0_print(ddp_info, f"[model] CrossView4DVAE: {n_params/1e6:.2f}M params")

        # ------------------------------------------------------------------
        # Loss / optim
        # ------------------------------------------------------------------
        loss_fn = VAELoss(VAELossConfig(
            rec_kind=args.rec_kind,
            kl_weight=args.kl_weight,
            kl_warmup_steps=args.kl_warmup,
            perceptual_weight=args.perceptual_weight,
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
                K = None; E = None

            optim.zero_grad(set_to_none=True)

            ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if use_amp else torch.enable_grad()
            )
            with ctx:
                output = model(
                    x, intrinsics=K, extrinsics=E, intrinsics_hw=intr_hw,
                )
                posterior = output["posterior"]
                recon = output["sample"]
                # Match shapes (decode may produce more frames than input due to
                # temporal_pre + k*tdf padding).
                t_min = min(recon.shape[1], x.shape[1])
                metrics = loss_fn(recon[:, :t_min], x[:, :t_min], posterior)
                loss = metrics["loss"]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim); scaler.update()
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
                    print(
                        f"[step {step:6d}/{args.steps}] "
                        f"loss={avg['loss']:.4f} rec={avg['rec']:.4f} "
                        f"kl={avg['kl']:.2f} (w={avg['kl_weight_now']:.1e}) "
                        f"|z|={avg['mean_abs_z']:.3f} logvar~{avg['logvar_mean']:.2f} "
                        f"lr={lr_now:.2e}  ({args.log_every / max(dt, 1e-6):.1f} it/s)"
                    )
                log_buf.clear()

            # ---- preview ----
            if is_main_process(ddp_info) and step % args.preview_every == 0:
                with torch.no_grad():
                    model.eval()
                    raw_model = model.module if hasattr(model, "module") else model
                    z_eval = posterior.mode().to(dtype=raw_model.from_latent.weight.dtype)
                    recon_eval = raw_model.decode(z_eval).sample
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
